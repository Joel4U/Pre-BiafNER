#
# @author: Allan
#

import torch
import torch.nn as nn
import numpy as np

from src.model.module.bilstm_encoder import BiLSTMEncoder
from src.model.module.linear_encoder import LinearEncoder
from src.model.embedder import TransformersEmbedder
from src.model.module.deplabel_gcn import DepLabeledGCN
from src.model.module.biaffine_decoder import BiaffineDecoder
from src.model.module.transformer_enc import TransformerEncoder
from typing import Tuple, Union
from src.config.config import DepModelType
from src.data.data_utils import head_to_adj, head_to_adj_label, _spans_from_upper_triangular, filter_clashed_by_priority
from src.config.transformers_util import _check_soft_target


def timestep_dropout(inputs, p=0.5, batch_first=True):
    '''
    :param inputs: (bz, time_step, feature_size)
    :param p: probability p mask out output nodes
    :param batch_first: default True
    :return:
    '''
    if not batch_first:
        inputs = inputs.transpose(0, 1)

    batch_size, time_step, feature_size = inputs.size()
    drop_mask = inputs.data.new_full((batch_size, feature_size), 1-p)
    drop_mask = torch.bernoulli(drop_mask).div(1 - p)
    # drop_mask = drop_mask.unsqueeze(-1).expand((-1, -1, time_step)).transpose(1, 2)
    drop_mask = drop_mask.unsqueeze(1)
    return inputs * drop_mask

class TransformersCRF(nn.Module):
    def __init__(self, config):
        super(TransformersCRF, self).__init__()
        self.transformer = TransformersEmbedder(transformer_model_name=config.embedder_type, is_freezing=config.is_freezing)
        self.tag_embedding = nn.Embedding(num_embeddings=config.pos_size, embedding_dim=48, padding_idx=0)
        self.transformer_drop = nn.Dropout(config.dropout)
        # self.dep_model = config.dep_model
        self.idx2label = config.idx2label
        self.sb_epsilon = config.sb_epsilon
        self.overlapping_level = config.overlapping_level
        self.enc_type = config.enc_type
        if config.enc_type == 'synlstm':
            self.deplabel_embedding = nn.Embedding(config.deplabel_size, embedding_dim=48, padding_idx=0)
            self.root_dep_label_id = config.root_dep_label_id
            self.context_encoder = DepLabeledGCN(config, config.context_outputsize * 2, self.transformer.get_output_dim()+96,  self.transformer.get_output_dim()+48)  ### lstm hidden size
        elif config.enc_type == 'adatrans' or config.enc_type == 'naivetrans':
            self.context_encoder = TransformerEncoder(d_model=self.transformer.get_output_dim() + 48, num_layers=3, n_head=8,
                                              feedforward_dim=2 * (self.transformer.get_output_dim() + 48), attn_type=config.enc_type, dropout=0.33)
            self.linear = nn.Linear(in_features=self.transformer.get_output_dim() + 48, out_features=config.context_outputsize * 2)
        elif config.enc_type == 'lstm':
            # self.context_encoder = nn.Linear(self.transformer.get_output_dim(), config.context_outputsize)
            self.context_encoder = BiLSTMEncoder(input_dim=self.transformer.get_output_dim() + 48,
                                        hidden_dim=config.context_outputsize, drop_lstm=config.dropout)
        
        self.biaffine_decoder = BiaffineDecoder(config)
        _span_size_ids = torch.arange(config.max_seq_length) - torch.arange(config.max_seq_length).unsqueeze(-1)
        self._span_non_mask = (_span_size_ids >= 0).to(torch.bool)
        # self.biaffine_decoder = BiaffineDecode(self.transformer.get_output_dim(), ffnn_size=config.affine_outputsize, num_cls=config.label_size, ffnn_drop=0.2)

    def forward(self, subword_input_ids: torch.Tensor,  word_seq_lens: torch.Tensor,
                    orig_to_tok_index: torch.Tensor, attention_mask: torch.Tensor, tag_ids: torch.Tensor,
                    depheads: torch.Tensor,  deplabels: torch.Tensor,
                    span_label_ids: torch.Tensor, is_train: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        bz, sent_len = orig_to_tok_index.size()
        max_seq_len = word_seq_lens.max()

        maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long, device=subword_input_ids.device).view(1, sent_len).expand(bz, sent_len)
        non_pad_mask = torch.le(maskTemp, word_seq_lens.view(bz, 1).expand(bz, sent_len))

        word_emb = self.transformer(subword_input_ids, orig_to_tok_index, attention_mask)
        tags_emb = self.tag_embedding(tag_ids)
        word_rep = torch.cat((word_emb, tags_emb), dim=-1).contiguous()
        if self.enc_type == 'synlstm':
            rel_emb = self.deplabel_embedding(deplabels)
            word_rep = torch.cat((word_rep, rel_emb), dim=-1).contiguous()
        if is_train:
            word_rep = timestep_dropout(word_rep, 0.5)
        if self.enc_type == 'synlstm':
            adj_matrixs = [head_to_adj(max_seq_len, orig_to_tok_index[i], depheads[i]) for i in range(bz)]
            adj_matrixs = np.stack(adj_matrixs, axis=0)
            adj_matrixs = torch.from_numpy(adj_matrixs)
            dep_label_adj = [head_to_adj_label(max_seq_len, orig_to_tok_index[i], depheads[i], deplabels[i], self.root_dep_label_id) for i
                             in range(bz)]
            dep_label_adj = torch.from_numpy(np.stack(dep_label_adj, axis=0)).long()
            en_out = self.context_encoder(word_rep, word_seq_lens, adj_matrixs, dep_label_adj)
        elif self.enc_type == 'lstm':
            en_out = self.context_encoder(word_rep, word_seq_lens)
        elif self.enc_type == 'adatrans' or self.enc_type == 'naivetrans':
            en_out_tmp = self.context_encoder(word_rep, non_pad_mask)
            en_out = self.linear(en_out_tmp)

        # 创建 span_mask，使用广播进行向量化操作
        # maskTemp = torch.arange(max_seq_len, device=span_label_ids.device).expand(bz, max_seq_len)
        # span_mask = (maskTemp < word_seq_lens.unsqueeze(1))             # 生成非填充部分的掩码
        # span_mask = span_mask.unsqueeze(1) & span_mask.unsqueeze(2)     # 扩展维度，创建二维掩码
        biaffine_score = self.biaffine_decoder(en_out)

        if is_train:
            losses = []
            for curr_label_ids, curr_scores, curr_len in zip(span_label_ids, biaffine_score, word_seq_lens.cpu().tolist()):
                curr_non_mask = self._get_span_non_mask(curr_len)
                if self.sb_epsilon <=0:
                    loss = nn.functional.cross_entropy(curr_scores[:curr_len, :curr_len][curr_non_mask], curr_label_ids[:curr_len, :curr_len][curr_non_mask], reduction="sum")
                else:
                    soft_target = curr_label_ids[:curr_len, :curr_len][curr_non_mask]
                    _check_soft_target(soft_target)
                    log_prob = curr_scores[:curr_len, :curr_len][curr_non_mask].log_softmax(dim=-1)
                    loss = -(log_prob * soft_target).sum(dim=-1)
                    loss = loss.sum()
                losses.append(loss)
            return torch.stack(losses).mean()
        else:
            batch_y_pred = []
            for curr_scores, curr_len in zip(biaffine_score, word_seq_lens.cpu().tolist()):
                # curr_non_mask1 = curr_non_mask1[:curr_len, :curr_len]
                curr_non_mask = self._get_span_non_mask(curr_len) 
                confidences, label_ids = curr_scores[:curr_len, :curr_len][curr_non_mask].softmax(dim=-1).max(dim=-1)
                labels = [self.idx2label[i] for i in label_ids.cpu().tolist()]
                chunks = [(label, start, end) for label, (start, end) in zip(labels, _spans_from_upper_triangular(curr_len)) if label != 'O']  # '<none>'
                confidences = [conf for label, conf in zip(labels, confidences.cpu().tolist()) if label != 'O']
                assert len(confidences) == len(chunks)
                # Sort chunks by confidences: high -> low
                chunks = [ck for _, ck in sorted(zip(confidences, chunks), reverse=True)]
                chunks = filter_clashed_by_priority(chunks, allow_level=self.overlapping_level)  # self.overlapping_level: Flat 0, Nested 1, 这里就是对实体排序，选取策略的地方， 返回了个数是动态变化的
                batch_y_pred.append(chunks)
            return batch_y_pred
        
    def _get_span_non_mask(self, seq_len: int):
        return self._span_non_mask[:seq_len, :seq_len]