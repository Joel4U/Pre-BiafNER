import argparse
from src.config import Config, evaluate_batch_insts
import time
from src.model import TransformersCRF
import torch
import torch.nn as nn
from src.config.transformers_util import get_huggingface_optimizer_and_scheduler#, get_huggingface_optimizer_and_scheduler_old
from tqdm import tqdm
from src.data import TransformersNERDataset, NERDataset, batch_iter, batch_variable
from src.data.data_utils import detect_overlapping_level
from src.config.metrics import precision_recall_f1_report
from torch.utils.data import DataLoader
from transformers import set_seed, AutoTokenizer
from logger import get_logger
from termcolor import colored
import math
import os
import sys
sys.path.append('/home/wp/SynDepBiaffine')
os.environ["TOKENIZERS_PARALLELISM"] = "true"

logger = get_logger()

def parse_arguments(parser):
    ###Training Hyperparameters
    parser.add_argument('--device', type=str, default="cuda:1", choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2'], help="GPU/CPU devices")
    parser.add_argument('--seed', type=int, default=44, help="random seed")
    parser.add_argument('--dataset', type=str, default="english")
    parser.add_argument('--batch_size', type=int, default=20, help="default batch size is 10 (works well for normal neural crf), here default 30 for bert-based crf")
    parser.add_argument('--num_epochs', type=int, default=100, help="Usually we set to 100.")
    parser.add_argument('--train_with_dev', type=bool, default=False, help="whether to train with development set")
    parser.add_argument('--max_no_incre', type=int, default=40, help="early stop when there is n epoch not increasing on dev")
    parser.add_argument('--max_grad_norm', type=float, default=1, help="The maximum gradient norm, if <=0, means no clipping, usually we don't use clipping for normal neural ncrf")
    parser.add_argument('--fp16', type=int, choices=[0, 1], default=0, help="use 16-bit floating point precision instead of 32-bit")
    ##model hyperparameter
    parser.add_argument('--other_lr', type=str, default=1e-3, help="between 1e-3 and 3e-3 on the randomly initialized weight")
    parser.add_argument('--pretr_lr', type=float, default=2e-5, help="between 8e-6 and 3e-5 working on the pretrained weights, such as: roberta is , electra is ")
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--context_outputsize', type=int, default=200, help="hidden size of the LSTM, usually we set to 200 for LSTM-CRF")
    parser.add_argument('--affine_outputsize', type=int, default=150, help="hidden size of the Syn-LSTM")
    parser.add_argument('--activation', type=str, default="ReLU", help="LeakyReLU, ReLU, ELU")
    parser.add_argument('--sb_epsilon', type=float, default=0.1, help="Boundary smoothing loss epsilon")

    parser.add_argument('--dropout', type=float, default=0.5, help="dropout for SynLSTM or LSTM")
    parser.add_argument('--embedder_type', type=str, default="bert-large-cased", help="you can use 'chinese-electra-180g-base-discriminator，roberta-base' and so on")
    parser.add_argument('--is_freezing', type=bool, default=False, help="you can freeze the word embedder")

    parser.add_argument("--earlystop_atr", type=str, default="micro", choices= ["micro", "macro"], help= "Choose between macro f1 score and micro f1 score for early stopping evaluation")
    parser.add_argument('--enc_type', type=str, default="synlstm", choices=["lstm", "adatrans", "synlstm", "naivetrans"])
    parser.add_argument('--parser_mode', type=str, default="span", choices=["crf", "span"], help="parser model consists of crf and span")

    args = parser.parse_args()
    for k in args.__dict__:
        logger.info(k + ": " + str(args.__dict__[k]))
    return args

# def train_model(config: Config, epoch: int, train_loader: DataLoader, dev_loader: DataLoader, test_loader: DataLoader):
def train_model(config, epoch, train_data, dev_data, test_data):
    train_num = len(train_data)
    # train_num = len(train_loader)
    print(f"[Data Info] number of training instances: {train_num}")
    print(colored(f"[Model Info]: Working with transformers package from huggingface with {config.embedder_type}", 'red'))
    model = TransformersCRF(config)
    # optimizer, scheduler = get_huggingface_optimizer_and_scheduler_old(model=model, plm_lr=config.plm_lr, lr=config.lr,
    #                                                                num_training_steps=train_num * epoch,
    #                                                                weight_decay=config.weight_decay, eps = 1e-8, warmup_step=train_num*max(2, epoch // 5))
    optimizer, scheduler = get_huggingface_optimizer_and_scheduler(model=model, pretr_lr=config.pretr_lr, other_lr=config.other_lr,
                                                                   num_training_steps=train_num * epoch,#num_training_steps=train_num // config.batch_size * epoch,
                                                                   weight_decay=1e-6, eps=1e-8,# warmup_step=train_num*max(2, epoch // 5)) 
                                                                   warmup_step=int(0.2 * train_num // config.batch_size * epoch))

    # print(colored(f"[Optimizer Info] Modify the optimizer info as you need.", 'red'))
    print(optimizer)
    model.to(config.device)
    best_dev = [-1, 0]
    best_test = [-1, 0]
    no_incre_dev = 0
    print(colored(f"[Train Info] Start training, you have set to stop if performace not increase for {config.max_no_incre} epochs",'red'))
    for i in tqdm(range(1, epoch + 1), desc="Epoch"):
        epoch_loss = 0
        start_time = time.time()
        model.train()
        # for iter, batch in enumerate(train_loader, 1):
        for iter, batch_data in enumerate(batch_iter(train_data, config.batch_size, True)):
            batcher = batch_variable(batch_data, config)
            with torch.cuda.amp.autocast(enabled=False):
                # loss = model(subword_input_ids = batch.input_ids.to(config.device), word_seq_lens = batch.word_seq_len.to(config.device),
                # orig_to_tok_index = batch.orig_to_tok_index.to(config.device), attention_mask = batch.attention_mask.to(config.device),
                # tag_ids = batch.tag_ids.to(config.device),
                # depheads=batch.dephead_ids.to(config.device), deplabels=batch.deplabel_ids.to(config.device),
                # span_label_ids=batch.span_label_ids.to(config.device))
                loss = model(batcher["input_ids"], batcher["word_seq_lens"], batcher["orig_to_tok_index"], batcher["attention_mask"],
                             batcher["pos_ids"], batcher["dephead_ids"], batcher["deplabel_ids"], batcher["spanlabel_ids"])
            epoch_loss += loss.item()
            loss.backward()
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        end_time = time.time()
        logger.info(f"Epoch {i}, PLM_lr: {scheduler.get_last_lr()[0]:.4e}，Other_lr: {scheduler.get_last_lr()[2]:.4e}, "
                    f"epoch_loss: {epoch_loss:.5f}, Time is {(end_time - start_time):.2f}s")

        model.eval() ## evaluation
        # if dev_loader is not None:
        if dev_data is not None:
            # dev_metrics = evaluate_model(config, model, dev_loader, "dev")
            dev_metrics = evaluate_model(config, model, dev_data, "dev")
        # test_metrics = evaluate_model(config, model, test_loader, "test")
        test_metrics = evaluate_model(config, model, test_data, "test")
        if test_metrics[2] > best_test[0]:
            no_incre_dev = 0
            # best_dev[0] = dev_metrics[2]
            # best_dev[1] = i
            best_test[0] = test_metrics[2]
            best_test[1] = i
        else:
            no_incre_dev += 1
        model.zero_grad()
        if no_incre_dev >= config.max_no_incre:
            print("early stop because there are %d epochs not increasing f1 on dev"%no_incre_dev)
            break

def evaluate_model(config, model, dataset, name):
# def evaluate_model(config: Config, model: TransformersCRF, data_loader: DataLoader, name: str):
    set_y_pred = []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        # for iter, batch in enumerate(data_loader, 0):
        for iter, batch_data in enumerate(batch_iter(dataset, config.batch_size, False)):
            batcher = batch_variable(batch_data, config)
            batch_y_pred = model(batcher["input_ids"], batcher["word_seq_lens"], batcher["orig_to_tok_index"], batcher["attention_mask"], 
                                 batcher["pos_ids"], batcher["dephead_ids"], batcher["deplabel_ids"], batcher["spanlabel_ids"], is_train=False)
            # batch_y_pred = model(subword_input_ids=batch.input_ids.to(config.device), word_seq_lens=batch.word_seq_len.to(config.device),
            #     orig_to_tok_index=batch.orig_to_tok_index.to(config.device), attention_mask=batch.attention_mask.to(config.device),
            #     tag_ids = batch.tag_ids.to(config.device),
            #     depheads=batch.dephead_ids.to(config.device), deplabels=batch.deplabel_ids.to(config.device),
            #     span_label_ids=batch.span_label_ids.to(config.device), is_train=False)
            set_y_pred.extend(batch_y_pred)
        # set_y_gold = [inst.span_labels for inst in data_loader.dataset.insts]
        set_y_gold = [inst.span_labels for inst in dataset.insts]
        _, ave_scores = precision_recall_f1_report(set_y_gold, set_y_pred)
        precise = ave_scores['micro']['precision']
        recall = ave_scores['micro']['recall']
        fscore = ave_scores['micro']['f1']
        logger.info(f"[{name} set Total] Prec.: { precise * 100:.2f}, Rec.: {recall * 100:.2f}, Micro F1: {fscore * 100:.2f}")
    return [precise, recall, fscore]


def main():
    parser = argparse.ArgumentParser(description="SpanSynLSTM implementation")
    opt = parse_arguments(parser)
    set_seed(opt.seed)
    conf = Config(opt)
    logger.info(f"[Data Info] Tokenizing the instances using '{conf.embedder_type}' tokenizer")
    conf.tokenizer = AutoTokenizer.from_pretrained(conf.embedder_type, use_fast=True)
    print(colored(f"[Data Info] Reading dataset from: \t{conf.train_file}\t{conf.dev_file}\t{conf.test_file}", "yellow"))
    # train_dataset = TransformersNERDataset(conf.parser_mode, 0, conf.train_file, conf.sb_epsilon, tokenizer, is_train=True, is_json=False)
    train_dataset = NERDataset(conf.train_file, conf.enc_type, conf.tokenizer, conf.sb_epsilon)
    if conf.enc_type == 'synlstm':
        conf.deplabel2idx = train_dataset.deplabel2idx
        conf.deplabel_size = len(train_dataset.deplabel2idx)
        conf.root_dep_label_id = train_dataset.root_dep_label_id
    conf.pos_size = len(train_dataset.pos2idx)
    conf.label2idx = train_dataset.label2idx
    conf.idx2label = train_dataset.idx2labels
    conf.label_size = len(train_dataset.label2idx)

    dev_dataset = NERDataset(conf.dev_file, conf.enc_type, conf.tokenizer, conf.sb_epsilon,
                             deplabel2idx=train_dataset.deplabel2idx, pos2idx=train_dataset.pos2idx, label2idx=train_dataset.label2idx, is_train=False)
    test_dataset = NERDataset(conf.test_file, conf.enc_type, conf.tokenizer, conf.sb_epsilon,
                              deplabel2idx=train_dataset.deplabel2idx, pos2idx=train_dataset.pos2idx, label2idx=train_dataset.label2idx, is_train=False)
    conf.max_entity_length = max(max(train_dataset.max_entity_length, dev_dataset.max_entity_length), test_dataset.max_entity_length)
    conf.max_seq_length = max(max(train_dataset.get_max_token_len(), dev_dataset.get_max_token_len()), test_dataset.get_max_token_len())
    all_insts = train_dataset.insts + dev_dataset.insts + test_dataset.insts
    conf.overlapping_level = max(detect_overlapping_level(inst.span_labels) for inst in all_insts)
    train_model(conf, conf.num_epochs, train_dataset, dev_dataset, test_dataset)
    # return
    # if conf.enc_type == 'synlstm':
    #     conf.root_dep_label_id = train_dataset.root_dep_label_id
    #     dev_dataset = TransformersNERDataset(conf.parser_mode, 0, conf.dev_file, conf.sb_epsilon, tokenizer,
    #                                          label2idx=train_dataset.label2idx, deplabel2idx=train_dataset.deplabel2idx, pos2idx=train_dataset.pos2idx, is_train=False)
    #     test_dataset = TransformersNERDataset(conf.parser_mode, 0, conf.test_file, conf.sb_epsilon, tokenizer, 
    #                                           label2idx=train_dataset.label2idx, deplabel2idx=train_dataset.deplabel2idx, pos2idx=train_dataset.pos2idx, is_train=False)
    # else:
    #     dev_dataset = TransformersNERDataset(conf.parser_mode, 0, conf.dev_file, conf.sb_epsilon, tokenizer, 
    #                                          label2idx=train_dataset.label2idx, deplabel2idx=None, is_train=False, pos2idx=train_dataset.pos2idx, is_json=False)
    #     test_dataset = TransformersNERDataset(conf.parser_mode, 0, conf.test_file, conf.sb_epsilon, tokenizer, 
    #                                           label2idx=train_dataset.label2idx, deplabel2idx=None, is_train=False, pos2idx=train_dataset.pos2idx, is_json=False)
    # num_workers = 8
    # conf.max_seq_length = max(max(train_dataset.get_max_token_len(), dev_dataset.get_max_token_len()), test_dataset.get_max_token_len())
    # if conf.train_with_dev:
    #     train_dataloader = DataLoader(train_dataset + dev_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=num_workers,
    #                                     collate_fn=train_dataset.collate_to_max_length)
    #     dev_dataloader = None
    #     test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=num_workers,
    #                                     collate_fn=test_dataset.collate_to_max_length)
    # else:
    #     train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=num_workers,
    #                                     collate_fn=train_dataset.collate_to_max_length)
    #     dev_dataloader = DataLoader(dev_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=num_workers,
    #                                     collate_fn=dev_dataset.collate_to_max_length)
    #     test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=num_workers,
    #                                     collate_fn=test_dataset.collate_to_max_length)

    # conf.max_entity_length = max(max(train_dataset.max_entity_length, dev_dataset.max_entity_length), test_dataset.max_entity_length)
    # all_insts = train_dataset.insts + dev_dataset.insts + test_dataset.insts
    # conf.overlapping_level = max(detect_overlapping_level(inst.span_labels) for inst in all_insts)
    # train_model(conf, conf.num_epochs, train_dataloader, dev_dataloader, test_dataloader)


if __name__ == "__main__":
    main()
