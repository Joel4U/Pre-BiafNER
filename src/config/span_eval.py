# encoding: utf-8


import torch


def span_f1(predicts,span_label_ltoken,real_span_mask_ltoken):
    '''
    :param predicts: the prediction of model
    :param span_label_ltoken: the label of span
    :param real_span_mask_ltoken: 1 for real span, and 0 for padding span.
    '''
    pred_label_idx = torch.max(predicts, dim=-1)[1] # (bs, n_span)
    pred_label_mask = (pred_label_idx!=0)  # (bs, n_span)
    all_correct = pred_label_idx == span_label_ltoken
    all_correct = all_correct*pred_label_mask*real_span_mask_ltoken.bool()
    correct_pred = torch.sum(all_correct)
    total_pred = torch.sum(pred_label_idx!=0 )
    total_golden = torch.sum(span_label_ltoken!=0)

    return torch.stack([correct_pred, total_pred, total_golden])



def span_f1_prune(all_span_idxs,predicts,span_label_ltoken,real_span_mask_ltoken):
    '''
    :param all_span_idxs: the positon of span;
    :param predicts: the prediction of model;
    :param span_label_ltoken: the label of the span.  SHAPE: (batch_size,n_span)
    :param real_span_mask_ltoken: 1 for real span, and 0 for padding span.
    '''
    pred_label_idx = torch.max(predicts, dim=-1)[1] # (bs, n_span)
    span_probs = predicts.tolist()
    nonO_idxs2labs, nonO_kidxs_all, pred_label_idx_new = get_pruning_predIdxs(pred_label_idx, all_span_idxs, span_probs)
    pred_label_idx = pred_label_idx_new.to(predicts.device)
    pred_label_mask = (pred_label_idx!=0)  # (bs, n_span)
    # all_correct = pred_label_idx == span_label_ltoken
    # all_correct = all_correct*pred_label_mask*real_span_mask_ltoken.bool()
    # correct_pred = torch.sum(all_correct)
    # total_pred = torch.sum(pred_label_idx!=0 )
    # total_golden = torch.sum(span_label_ltoken!=0)

    # 根据 real_span_mask_ltoken 过滤填充部分
    valid_positions = real_span_mask_ltoken.bool()
    # 计算所有正确的位置
    all_correct = (pred_label_idx == span_label_ltoken) & pred_label_mask & valid_positions
    correct_pred = torch.sum(all_correct)
    total_pred = torch.sum((pred_label_idx != 0) & valid_positions)
    total_golden = torch.sum((span_label_ltoken != 0) & valid_positions)

    return torch.stack([correct_pred, total_pred, total_golden]) #,pred_label_idx

def get_predict(args,all_span_word, words,predicts,span_label_ltoken,all_span_idxs):
    '''
    :param all_span_word: tokens for a span;
    :param words: token in setence-level;
    :param predicts: the prediction of model;
    :param span_label_ltoken: the label for span;
    :param all_span_idxs: the position for span;
    '''
    pred_label_idx = torch.max(predicts, dim=-1)[1] # (bs, n_span)
    # for context
    idx2label = {}
    label2idx_list = args.label2idx_list
    for labidx in label2idx_list:
        lab, idx = labidx
        idx2label[int(idx)] = lab

    batch_preds = []
    for span_idxs,word,ws,lps,lts in zip(all_span_idxs,words,all_span_word,pred_label_idx,span_label_ltoken):
        text = ' '.join(word) +"\t"
        for sid,w,lp,lt in zip(span_idxs,ws,lps,lts):
            if lp !=0 or lt!=0:
                plabel = idx2label[int(lp.item())]
                tlabel = idx2label[int(lt.item())]
                sidx, eidx = sid
                ctext = ' '.join(w)+ ':: '+str(int(sidx))+','+str(int(eidx+1))  +':: '+tlabel +':: '+plabel +'\t'
                text +=ctext
        batch_preds.append(text)
    return batch_preds


def get_predict_prune(idx2label,all_span_word, words,predicts_new,span_label_ltoken,all_span_idxs):
    '''
    :param all_span_word: tokens for a span;
    :param words: token in setence-level;
    :param predicts_new: the prediction of model;
    :param span_label_ltoken: the label for span;
    :param all_span_idxs: the position for span;
    '''
    # for context
    # idx2label = {}
    # # label2idx_list = args.label2idx_list
    # for labidx in label2idx_list:
    #     lab, idx = labidx
    #     idx2label[int(idx)] = lab

    batch_preds = []
    for span_idxs,word,ws,lps,lts in zip(all_span_idxs,words,all_span_word,predicts_new,span_label_ltoken):
        text = ' '.join(word) +"\t"
        for sid,w,lp,lt in zip(span_idxs,ws,lps,lts):
            if lp !=0 or lt!=0:
                plabel = idx2label[int(lp.item())]
                tlabel = idx2label[int(lt.item())]
                sidx, eidx = sid
                ctext = ' '.join(w)+ ':: '+str(int(sidx))+','+str(int(eidx+1))  +':: '+tlabel +':: '+plabel +'\t'
                text +=ctext
        batch_preds.append(text)
    return batch_preds


def has_overlapping(idx1, idx2):
    overlapping = True
    if (idx1[0] > idx2[1] or idx2[0] > idx1[1]):
        overlapping = False
    return overlapping

def clean_overlapping_span(idxs_list,nonO_idxs2prob):
    kidxs = []
    didxs = []
    for i in range(len(idxs_list)-1):
        idx1 = idxs_list[i]

        kidx = idx1
        kidx1 = True
        for j in range(i+1,len(idxs_list)):
            idx2 = idxs_list[j]
            isoverlapp = has_overlapping(idx1, idx2)
            if isoverlapp:
                prob1 = nonO_idxs2prob[idx1]
                prob2 = nonO_idxs2prob[idx2]

                if prob1 < prob2:
                    kidx1 = False
                    didxs.append(kidx1)
                elif prob1 == prob2:
                    len1= idx1[1] - idx1[0]+1
                    len2 = idx1[1] - idx1[0] + 1
                    if len1<len2:
                        kidx1 = False
                        didxs.append(kidx1)
        if kidx1:
            flag=True
            for idx in kidxs:
                isoverlap= has_overlapping(idx1,idx)
                if isoverlap:
                    flag=False
                    prob1 = nonO_idxs2prob[idx1]
                    prob2 = nonO_idxs2prob[idx]
                    if prob1>prob2: # del the keept idex
                        kidxs.remove(idx)
                        kidxs.append(idx1)
                    break
            if flag==True:
                kidxs.append(idx1)

    if len(didxs)==0:
        kidxs.append(idxs_list[-1])
    else:
        if idxs_list[-1] not in didxs:
            kidxs.append(idxs_list[-1])


    return kidxs

def get_pruning_predIdxs(pred_label_idx, all_span_idxs,span_probs):
    nonO_kidxs_all = []
    nonO_idxs2labs = []
    # begin{Constraint the span that was predicted can not be overlapping.}
    for i, (bs, idxs) in enumerate(zip(pred_label_idx, all_span_idxs)):
        # collect the span indexs that non-O
        nonO_idxs2lab = {}
        nonO_idxs2prob = {}
        nonO_idxs = []
        for j, (plb, idx) in enumerate(zip(bs, idxs)):
            plb = int(plb.item())
            if plb != 0:  # only consider the non-O label span...
                nonO_idxs2lab[idx] = plb
                nonO_idxs2prob[idx] = span_probs[i][j][plb]
                nonO_idxs.append(idx)
        nonO_idxs2labs.append(nonO_idxs2lab)
        if len(nonO_idxs) != 0:
            nonO_kidxs = clean_overlapping_span(nonO_idxs, nonO_idxs2prob)
        else:
            nonO_kidxs = []
        nonO_kidxs_all.append(nonO_kidxs)

    pred_label_idx_new = []
    n_span = pred_label_idx.size(1)
    for i, (bs, idxs) in enumerate(zip(pred_label_idx, all_span_idxs)):
        pred_label_idx_new1 = []
        for j, (plb, idx) in enumerate(zip(bs, idxs)):
            nlb_id = 0
            if idx in nonO_kidxs_all[i]:
            # if any(idx.equal(kidx) for kidx in nonO_kidxs_all[i]):
                nlb_id = plb
            pred_label_idx_new1.append(nlb_id)
        while len(pred_label_idx_new1) <n_span:
            pred_label_idx_new1.append(0)

        pred_label_idx_new.append(pred_label_idx_new1)
    pred_label_idx_new = torch.LongTensor(pred_label_idx_new)
    return nonO_idxs2labs,nonO_kidxs_all,pred_label_idx_new

def ner_gold(ner_ids, sent_lens, ner_vocab=None):
    '''
    :param ner_ids: (b, t, t)
    :param sent_lens:  (b, )
    '''
    if ner_vocab is None:# 如果ner_vocab为None，则直接将pad_idx和unk_idx设置为特定值
        pad_idx = 0
        unk_idx = None
    else:
        pad_idx = ner_vocab.pad_idx
        unk_idx = ner_vocab.unk_idx
    gold_res = []
    for ner_id, l in zip(ner_ids, sent_lens):
        res = []
        for s in range(l):
            for e in range(s, l):
                type_id = ner_id[s, e].item()
                if type_id not in [pad_idx, unk_idx]:
                    res.append((s, e, type_id))
        gold_res.append(res)
    return gold_res

def ner_pred(pred_score, sent_lens, ner_vocab=None):
    '''
    :param pred_score: (b, t, t, c)
    :param sent_lens: (b, )
    '''
    if ner_vocab is None:# 如果ner_vocab为None，则直接将pad_idx和unk_idx设置为特定值
        pad_idx = 0
        unk_idx = None
    else:
        pad_idx = ner_vocab.pad_idx
        unk_idx = ner_vocab.unk_idx
    # (b, t, t)
    type_idxs = pred_score.detach().argmax(dim=-1)
    # (b, t, t)
    span_max_score = pred_score.detach().gather(dim=-1, index=type_idxs.unsqueeze(-1)).squeeze(-1)
    final = []
    for span_score, tids, l in zip(span_max_score, type_idxs, sent_lens):
        cands = []
        for s in range(l):
            for e in range(s, l):
                type_id = tids[s, e].item()
                if type_id not in [pad_idx, unk_idx]:
                    cands.append((s, e, type_id, span_score[s, e].item()))

        pre_res = []
        for s, e, cls, _ in sorted(cands, key=lambda x: x[3], reverse=True):
            for s_, e_, _ in pre_res:
                if s_ < s <= e_ < e or s < s_ <= e < e_:  # flat ner
                    break
                if s <= s_ <= e_ <= e or s_ <= s <= e <= e_:  # nested ner
                    break
            else:
                pre_res.append((s, e, cls))
        final.append(pre_res)

    return final

def calc_prf(nb_right, nb_pred, nb_gold):
    p = nb_right / (nb_pred + 1e-30)
    r = nb_right / (nb_gold + 1e-30)
    f = (2 * nb_right) / (nb_gold + nb_pred + 1e-30)
    return p*100, r*100, f*100

def calc_acc(preds, golds, return_prf=False):
    '''
    :param preds: [(s, e, cls_id) ...]
    :param golds: [(s, e, cls_id) ...]
    :param return_prf: if True, return prf value, otherwise return number value
    '''
    assert len(preds) == len(golds)
    nb_pred, nb_gold, nb_right = 0, 0, 0
    for pred_spans, gold_spans in zip(preds, golds):
        pred_span_set = set(pred_spans)
        gold_span_set = set(gold_spans)
        nb_pred += len(pred_span_set)
        nb_gold += len(gold_span_set)
        nb_right += len(pred_span_set & gold_span_set)

    if return_prf:
        return calc_prf(nb_right, nb_pred, nb_gold)
    else:
        return nb_right, nb_pred, nb_gold