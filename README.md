# Pre-BiafNER

# Performance

| Pretrained Model  | Model | Dataset | F1 |
| ------------- | ------------- |------------- |------------- |
| ---  | SOTA | ConLL2003 | 94.6
| bert-large-cased  | Biaffine NER | ConLL2003  |  	93.5
| bert-base-cased(Our) (wo doc-context) | Deep Biaffine | ConLL2003  |  92.31
| bert-large-cased(Our) (wo doc-context) | Deep Biaffine | ConLL2003  |  92.52
| ------------- | ------------- |------------- |------------- |
| ---| SOTA | Ontonotes | 92.07
| bert-large-cased (wo doc-context) | SynLSTM | Ontonotes  |  90.85
| bert-large-cased (wo doc-context) | AELGCN | Ontonotes  | 91.16
| bert-large-cased  | Biaffine NER | Ontonotes  |  91.3
| bert-base-cased(Our) (wo doc-context) | Deep Biaffine | Ontonotes  |  90.07
| bert-large-cased(Our) (wo doc-context)  | Deep Biaffine | Ontonotes  |  90.66

Conll03
bert-base-cased, plm_lr: 2e-05, other lr:0.002, lstm layer: 1, epoch: 100, batchsize: 48, Micro F1: 92.31
bert-base-cased, plm_lr: 2e-05, other lr:0.001,  lstm layer: 3, epoch: 100, batchsize: 48,Micro F1: 92.30 + pos
bert-base-cased, plm_lr: 2e-05, other lr:0.003,  lstm layer: 1, epoch: 100, batchsize: 48,Micro F1: 92.31 + pos(48)
bert-large-cased, plm_lr: 2e-05, other lr:0.003,  lstm layer: 1, epoch: 100, batchsize: 48,Micro F1: 92.52 + pos(48)


Ontonotes
bert-base-cased, plm_lr: 2e-05, other lr:0.001, synlstm, epoch: 100, batchsize: 48, Micro F1: 90.07
bert-base-cased, plm_lr: 2e-05, other lr:0.001, synlstm, epoch: 100, batchsize: 48, Micro F1: 90.07 + pos(48)+ dep(48)
bert-base-cased, plm_lr: 2e-05, other lr:0.001, synlstm, embedder dropout 0.5, epoch: 100, batchsize: 48, 90.00, 90.25, Micro F1: 90.13 + pos(48)+ dep(48)
