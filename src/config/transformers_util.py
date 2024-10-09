from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from torch.optim import AdamW

    
def _check_soft_target(x: torch.Tensor):
    assert x.dim() == 2
    if x.size(0) > 0:
        assert (x >= 0).all().item()
        assert (x.sum(dim=-1) - 1).abs().max().item() < 1e-6


def collect_params(model: torch.nn.Module, param_groups: list):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model
    param_groups : list
        [{'params': List[torch.nn.Parameter], 'lr': lr, ...}, ...]
    """
    existing = [params for group in param_groups for params in group['params']]
    missing = []
    for params in model.parameters():
        if all(params is not e_params for e_params in existing):
            missing.append(params)
    return missing

def linear_decay_lr_with_warmup(num_warmup_steps: int, num_total_steps: int):
    assert num_warmup_steps >= 1
    assert num_total_steps >= num_warmup_steps

    def lr_lambda(step: int):
        if step < num_warmup_steps:
            return step / num_warmup_steps
        elif step < num_total_steps:
            return (num_total_steps - step) / (num_total_steps - num_warmup_steps)
        else:
            return 0.0

    return lr_lambda
    
def get_huggingface_optimizer_and_scheduler(model: nn.Module,
                                            pretr_lr: float, other_lr: float,
                                            num_training_steps: int,
                                            weight_decay: float = 0.0,
                                            eps: float = 1e-8,
                                            warmup_step: int = 0):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
    #         "weight_decay": weight_decay,
    #     },
    #     {
    #         "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.0,
    #     },
    # ]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "transformer" in n and not any(nd in n for nd in no_decay)],
            "lr": pretr_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if "transformer" in n and any(nd in n for nd in no_decay)],
            "lr": pretr_lr,
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if "transformer" not in n and not any(nd in n for nd in no_decay)],
            "lr": other_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if "transformer" not in n and any(nd in n for nd in no_decay)],
            "lr": other_lr,
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, eps=eps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_step, num_training_steps=num_training_steps
    )
    return optimizer, scheduler

# def get_huggingface_optimizer_and_scheduler(model: nn.Module,
#                                             lr: float, plm_lr: float,
#                                             num_total_steps: int, num_warmup_steps: int = 0,
#                                             weight_decay: float = 0.0,
#                                             eps: float = 1e-8):
#     # logger.info(f"Using AdamW optimizeer by HuggingFace with {learning_rate} learning rate, "
#     #               f"eps: {eps}, weight decay: {weight_decay}, warmup_step: {warmup_step}, ")
#     no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
#     pretrained_parameters = []
#     other_parameters = []
#     for name, param in model.named_parameters():
#         if param.requires_grad:
#             if any(nd in name for nd in no_decay):
#                 pretrained_parameters.append(param)
#             else:
#                 other_parameters.append(param)

#     optimizer_grouped_parameters = [{'params': model.transformer.model.parameters(), 'lr': plm_lr, 'weight_decay': weight_decay, 'eps': eps}]
#     optimizer_grouped_parameters.append({'params': collect_params(model, optimizer_grouped_parameters), 'lr': lr, 'weight_decay': weight_decay, 'eps': eps})

#     optimizer = AdamW(optimizer_grouped_parameters)

#     lr_lambda = linear_decay_lr_with_warmup(num_warmup_steps=num_warmup_steps, num_total_steps=num_total_steps)
#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
#     scaler = torch.cuda.amp.GradScaler(enabled=False)
#     return optimizer, scheduler, scaler