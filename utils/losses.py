import torch
import torch.nn.functional as F



def nsgan_criterion(pred, is_real):
    if is_real: target = torch.ones_like(pred)
    else:       target = torch.zeros_like(pred)
    loss = F.binary_cross_entropy_with_logits(pred, target)
    return loss