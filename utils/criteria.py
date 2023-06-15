import torch
import torch.nn.functional as F



def nsgan_loss(pred, is_real):
    if is_real: target = torch.ones_like(pred)
    else:       target = torch.zeros_like(pred)
    loss = F.binary_cross_entropy_with_logits(pred, target)
    return loss


def r1_regularizer(discriminator, real, r1_gamma):
    pred_real = discriminator(real)
    image_grad = torch.autograd.grad(outputs=[pred_real.sum()], inputs=[real])[0]
    return r1_gamma * 0.5 * image_grad.square().sum(dim=[1, 2, 3]).mean()