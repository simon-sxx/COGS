# coding=utf-8
import torch
import torch.nn as nn
import math


def lr_lambda(epoch, total_epochs, warmup_epochs, lr_start, lr_end):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(math.pi * progress)))
        return lr_end + (lr_start - lr_end) * cosine_decay

def build_optim(fitting_model, args, total_epoch):
    if isinstance(fitting_model, nn.Module):
        parameters = fitting_model.parameters()
    elif isinstance(fitting_model, nn.Parameter):
        parameters = [fitting_model]
    elif isinstance(fitting_model, list):
        parameters = fitting_model
    else:
        raise NotImplementedError

    if args.optim == "adam":
        optimizer = torch.optim.Adam(parameters, lr=args.lr_start, weight_decay=args.weight_decay)
    elif args.optim == "adamW":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr_start, weight_decay=args.weight_decay)
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=args.lr_start, momentum=0.9)
    else:
        raise NotImplementedError

    if hasattr(args, "scheduler"):
        if args.scheduler == "cos":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch // 5)  # * iters
        elif args.scheduler == "step":
            gamma = (args.lr_end / args.lr_start) ** (1 / total_epoch)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
        elif args.scheduler == "cosine_warmup":
            warmup_epochs = args.warmup_epochs
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda epoch: lr_lambda(epoch, total_epoch, warmup_epochs, args.lr_start, args.lr_end))
        else:
            raise NotImplementedError
    else:
        scheduler = None

    return optimizer, scheduler