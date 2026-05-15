# coding=utf-8
import torch
import torch.nn as nn
import tqdm
import numpy as np

from alg.cls_alg.algs.COGS import COGS
from utils.utils import eval_ood_performance


ALGORITHMS = [
    'COGS',
]


def get_algorithm_class(algorithm_name):
    if algorithm_name not in globals():
        raise NotImplementedError(
            "Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

# model validation
def model_valid(network, loader, mode="valid/", device="cuda"):
    pred_epoch = []
    label_epoch = []
    loss_dict = {}
    loss_list = []

    network.eval()
    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(loader))
        for data in loader:
            x = data[0].transpose(1, 2).to(device).float() 
            labels = data[1].to(device).long()
                
            pred_batch, loss_dict_vae = network.predict(x, mode=mode)
            loss_batch = network.cls_loss(pred_batch, labels)
            loss_list.append(loss_batch.item())

            pred_epoch.append(pred_batch.detach().cpu())
            label_epoch.append(labels.detach().cpu())

            pbar.update(1)

        pbar.close()
    loss_dict["cls_loss"] = np.mean(loss_list)
    for key, value in loss_dict_vae.items():
        loss_dict[key] = value
    network.train()
    scores = eval_ood_performance(
        pred_epoch,
        label_epoch,
        name=mode
    )
    return scores, loss_dict