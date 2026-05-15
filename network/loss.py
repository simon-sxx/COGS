import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.mean(torch.sum(entropy, dim=1))
    return entropy


def Entropylogits(input, redu='mean'):
    input_ = F.softmax(input, dim=1)
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    if redu == 'mean':
        entropy = torch.mean(torch.sum(entropy, dim=1))
    elif redu == 'None':
        entropy = torch.sum(entropy, dim=1)
    return entropy


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets):
        if inputs.shape[1] == 1:
            inputs = torch.cat([1-inputs, inputs], dim=1)
        
        ce_loss = F.cross_entropy(inputs, targets, 
                                  reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()
    

class Multitask_CE(nn.Module):
    def __init__(self, base_loss="focalloss"):
        super().__init__()
        
        if base_loss == "ce" or base_loss == "crossentropy": 
            self.base_loss = nn.CrossEntropyLoss()
        elif base_loss == "focalloss":
            self.base_loss = FocalLoss()
        else:
            raise NotImplementedError
        
    def forward(self, y_list, label_list):
        if not isinstance(y_list, list):
            y_list = [y_list]
        if not isinstance(label_list, list):
            label_list = [label_list]
        
        val = 0
        for y, label in zip(y_list, label_list):
            val += self.base_loss(y, label)
        return val


class Gauss_Gate_Regularizer_Gauss(nn.Module):
    def __init__(self, sigma):
        self.sigma = sigma
        super().__init__()
        
    def forward(self, mu):
        graph_erf = 0.5 - 0.5 * torch.erf(-(mu + 0.5) / (np.sqrt(2) * self.sigma))
        return torch.mean(graph_erf)


class L1_Regularizer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, mu):
        return torch.norm(mu, p=1)

class OrthogonalProjectionLoss(nn.Module):
    def __init__(self, gamma=1.0):
        super(OrthogonalProjectionLoss, self).__init__()
        self.gamma = gamma

    def forward(self, features, labels=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        #  features are normalized
        features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]  # extend dim

        mask = torch.eq(labels, labels.t()).bool().to(device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = torch.abs(mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)

        loss = (1.0 - pos_pairs_mean) + (self.gamma * neg_pairs_mean)

        return loss

class Causal_Loss(nn.Module):
    def __init__(self, data_loss, norm="l1", norm_by_shape=True):
        super().__init__()
        
        self.data_loss = data_loss
        self.norm_by_shape = norm_by_shape
        if norm == "l1":
            self.norm = lambda x: torch.norm(x, p=1)
        else:
            raise NotImplementedError
        
    def forward(self, y, label, graph):
        
        loss_sparsity = 0

        
        if self.norm_by_shape:
            norm = self.norm(graph) / np.prod(graph.shape)
        else:
            norm = self.norm(graph)
        loss_sparsity += norm
            
        loss_data = self.data_loss(y, label)

        return loss_data, loss_sparsity

class CrossEntropyLossMaybeSmooth(nn.CrossEntropyLoss):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    def __init__(self, smooth_eps=0.0):
        super(CrossEntropyLossMaybeSmooth, self).__init__()
        self.smooth_eps = smooth_eps

    def forward(self, output, target, smooth=False):
        if not smooth:
            return F.cross_entropy(output, target)

        target = target.contiguous().view(-1)
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        smooth_one_hot = one_hot * (1 - self.smooth_eps) + (1 - one_hot) * self.smooth_eps / (n_class - 1)
        log_prb = F.log_softmax(output, dim=1)
        loss = -(smooth_one_hot * log_prb).sum(dim=1).mean()
        return loss


class Irregular_Recon_Loss(nn.Module):
    def __init__(self, data_loss):
        super().__init__()
        self.data_loss = data_loss
        
    def forward(self, y, gt):
        if len(gt.shape) == 3: # b n d
            gt_arr = gt[:, :, 0]
            gt_abs = gt[:, :, 1]
            b, n, d = gt.shape
        elif len(gt.shape) == 4: # b n t d
            gt_arr = gt[:, :, :, 0]
            gt_abs = gt[:, :, :, 1]
            b, n, t, d = gt.shape
        
        loss = self.data_loss(y * (1 - gt_abs), gt_arr * (1 - gt_abs)) / torch.mean(1 - gt_abs)
        return loss
    
def build_loss(name):
    if name == "ce" or name == "crossentropy":
        basic_loss = nn.CrossEntropyLoss()
    elif name == "focal_loss":
        basic_loss = FocalLoss()
    elif name == "mse":
        basic_loss = nn.MSELoss()
    else:
        raise NotImplementedError
    return basic_loss