import torch
import torch.nn as nn
import argparse
import numpy as np
from torch.autograd import Function
from torch.autograd.functional import jacobian
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm
from argparse import Namespace
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

class NPChangeTransitionPrior(nn.Module):

    def __init__(
        self, 
        lags, 
        latent_size, 
        embedding_dim,
        num_layers=2, 
        hidden_dim=64):
        super().__init__()
        self.L = lags       
        gs = [MLP(embedding_dim+lags*latent_size+1, 
                    hidden_dim,
                    1, 
                    num_layers, 
                    dropout=0,
                    act="leakyrelu",
                    use_norm=False,
                    norm_type="ln") for i in range(latent_size)]
        
        self.gs = nn.ModuleList(gs)
        self.fc = MLP(embedding_dim, 
                    hidden_dim,
                    embedding_dim, 
                    2, 
                    dropout=0,
                    act="leakyrelu",
                    use_norm=False,
                    norm_type="ln")

    def forward(self, x, embeddings, masks=None):
        batch_size, length, input_dim = x.shape
        embeddings = self.fc(embeddings)
        x = x.unfold(dimension = 1, size = self.L+1, step = 1)  # x: [BS, T, D] -> [BS, T-L, D, L+1]
        x = torch.swapaxes(x, 2, 3)     # [BS, T-L, L+1, D]
        time_steps = length - self.L
        embeddings = embeddings.unsqueeze(1).expand(batch_size, time_steps, -1).reshape(-1, embeddings.size(-1))
    
        x = x.reshape(-1, self.L+1, input_dim)
        yy, xx = x[:,-1:], x[:,:-1]
        xx = xx.reshape(-1, self.L*input_dim)
        residuals = []
        hist_jac = []
        sum_log_abs_det_jacobian = 0
        for i in range(input_dim):
            if masks is None:
                inputs = torch.cat((embeddings, xx, yy[:,:,i]), dim=-1)
            else:
                mask = masks[i]
                inputs = torch.cat((embeddings, xx*mask, yy[:,:,i]), dim=-1)
            residual = self.gs[i](inputs)
            with torch.enable_grad():
                pdd = torch.func.vmap(torch.func.jacfwd(self.gs[i]))(inputs)
                if torch.isinf(pdd).any():
                    print("pdd is inf!")
            # Determinant of low-triangular mat is product of diagonal entries
            eps = 1e-8
            logabsdet = torch.log(torch.abs(pdd[:, 0, -1]) + eps)
            hist_jac.append(torch.unsqueeze(pdd[:, 0, :-1], dim=1))
            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)

        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, -1, input_dim)
        sum_log_abs_det_jacobian = torch.sum(sum_log_abs_det_jacobian.reshape(batch_size, length-self.L), dim=1)
        return residuals, sum_log_abs_det_jacobian, hist_jac

class NPTransitionPrior(nn.Module):

    def __init__(
        self, 
        lags, 
        latent_size, 
        num_layers=2, 
        hidden_dim=64):
        super().__init__()
        self.L = lags       
        gs = [MLP(lags*latent_size+1, 
                    hidden_dim,
                    1, 
                    num_layers, 
                    dropout=0,
                    act="leakyrelu",
                    use_norm=False,
                    norm_type="ln") for i in range(latent_size)]
        
        self.gs = nn.ModuleList(gs)
    
    def forward(self, x, masks=None):
        batch_size, length, input_dim = x.shape
        x = x.unfold(dimension = 1, size = self.L+1, step = 1)  # x: [BS, T, D] -> [BS, T-L, D, L+1]
        x = torch.swapaxes(x, 2, 3)     # [BS, T-L, L+1, D]
        x = x.reshape(-1, self.L+1, input_dim)
        yy, xx = x[:,-1:], x[:,:-1]
        xx = xx.reshape(-1, self.L*input_dim)
        residuals = []
        hist_jac = []
        sum_log_abs_det_jacobian = 0
        for i in range(input_dim):
            if masks is None:
                inputs = torch.cat((xx, yy[:,:,i]), dim=-1)
            else:
                mask = masks[i]
                inputs = torch.cat((xx*mask, yy[:,:,i]), dim=-1)
            residual = self.gs[i](inputs)
            with torch.enable_grad():
                pdd = torch.func.vmap(torch.func.jacfwd(self.gs[i]))(inputs)
                if torch.isinf(pdd).any():
                    print("pdd is inf!")
            # Determinant of low-triangular mat is product of diagonal entries
            eps = 1e-8
            logabsdet = torch.log(torch.abs(pdd[:, 0, -1]) + eps)
            hist_jac.append(torch.unsqueeze(pdd[:, 0, :-1], dim=1))
            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)

        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, -1, input_dim)
        sum_log_abs_det_jacobian = torch.sum(sum_log_abs_det_jacobian.reshape(batch_size, length-self.L), dim=1)
        return residuals, sum_log_abs_det_jacobian, hist_jac

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, layers=1, type="bn"):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x

class classifier(nn.Module):
    def __init__(self, in_dim, class_num, cls_type="linear"):
        super(classifier, self).__init__()
        self.type = cls_type
        if self.type == 'wn':
            self.fc = weightNorm(
                nn.Linear(in_dim, class_num), name="weight")
        else:
            self.fc = nn.Linear(in_dim, class_num)

    def forward(self, x):
        x = self.fc(x)
        return x
    
class MLP(nn.Module):
    def __init__(self, in_dim, n_hid, out_dim, n_layer, dropout=0, act="leakyrelu", use_norm=False, norm_type="bn"):
        """Component for encoder and decoder

        Args:
            in_dim (int): input dimension.
            n_hid (int): model layer dimension.
            out_dim (int): output dimension.
        """
        super(MLP, self).__init__()
        dims = [(in_dim, n_hid)] + [(n_hid, n_hid) for _ in range(n_layer - 1)] + [(n_hid, out_dim)]
        fc_layers = [nn.Linear(pair[0], pair[1]) for pair in dims]
        if use_norm:
            if norm_type == "bn":
                norm_layers = [nn.BatchNorm1d(n_hid) for _ in range(n_layer)]
            elif norm_type == "ln":
                norm_layers = [nn.LayerNorm(n_hid) for _ in range(n_layer)]

        if act == "leakyrelu":
            act_fn = nn.LeakyReLU(0.2)
        elif act == "relu":
            act_fn = nn.ReLU()
        elif act == "tanh":
            act_fn = nn.Tanh()
        elif act == "sigmoid":
            act_fn = nn.Sigmoid()
        elif act == "gelu":
            act_fn = nn.GELU()
        else:
            raise NotImplementedError

        act_layers = [act_fn for _ in range(n_layer)]
        layers = []
        for i in range(n_layer):
            layers.append(fc_layers[i])
            if use_norm:
                layers.append(norm_layers[i])
            layers.append(act_layers[i])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(fc_layers[-1])
        self.network = nn.Sequential(*layers)
        self.init_weights()

    def forward(self, x):
        return self.network(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

class Discriminator(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, num_domains=4):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_domains),
        ]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class cnn_net(nn.Module):
    def __init__(self, args):
        super(cnn_net, self).__init__()
        self.taskname = args.taskname
        self.input_channel = args.encoder.z_dim
        self.hid_channel = args.featurizer.hid_channel
        self.out_channel = args.featurizer.out_channel
        self.t_length = args.t_length
        self.kernel_size = args.featurizer.kernel_size
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.input_channel,
                    out_channels=self.hid_channel, 
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2),
            nn.BatchNorm1d(self.hid_channel),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.hid_channel,
                    out_channels=self.hid_channel * 2, 
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2),
            nn.BatchNorm1d(self.hid_channel * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=self.hid_channel * 2,
                    out_channels=self.out_channel, 
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2),
            nn.BatchNorm1d(self.out_channel),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        )
        self.out_dim = self.calc_fc_dim()
    
    def calc_fc_dim(self):
        x = torch.zeros(1, self.input_channel, self.t_length)
        x = self.conv3(self.conv2(self.conv1(x)))
        return np.prod(x.shape[1:])
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, self.out_dim)
        return x


class tea_feat_net(nn.Module):
    def __init__(self, input_channel, hid_channel, out_channel, t_length, kernel_size):
        super(tea_feat_net, self).__init__()
        self.input_channel = input_channel
        self.hid_channel = hid_channel
        self.out_channel = out_channel
        self.t_length = t_length
        self.kernel_size = kernel_size
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.input_channel,
                    out_channels=self.hid_channel, 
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2),
            nn.BatchNorm1d(self.hid_channel),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.hid_channel,
                    out_channels=self.hid_channel * 2, 
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2),
            nn.BatchNorm1d(self.hid_channel * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=self.hid_channel * 2,
                    out_channels=self.out_channel, 
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2),
            nn.BatchNorm1d(self.out_channel),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        )
        self.out_dim = self.calc_fc_dim()
    
    def calc_fc_dim(self):
        x = torch.zeros(1, self.input_channel, self.t_length)
        x = self.conv3(self.conv2(self.conv1(x)))
        return np.prod(x.shape[1:])
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, self.out_dim)
        return x

class gru_net(nn.Module):
    def __init__(self, ood_args):
        super(gru_net, self).__init__()
        self.ood_args = ood_args
        self.input_dim = self.ood_args.input_dim
        self.num_layers = self.ood_args.featurizer.layers
        self.hid_dim = self.ood_args.featurizer.hid_dim
        self.dropout = self.ood_args.featurizer.dropout

        in_size = self.input_dim
        features = nn.ModuleList()
        for i in range(self.num_layers):
            gru = nn.GRU(
                input_size=in_size,
                num_layers=1,
                hidden_size=self.hid_dim,
                batch_first=True,
                dropout=self.dropout
            )
            features.append(gru)
            in_size = self.hid_dim
        self.gru_s = nn.Sequential(*features)
        
        self.out_feat_dim = self.hid_dim
    
    def forward(self, x):
        x_input = x
        for i in range(self.num_layers):
            out, hidden_state = self.gru_s[i](x_input.float())
            x_input = out
        return out, hidden_state

def dict_to_namespace(d):
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_namespace(value)
    return Namespace(**d)

class cnn_featurizer(nn.Module):
    def __init__(self, args):
        super(cnn_featurizer, self).__init__()
        self.taskname = args.taskname
        self.input_channel = args.encoder.z_dim
        self.hid_channel = args.featurizer.hid_channel
        self.out_channel = args.featurizer.out_channel
        self.t_length = args.t_length
        self.kernel_size = args.featurizer.kernel_size

        self.conv1 = nn.Sequential(
                nn.Conv1d(in_channels=self.input_channel,
                        out_channels=self.hid_channel, 
                        kernel_size=self.kernel_size,
                        stride=1,
                        padding=self.kernel_size // 2),
                nn.BatchNorm1d(self.hid_channel),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
            )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.hid_channel,
                    out_channels=self.hid_channel * 2, 
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2),
            nn.BatchNorm1d(self.hid_channel * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=self.hid_channel * 2,
                    out_channels=self.out_channel, 
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2),
            nn.BatchNorm1d(self.out_channel),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        )
        self.conv_feat_dim = self.calc_fc_dim()
        self.projector = nn.Sequential(
            nn.Linear(self.conv_feat_dim, self.hid_channel),
            nn.LayerNorm(self.hid_channel),
            nn.ReLU(inplace=True),
            nn.Linear(self.hid_channel, args.env_infer.env_proto_dim)
        )

    def calc_fc_dim(self):
        x = torch.zeros(1, self.input_channel, self.t_length)
        x = self.conv3(self.conv2(self.conv1(x)))
        return np.prod(x.shape[1:])
    
    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        feat = x.view(-1, self.conv_feat_dim)
        proto = self.projector(feat)
        return proto, feat