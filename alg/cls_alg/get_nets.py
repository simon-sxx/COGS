import torch
import torch.nn as nn
import sys
sys.path.append('../')
from network.nets import cnn_net, MLP


def get_feat(args):
    if args.featurizer.model == 'cnn':
        net = cnn_net(args)
    elif args.featurizer.model == "mlp":
        net = MLP(args.encoder.z_dim * args.t_length, 
                args.featurizer.hid_dim,
                args.featurizer.hid_dim, 
                args.featurizer.layer_num,
                dropout=args.featurizer.dropout,
                act=args.featurizer.act_fn
            )
    else:
        raise NotImplementedError
    
    return net

def get_encoder(args):
    z_dim = args.encoder.z_dim * 2
    if args.encoder.model == "mlp":
        net = MLP(args.input_dim, 
                args.encoder.hid_dim,
                z_dim, 
                args.encoder.layer_num,
                dropout=args.encoder.dropout,
                act=args.encoder.act_fn,
                use_norm=args.encoder.use_norm,
                norm_type=args.encoder.norm_type
            )
    else:
        raise NotImplementedError
    
    return net

def get_decoder(args):
    if args.decoder.model == "mlp":
        net = MLP(args.encoder.z_dim, 
                args.decoder.hid_dim,
                args.input_dim, 
                args.decoder.layer_num,
                dropout=args.decoder.dropout,
                act=args.decoder.act_fn,
                use_norm=args.decoder.use_norm,
                norm_type=args.decoder.norm_type
            )
    else:
        raise NotImplementedError
    
    return net


