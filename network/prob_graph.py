import torch
from network.gumbel_softmax import gumbel_softmax
import torch.nn as nn 

def build_graph(args, graph_in_dim, graph_out_dim, device="cuda", mode="multi_pred"):
    if mode == "multi_pred":
        if hasattr(args, "time_graph") and args.time_graph.enable:
            graph = nn.Parameter(
                torch.ones([args.time_graph.time_chunk_num, graph_in_dim, graph_out_dim]).to(device)
            )
            graph.data[-1, :] = 0
            graph.data[:-1, :] = 5
        else:
            graph = nn.Parameter(
                torch.ones([graph_in_dim, graph_out_dim]).to(device) * 0
            )
    elif mode == "unify_pred":
        if hasattr(args, "time_graph") and args.time_graph.enable:
            graph = nn.Parameter(
                torch.ones([args.time_graph.time_chunk_num, graph_in_dim]).to(device)
            )
            graph.data[-1, :] = 0
            graph.data[:-1, :] = 0
        else:
            graph = nn.Parameter(
                torch.ones([graph_in_dim,]).to(device) * 0
            )
    elif mode == "discov":
        if hasattr(args, "disable_graph") and args.disable_graph:
            print("Using full graph and disable graph discovery...")
            graph = nn.Parameter(torch.ones([graph_in_dim, graph_in_dim]).to(device) * 1000)
        else:
            graph = nn.Parameter(torch.ones([graph_in_dim, graph_in_dim]).to(device) * 0)

    return graph

def cumulative_time_graph(prob_graph, t_length, cumu_type="prod"):      
    l, n = prob_graph.shape
    
    chunk_size = t_length // l
    cum_graph = torch.zeros([t_length, n], device=prob_graph.device, dtype=prob_graph.dtype)
    for li in range(l-1, -1, -1): 
        if cumu_type == "prod":
            chunk_prob = torch.prod(prob_graph[li:], dim=0, keepdim=True)
        elif cumu_type == "sum":
            chunk_prob = torch.sum(prob_graph[:li+1], dim=0, keepdim=True)
        elif cumu_type == "equal":
            chunk_prob = prob_graph[li].unsqueeze(0)
        cum_graph[li*chunk_size:(li+1)*chunk_size] = chunk_prob.repeat([chunk_size,] + [1 for _ in chunk_prob.shape[1:]])

    cum_graph = torch.clip(cum_graph, 0, 1)
    return cum_graph

def bernonlli_sample(theta, batch_size, prob=True, hard_mask=False, t_length=None, threshold=0.5, time_cumu_type="prod", time_cumulative=False):
    prob_graph = torch.sigmoid(theta)

    if time_cumulative:
        prob_graph = cumulative_time_graph(prob_graph, t_length, cumu_type=time_cumu_type)
    sample_matrix = prob_graph[None].expand([batch_size, ] + [-1 for _ in prob_graph.shape]).clone()
    if hard_mask:
        sample_matrix = (sample_matrix > threshold).int()
        return sample_matrix
    else:
        if prob:
            return torch.bernoulli(sample_matrix)
        else:
            sample_matrix[sample_matrix < 1e-3] = torch.zeros_like(sample_matrix[sample_matrix < 1e-3])
            return sample_matrix

        

def gumbel_sample(theta, batch_size, tau=1, t_length=None, time_cumu_type="prod", time_cumulative=False, batch_dim=False, time_dim=True, causalgraph_2d=False):
    # dim: self.graph.shape
    prob_graph = torch.sigmoid(theta)
    
    if time_cumulative:
        prob_graph = cumulative_time_graph(prob_graph, t_length, cumu_type=time_cumu_type)
        
    # add first_batch dim and last_prob_dim
    prob_graph = prob_graph[None, ..., None].expand([batch_size, ] + [-1 for _ in prob_graph.shape] + [-1, ])
    logits = torch.concat([prob_graph, (1 - prob_graph)], axis=-1)
    samples = gumbel_softmax(logits, tau=tau)[..., 0]
    
    return samples, prob_graph
        
        
def freeze_graph(args, theta):
    args.data_pred.prob = False
    args.data_pred.hard_mask = True
    delattr(args, "graph_discov")
    print("Freezing graph!")