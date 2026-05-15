import os
from os.path import join as opj
from os.path import dirname as opd
import sys
import numpy as np
import argparse
from omegaconf import OmegaConf
from datetime import datetime
import tqdm
import time

import torch
import torch.nn as nn
import torch.optim as optim

from utils.utils import reproduc
from utils.logger import MyLogger
from datautil.cls_datautils.prep_cls_data import prep_cls_data
from alg.cls_alg.get_alg import get_algorithm_class, model_valid
from utils.utils import score_dict_2_string, eval_ood_performance, print_row, train_valid_target_eval_names, alg_loss_dict
    
# model train
def train(args, log, model, train_loader_list, valid_loader, target_loader, device="cuda"):
    total_step = 0
    best_valid_acc, target_acc = 0, 0
    best_valid_epoch = 0
    
    if args.algorithm == "COGS":
        model.initialize_balanced_environments(train_loader_list)

    for epoch_i in range(1, args.total_epoch + 1):
        all_scores = {}
        print(f"Epoch {epoch_i:d} Model training")
        pred_epoch, label_epoch = model.update_epoch(train_loader_list, epoch_i)

        if epoch_i > -1:
            train_scores = eval_ood_performance(
                pred_epoch,
                label_epoch,
                name="train/"
            )
            all_scores.update(train_scores)
        
        if epoch_i % args.valid_every == 0:
            print(f"Epoch {epoch_i:d} Validation")
            valid_scores, loss_dict = model_valid(model, valid_loader, mode="valid/", device=device)
            if epoch_i > -1:
                all_scores.update(valid_scores)
                if valid_scores["valid/accuracy"] >= best_valid_acc:
                    best_valid_acc = valid_scores["valid/accuracy"]
                    best_valid_epoch = epoch_i
                    print(f"Epoch {epoch_i:d} Target test")
                    target_scores, _ = model_valid(model, target_loader, mode="target/", device=device)
                    all_scores.update(target_scores)
                    target_acc = target_scores["target/accuracy"]
                    print(f'Epoch:{epoch_i}/{args.total_epoch}, Valid Accuracy: {best_valid_acc}')
                    print(f'Epoch:{epoch_i}/{args.total_epoch}, Target Accuracy: {target_acc}') 
                    log.log_txt(score_dict_2_string(all_scores), "scores.txt", epoch_i)
                    if hasattr(model, 'save_graph') and callable(model.save_graph):
                        model.save_graph()
                    test_path = opj(log.log_dir, f"iter_{epoch_i:d}", "model.pt")
                    torch.save(model.state_dict(), test_path)      
    print(f'Best Valid Epoch:{best_valid_epoch}/{args.total_epoch}, Best Valid Accuracy: {best_valid_acc}')                 
    print(f'Best Valid Epoch:{best_valid_epoch}/{args.total_epoch}, Target Accuracy: {target_acc}')
    return opd(test_path)

# model test
def test(test_dir, args, log, model, valid_loader, target_loader, device="cuda"):
    ori_log_dir = model.log.log_dir
    model.log.log_dir = test_dir
    
    model.load_model(test_dir)
    print("Model Loaded.")
    if hasattr(model, 'load_graph') and callable(model.load_graph):
        model.load_graph(test_dir)
        print("Graph Loaded.")
        
    all_scores = {}
    for name, dataloader in {
        "target/": target_loader,
        "valid/": valid_loader,
    }.items():
        print(f"Testing on {name} set")
        
        scores, _ = model_valid(model, dataloader, mode=name, device=model.device)
        
        all_scores.update(scores)

    model.log.log_txt(score_dict_2_string(all_scores), "scores.txt", 0)
    model.log.log_dir = ori_log_dir
    

def main(args, device="cuda", mode="train", test_path=''):
    reproduc(**args.reproduc) 
    timestamp = datetime.now().strftime("_%Y_%m%d_%H%M%S")
    env_name = ""
    if hasattr(args.data, "test_envs"):
        for test_env in args.data.test_envs:
            env_name += str(test_env) + "_"
    args.task_name = env_name + args.task_name
    args.task_name += "_" + mode + timestamp
    proj_path = opj(args.dir_name, args.task_name)
    log = MyLogger(proj_path, **args.log)
    log.log_opt(args)
    
    train_loader_list, valid_loader, target_loader = prep_cls_data(args.data)
    algorithm_class = get_algorithm_class(args.ts_ood.algorithm)
    algorithm = algorithm_class(args.ts_ood, log, device).to(device)
    
    if mode == "train":
        test_path = train(args.ts_ood, log, algorithm, train_loader_list, valid_loader, target_loader, device=device)
        test(test_path, args.ts_ood, log, algorithm, valid_loader, target_loader, device=device)
    elif mode == "test":
        test(test_path, args.ts_ood, log, algorithm, valid_loader, target_loader, device=device)

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    
    parser = argparse.ArgumentParser(description="Causal OoD for Classification")
    parser.add_argument(
        "-o",
        type=str,
        default=opj(opd(__file__), "opt/demo.yaml"),
        help="yaml file path"
    )
    parser.add_argument("-g", type=str, default="3", help="available gpu list")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test_path", type=str, default="", help="model path for testing")
    args = parser.parse_args()
    
    if args.g == "cpu":
        device = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.g
        device = "cuda"
        
    main(OmegaConf.load(args.o), device=device, mode="test" if args.test else "train", test_path=args.test_path)