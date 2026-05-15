# encoding=utf-8
import os
from os.path import join as opj
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import argparse
from omegaconf import OmegaConf
import copy

class data_loader_dsads(Dataset):
    def __init__(self, samples, labels, domains):
        self.samples = samples
        self.labels = labels
        self.domains = domains
        self.pclabels = np.ones(self.labels.shape)*(-1)
        self.pdlabels = np.ones(self.labels.shape)*(0)
        
    def set_labels(self, tlabels=None, label_type='domain_label'):
        assert len(tlabels) == len(self.x)
        if label_type == 'pclabel':
            self.pclabels = tlabels
        elif label_type == 'pdlabel':
            self.pdlabels = tlabels
        elif label_type == 'domain_label':
            self.domains = tlabels
        elif label_type == 'class_label':
            self.labels = tlabels

    def set_labels_by_index(self, tlabels=None, tindex=None, label_type='domain_label'):
        if label_type == 'pclabel':
            self.pclabels[tindex] = tlabels
        elif label_type == 'pdlabel':
            self.pdlabels[tindex] = tlabels
        elif label_type == 'domain_label':
            self.domains[tindex] = tlabels
        elif label_type == 'class_label':
            self.labels[tindex] = tlabels

    def __getitem__(self, index):
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        pctarget, pdtarget = self.pclabels[index], self.pdlabels[index]
        sample = torch.from_numpy(sample)
        sample = sample.permute(1, 0)
        return sample, target, domain, pctarget, pdtarget, index

    def __len__(self):
        return len(self.samples)

def load_domain_data(env, people_group, act_dir, act_label):
    x_list, y_list, d_list = [], [], []
    for person in people_group:
        person_dir = opj(act_dir, f"p{person}")
        for file in os.listdir(person_dir):
            x = np.loadtxt(opj(person_dir, file), delimiter=',')
            x_list.append(x[np.newaxis, :, :])
            y_list.append(act_label-1)
            d_list.append(env)

    x = np.concatenate(x_list, axis=0)
    y = np.array(y_list).reshape((-1,))
    d = np.array(d_list).reshape((-1,))
    
    return x, y, d

def prep_dsads_data(args):
    data_dir = opj(args.data_dir, "dsads")
    test_envs = args.test_envs
    split_type = args.split_type
    
    rate = 0.2
    domain_dict = {0: [1, 2], 1: [3, 4], 2: [5, 6], 3: [7, 8]}  
          
    source_domain_dict = domain_dict.copy()
    source_domain_dict.pop(test_envs[0])
    
    dir_list = os.listdir(data_dir)
    dir_list.sort()
    env_data_list = [[] for _ in range(len(source_domain_dict))]
    valid_x_list, valid_y_list, valid_d_list = [], [], []
    target_x_list, target_y_list, target_d_list = [], [], []
    for act in dir_list:
        act_label = int(act[-2:])
        act_dir = os.path.join(data_dir, act)
        # source
        for i, (env, people_group) in enumerate(source_domain_dict.items()):
            x, y, d = load_domain_data(env, people_group, act_dir, act_label)
            if test_envs[0] == 0:
                d = d-1
            elif test_envs[0] == 1 and int(env) > 1:
                d = d-1
            elif test_envs[0] == 2 and int(env) > 2:
                d = d-1

            l = x.shape[0]
            indexall = np.arange(l)
            np.random.shuffle(indexall)
            ted = int(l*rate)
            indextr, indexval = indexall[ted:], indexall[:ted]
            train_x, train_y, train_d = x[indextr], y[indextr], d[indextr]
            valid_x, valid_y, valid_d = x[indexval], y[indexval], d[indexval]

            env_data_list[i].append((train_x, train_y, train_d))

            valid_x_list.append(valid_x)
            valid_y_list.append(valid_y)
            valid_d_list.append(valid_d)
        # target 
        target_people = domain_dict[test_envs[0]]
        x, y, d = load_domain_data(list(domain_dict.keys())[-1], target_people, act_dir, act_label) 
        target_x_list.append(x)
        target_y_list.append(y)   
        target_d_list.append(d)
            
    train_loader_list = []        
    for env_i, env_data in enumerate(env_data_list):
        train_x_list, train_y_list, train_d_list = [], [], []
        for train_x, train_y, train_d in env_data:
            train_x_list.append(train_x)
            train_y_list.append(train_y)
            train_d_list.append(train_d)

        train_x = np.concatenate(train_x_list, axis=0)
        train_y = np.concatenate(train_y_list, axis=0)
        train_d = np.concatenate(train_d_list, axis=0)

        train_set = data_loader_dsads(train_x, train_y, train_d)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
        train_loader_list.append(train_loader)
        print(f"env:{env_i}, train_x shape: {train_x.shape}")

    valid_x = np.concatenate(valid_x_list, axis=0)
    valid_y = np.concatenate(valid_y_list, axis=0)
    valid_d = np.concatenate(valid_d_list, axis=0)
    
    target_x = np.concatenate(target_x_list, axis=0)
    target_y = np.concatenate(target_y_list, axis=0)
    target_d = np.concatenate(target_d_list, axis=0)
    
    valid_set = data_loader_dsads(valid_x, valid_y, valid_d)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
    for i in range(len(train_loader_list)):
        print(f"env:{i}, train_loader batch: ", len(train_loader_list[i]))
    print("valid_loader batch: ", len(valid_loader))
    
    print('target_domain:', test_envs[0])
    target_set = data_loader_dsads(target_x, target_y, target_d)
    # sample, _, _ = target_set.__getitem__(0)
    # print(sample[0:5])
    target_loader = DataLoader(target_set, batch_size=args.batch_size, shuffle=False)
    print('target_loader batch: ', len(target_loader))
    print(f"train_x shape: {train_x.shape}, valid_x shape: {valid_x.shape}, target_x shape: {target_x.shape}")
    
    return train_loader_list, valid_loader, target_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', type=str, default="")
    args = parser.parse_args()
    prep_dsads_data(OmegaConf.load(args.o).data)