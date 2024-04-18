import argparse
import sys
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from torch_scatter import scatter

from logger import Logger, save_result
from dataset import load_dataset
from data_utils import normalize, gen_normalized_adjs, eval_acc, eval_rocauc, eval_f1, to_sparse_tensor, \
    load_fixed_splits, adj_mul, get_gpu_memory_map, count_parameters,compute_degrees
from eval import evaluate, evaluate_large
from parse import parse_method, parser_add_main_args
from manifolds import Optimizer
import wandb
import time
import pickle

import warnings
warnings.filterwarnings('ignore')

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

### Parse args ###
parser = argparse.ArgumentParser(description='Training Pipeline for Node Classification')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
dataset = load_dataset(args.data_dir, args.dataset, args.sub_dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

# get the splits for all runs
if args.rand_split:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.runs)]
elif args.rand_split_class:
    split_idx_lst = [dataset.get_idx_split(split_type='class', label_num_per_class=args.label_num_per_class)
                     for _ in range(args.runs)]
elif args.dataset in ['ogbn-proteins', 'ogbn-arxiv', 'ogbn-products']:
    split_idx_lst = [dataset.load_fixed_splits()
                     for _ in range(args.runs)]
else:
    split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset, protocol=args.protocol)

### Basic information of datasets ###
n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

degrees = compute_degrees(dataset.graph['edge_index'], dataset.graph['num_nodes'])
print("SUM IS "+str(degrees.sum()))
print(degrees)
print("SHAPE IS "+str(degrees.shape))
lowest_degree = degrees.min()
highest_degree = degrees.max()
print("HIGHEST DEGREE IS "+str(highest_degree.item()))
print("LOWEST DEGREE IS "+str(lowest_degree.item()))

sorted_degrees, _ = torch.sort(degrees)
percentile_index = int(len(sorted_degrees) * 0.8)
threshold = sorted_degrees[percentile_index]
print('Mean degree: {:.2f}'.format(degrees.float().mean().item()))
print('Std degree: {:.2f}'.format(degrees.float().std().item()))
print('Number of nodes with degree 0: {}'.format((degrees == 0).sum().item()))
print('Threshold: {:.2f}'.format(threshold))
# threshold = 0.2 * highest_degree
less_than_degree = (degrees <= threshold).sum().item()
greater_than_degree = (degrees > threshold).sum().item()
print(f"Number of nodes with degree less than {threshold}: {less_than_degree}, it accounts for {less_than_degree/degrees.shape[0]:.2f}", )
print(f"Number of nodes with degree greater than {threshold}: {greater_than_degree}, it accounts for {greater_than_degree/degrees.shape[0]:.2f}", )

# whether or not to symmetrize
if not args.directed and args.dataset != 'ogbn-proteins':
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)

dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)

### Load method ###
# model = parse_method(args, c, d, device)
if args.wandb_name == '0':
    from datetime import datetime
    now = datetime.now()
    timestamp = now.strftime("%d-%H%M")
    args.wandb_name = timestamp
### Loss function (Single-class, Multi-class) ###
if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

### Performance metric (Acc, AUC, F1) ###
if args.metric == 'rocauc':
    eval_func = eval_rocauc
elif args.metric == 'f1':
    eval_func = eval_f1
else:
    eval_func = eval_acc

logger = Logger(args.runs, args)

# model.train()
# print('MODEL:', model)
if args.wandb_name == '0':
    from datetime import datetime
    now = datetime.now()
    timestamp = now.strftime("%d-%H%M")
    args.wandb_name = timestamp

### Training loop ###
for run in range(args.runs):
    if args.use_wandb:
        wandb.init(project=f'HyperbolicFormer({args.dataset})', config=vars(args), name=f'{args.dataset}-Params-{args.wandb_name}-run-{run}')
    if args.dataset in ['cora', 'citeseer', 'pubmed'] and args.protocol == 'semi':
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    model = parse_method(args, c, d, device)  # init model
    model.reset_parameters()
    optimizer = Optimizer(model, args)

    best_val = float('-inf')

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        train_start = time.time()
        out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])
        if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
            if dataset.label.shape[1] == 1:
                true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
            else:
                true_label = dataset.label
            loss = criterion(out[train_idx], true_label.squeeze(1)[
                train_idx].to(torch.float))
        else:
            out = F.log_softmax(out, dim=1)
            loss = criterion(
                out[train_idx], dataset.label.squeeze(1)[train_idx])
        loss.backward()
        optimizer.step()

        print(f'🔥🔥 Epoch: {epoch:02d}, Loss: {loss:.4f} || Train Time: {time.time() - train_start:.2f}s')
        eval_start = time.time()
        if epoch % args.eval_step == 0:
            result = evaluate_large(model, dataset, split_idx, eval_func, criterion, args, degrees, threshold, device=device)
            logger.add_result(run, result)


            if epoch % args.display_step == 0:
                degrees_in_test = degrees[split_idx['test']]
                top_indices = split_idx['test'][degrees_in_test > threshold]
                bottom_indices = split_idx['test'][degrees_in_test <= threshold]

                max_top_acc = top_indices.shape[0] / split_idx['test'].shape[0]
                max_bottom_acc = bottom_indices.shape[0] / split_idx['test'].shape[0]
                print_str = f'Epoch: {epoch:02d}, ' + \
                            f'Loss: {loss:.4f}, ' + \
                            f'Train: {100 * result[0]:.2f}%, ' + \
                            f'Valid: {100 * result[1]:.2f}%, ' + \
                            f'Test: {100 * result[2]:.2f}%,  ' + \
                            f'Top: {100 * result[5]:.2f} | {100*max_top_acc:.2f}%,  ' +\
                            f'Bottom: {100 * result[6]:.2f} | {100*max_bottom_acc:.2f}%'
                print(print_str)
            if args.use_wandb:
                wandb.log({"run": run, "epoch": epoch, "loss": loss.item(), "train_acc": result[0],
                           "val_acc": result[1], "test_acc": result[2], "val_loss": result[3]})
    logger.print_statistics(run)

if args.use_wandb:
    wandb.finish()
results = logger.print_statistics()
logger.save(vars(args), results, f'results/{args.dataset}.csv')