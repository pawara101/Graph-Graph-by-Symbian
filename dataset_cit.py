from data.loader import CausalityInTrafficAccident
import torch
from torch.utils.data import Dataset, DataLoader
import argparse, os
from data.utils_cit import *


parser = argparse.ArgumentParser(description='Training Framework for Cause and Effect Event Classification')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--feature', type=str, default="i3d-rgb-x8")

parser.add_argument('--input_size', type=int, default=1024)
parser.add_argument('--hidden_size', type=int, default=256)

parser.add_argument('--loss_type', type=str, default='CrossEntropy')
parser.add_argument('--num_experiments', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=2000)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--use_dropout', type=float, default=0.5)

parser.add_argument('--architecture_type', type=str, default='TSN')
parser.add_argument('--consensus_type', type=str, default='average')
parser.add_argument('--num_segments', type=int, default=4)
parser.add_argument('--new_length', type=int, default=1)

parser.add_argument('--dataset_ver', type=str, default='Mar9th')
parser.add_argument('--feed_type', type=str, default='classification')
parser.add_argument('--logdir', type=str, default='runs')

parser.add_argument("--random_seed", type=int, default=0)

args = parser.parse_args()

p = vars(args)
p['device'] = 0
dataset_test  = CausalityInTrafficAccident(p, split='test', test_mode=True)

device = p['device']
dataloader_test = DataLoader(dataset_test, batch_size=p['batch_size'])

print("done")

print("--------------------------------")
print(len(dataset_test))
print(dataloader_test.batch_size)
print(dataloader_test.batch_sampler)
print(type(dataloader_test))


# # Iterate through the DataLoader and print the data
# for i, batch in enumerate(dataloader_test):
#     print(f"Batch {i+1}")
#     print("Batch shape:", batch.shape)
    
#     # Optionally break after the first batch to avoid printing too much data
#     if i == 0:
#         break



# Iterate through the DataLoader and print the data
for i, batch in enumerate(dataset_test):
    print(f"Batch {i+1}")
    print("Batch type:", type(batch))
    
    # Iterate over items in the batch
    for item in batch:
        print("Item type:", type(item))
        if isinstance(item, torch.Tensor):
            print("Item shape:", item.shape)
            print("Item content:", item)
        else:
            print("Item content:", item)
    
    # Optionally break after the first batch to avoid printing too much data
    if i == 0:
        break
