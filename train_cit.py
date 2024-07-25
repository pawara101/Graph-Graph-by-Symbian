import argparse, sys, os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from models import *
from loaderData import CausalityInTrafficAccident
from causality_in_traffic.utils import *

parent_dir = '/home/pawarad/Data-science/Neural-networks/GG_on_New_data'
sys.path.append(parent_dir)
MODEL_SAVE_PATH = 'model_checkpoints/dad/SpaceTempGoG_detr_dad_6.pth'
PICKLE_MODULE = "/home/pawarad/Data-science/Neural-networks/Graph-Graph/data/CausalityInTrafficAccident/annotation-Mar9th-25fps.pkl"

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
print(p)

dataset_test = CausalityInTrafficAccident(p, split='test', test_mode=True)
print(dataset_test)
dataloader_test = DataLoader(dataset_test, batch_size=p['batch_size'])
print("Dataset Length :", len(dataset_test))
print("------------------")
print(next(iter(dataloader_test)))
print("------------------")

## Select Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_load = torch.load(f=MODEL_SAVE_PATH)
# print("Loaded model :",model_load)

model = SpaceTempGoG_detr_dad(input_dim=4096, embedding_dim=256, img_feat_dim=2048, num_classes=2).to(device)
# Load in the saved state_dict()
model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))


for di in range(0, args.num_experiments):

    stats_test = process_epoch('test', args.num_epochs, p, dataloader_test, model)
    print(stats_test)

