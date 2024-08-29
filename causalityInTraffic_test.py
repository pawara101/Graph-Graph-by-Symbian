import argparse,sys , os #pickle, os, #math, random, sys, time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

parent_dir = '/home/pawarad/Data-science/Neural-networks/Graph-Graph'
sys.path.append(parent_dir)
from models import *
from train_dad import test_model

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loaderData import CausalityInTrafficAccident

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
print(args)

p['device'] = 0
print(p)
dataset_test  = CausalityInTrafficAccident(p, split='test', test_mode=True)

print(dataset_test)

dataloader_test = DataLoader(dataset_test, batch_size=p['batch_size'])
print("Dataset Length :", len(dataset_test))

print("====")
# print(next(iter(dataloader_test)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpaceTempGoG_detr_dad(input_dim=4096, embedding_dim=256, img_feat_dim=2048, num_classes=2).to(device) ## Pre-trained model

model_pretrained = torch.load("model_checkpoints/dad/SpaceTempGoG_detr_dad_6.pth")

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
'''----------------------------------------------------------------'''
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


#####################################################################
# process_epoch
#####################################################################
def process_epoch(phase, _epoch, p, _dataloader, _model, _optim=None):
    losses = AverageMeter()
    top1_c = AverageMeter()
    top2_c = AverageMeter()
    top1_e = AverageMeter()
    top2_e = AverageMeter()
    top1_all = AverageMeter()
    
    if(phase == 'train'):
        _model.train()
    elif(phase == 'val'):
        _model.eval()
    elif(phase == 'test'):
        _model.eval()
        state_dict = torch.load(p['logdir'] + 'model_max.pth')
        _model.load_state_dict(state_dict)

    for iter, _data in enumerate(_dataloader):
        feat_rgb, label_cause, label_effect = _data
        batch_size = feat_rgb.size(0)
        if(phase=='train'):
            _optim.zero_grad()

        loss, logits = _model.forward_all(feat_rgb.cuda(), [label_cause.cuda(), label_effect.cuda()])

        if(phase=='train'):
            loss.backward()
            _optim.step()

        # measure accuracy and record loss
        prec1_c, prec2_c = accuracy(logits[0], label_cause.cuda(), topk=(1,2))
        prec1_e, prec2_e = accuracy(logits[1], label_effect.cuda(), topk=(1,2))

        losses.update(loss.item(), batch_size)
        top1_c.update(prec1_c.item(), batch_size)
        top2_c.update(prec2_c.item(), batch_size)
        top1_e.update(prec1_e.item(), batch_size)
        top2_e.update(prec2_e.item(), batch_size)

        stats = dict()
        stats['loss'] = losses.avg
        stats['top1.cause'] = top1_c.avg
        stats['top2.cause'] = top2_c.avg
        stats['top1.effect'] = top1_e.avg
        stats['top2.effect'] = top2_e.avg
        return stats


for batch_i, _data in enumerate(dataloader_test):
    feat_rgb, label_cause, label_effect = _data
    batch_size = feat_rgb.size(0)
    print("Batch Size :", batch_size)
    print("RGB Features :", feat_rgb.shape)
    print("Label Cause :", label_cause.shape)

    loss, logits = model.forward(feat_rgb.cuda(), [label_cause.cuda(), label_effect.cuda()])

    stats_test = process_epoch('test', batch_i, p, dataloader_test, model)
    print(stats_test)