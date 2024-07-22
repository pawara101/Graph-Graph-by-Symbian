'''Causality in Traffic dataset test'''
import torch
import numpy as np
from models import *
from dataset_ccd import *
from torch.utils.data import DataLoader

import argparse
import os 

import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics.functional import pairwise_cosine_similarity
import scipy.io as io

import spacy

import sklearn 
from sklearn.metrics import confusion_matrix

import time
from sklearn.metrics import average_precision_score
from eval_utils import evaluation

from loaderData import CausalityInTrafficAccident

dataset_test  = CausalityInTrafficAccident(p, split='test', test_mode=True)