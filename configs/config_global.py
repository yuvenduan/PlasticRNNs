import os.path as osp
import logging
import torch

NP_SEED = 1234
TCH_SEED = 2147483647

# config for ConvRNNs
ROOT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
FIG_DIR = osp.join(ROOT_DIR, 'figures')

# device to run algorithm on
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
MAP_LOC = "cuda:0" if USE_CUDA else torch.device('cpu')
LOG_LEVEL = logging.INFO