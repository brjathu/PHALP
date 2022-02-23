import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

import os
import json
import copy
import heapq
from PIL import Image
from tqdm import tqdm
import argparse
import cv2
import numpy as np
import argparse

from .heads.mesh import *
from .heads.smplx_head_prediction import SMPLXHeadPrediction
from .transformers import RelationTransformerModel


from yacs.config import CfgNode as CN
import neural_renderer as nr
from .renderer import Renderer


from .utils import *

class Pose_transformer(nn.Module):
    
    def __init__(self, opt):
        super(Pose_transformer, self).__init__()
        
        config = "utils/config.yaml"
        with open(config, 'r') as f:
            cfg = CN.load_cfg(f); cfg.freeze()

        self.cfg                   = cfg
        self.relational            = RelationTransformerModel(cfg.MODEL.TRANSFORMER)  
        self.smplx_head_prediction = SMPLXHeadPrediction(cfg)      
    