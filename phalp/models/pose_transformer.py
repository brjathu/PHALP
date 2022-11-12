import torch.nn as nn

from .heads.smpl_head_prediction import SMPLHeadPrediction
from .transformers import RelationTransformerModel


class Pose_transformer(nn.Module):
    
    def __init__(self, cfg):
        super(Pose_transformer, self).__init__()
    
        self.relational            = RelationTransformerModel(cfg.MODEL.TRANSFORMER)  
        self.smpl_head_prediction  = SMPLHeadPrediction(cfg)      
    
