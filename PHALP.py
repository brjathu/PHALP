import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import DataLoader
from torchvision.utils import save_image, make_grid

import os
import json
import copy
import math
import heapq
from PIL import Image
import argparse
import cv2
import numpy as np
from yacs.config import CfgNode as CN
from collections import OrderedDict
from scipy.linalg import lstsq
from sklearn.linear_model import Ridge
import scipy as sp
import scipy.stats as stats
from scipy.stats import t

from models.hmar import HMAR
from models.utils import *
from utils.utils import get_prediction_interval


class PHALP_tracker(nn.Module):

    def __init__(self, opt):
        super(PHALP_tracker, self).__init__()

        self.opt        = opt
        
        key_            = 0
        config          = os.path.join('utils/config.yaml')
        checkpoint      = '_DATA/hmar_weights_1.pt'
        self.config     = config
        self.HMAR       = HMAR(config, self.opt)
        checkpoint_file = torch.load(checkpoint)
        state_dict_filt = {k[key_:]: v for k, v in checkpoint_file['model'].items() if not("perceptual_loss" in k)}

        checkpoint      = '_DATA/hmar_weights_2.pth'
        checkpoint_file = torch.load(checkpoint)
        for k, v in checkpoint_file['model'].items():
            if ("encoding_head" in k): state_dict_filt.setdefault(k[5:], v)
            
        self.HMAR.load_state_dict(state_dict_filt, strict=False)
        self.HMAR.cuda()
        self.HMAR.eval()
        self.device = torch.device('cuda')

    def forward_for_tracking(self, vectors, attibute="A", time=1):
        
        if(attibute=="P"):

            vectors_pose         = vectors[0]
            vectors_data         = vectors[1]
            vectors_time         = vectors[2]

            en_pose              = torch.from_numpy(vectors_pose)
            en_data              = torch.from_numpy(vectors_data)
            en_time              = torch.from_numpy(vectors_time)

            if(len(en_pose.shape)!=3):
                en_pose          = en_pose.unsqueeze(0)         
                en_time          = en_time.unsqueeze(0)         
                en_data          = en_data.unsqueeze(0)         

            BS                   = en_pose.size(0)
            history              = en_pose.size(1)
            attn                 = torch.ones(BS, history, history)
            xf_trans             = self.HMAR.pose_transformer.relational(en_pose[:, :, 2048:].float().cuda(), en_data.float().cuda(), attn.float().cuda())  #bs, 13, 2048
            xf_trans             = xf_trans.view(-1, 2048)
            movie_strip_t        = self.HMAR.pose_transformer.smplx_head_prediction(en_pose[:, :, 2048:].float().view(-1, 2048).cuda(), xf_trans)  #bs*13, 2048 -> bs*13, 12, 2048
            movie_strip_t        = movie_strip_t.view(BS, history, 12, 2048)
            xf_trans             = xf_trans.view(BS, history, 2048)

            time[time>11]=11
            pose_pred = []
            for i in range(len(time)):
                pose_pred.append(movie_strip_t[i, -1, time[i], :])
            pose_pred = torch.stack(pose_pred)
            en_pose_x            = torch.cat((xf_trans[:, -1, :], pose_pred), 1)

            return en_pose_x.cpu()



        if(attibute=="L"):
            vectors_loca         = vectors[0]
            vectors_time         = vectors[1]
            vectors_conf         = vectors[2]

            en_loca              = torch.from_numpy(vectors_loca)
            en_time              = torch.from_numpy(vectors_time)
            en_conf              = torch.from_numpy(vectors_conf)
            time                 = torch.from_numpy(time)

            if(len(en_loca.shape)!=3):
                en_loca          = en_loca.unsqueeze(0)             
                en_time          = en_time.unsqueeze(0)             
            else:
                en_loca          = en_loca.permute(0, 1, 2)         

            BS = en_loca.size(0)
            t_ = en_loca.size(1)

            en_loca_xy           = en_loca[:, :, :90]
            en_loca_xy           = en_loca_xy.view(BS, t_, 45, 2)
            en_loca_n            = en_loca[:, :, 90:]
            en_loca_n            = en_loca_n.view(BS, t_, 3, 3)

            if(self.opt.cva_type == "least_square"):
                new_en_loca_n = []
                for bs in range(BS):
                    x0_                  = np.array(en_loca_xy[bs, :, 44, 0])
                    y0_                  = np.array(en_loca_xy[bs, :, 44, 1])

                    x_                   = np.array(en_loca_n[bs, :, 0, 0])
                    y_                   = np.array(en_loca_n[bs, :, 0, 1])
                    n_                   = np.log(np.array(en_loca_n[bs, :, 0, 2]))
                    t_                   = np.array(en_time[bs, :])
                    n                    = len(t_)

                    loc_                 = torch.diff(en_time[bs, :], dim=0)!=0
                    loc_                 = torch.sum(loc_)

                    M = t_[:, np.newaxis]**[0, 1]
                    time_ = 48 if time[bs]>48 else time[bs]

                    clf = Ridge(alpha=5.0)
                    clf.fit(M, n_)
                    n_p = clf.predict(np.array([1, time_+1+t_[-1]]).reshape(1, -1))
                    n_p = n_p[0]
                    n_hat = clf.predict(np.hstack((np.ones((t_.size, 1)), t_.reshape((-1, 1)))))
                    n_pi  = get_prediction_interval(n_, n_hat, t_, time_+1+t_[-1])


                    clf  = Ridge(alpha=1.0)
                    clf.fit(M, x0_)
                    x_p  = clf.predict(np.array([1, time_+1+t_[-1]]).reshape(1, -1))
                    x_p  = x_p[0]
                    x_p_ = (x_p-0.5)*np.exp(n_p)/5000.0*256.0
                    x_hat = clf.predict(np.hstack((np.ones((t_.size, 1)), t_.reshape((-1, 1)))))
                    x_pi  = get_prediction_interval(x0_, x_hat, t_, time_+1+t_[-1])

                    clf  = Ridge(alpha=1.0)
                    clf.fit(M, y0_)
                    y_p  = clf.predict(np.array([1, time_+1+t_[-1]]).reshape(1, -1))
                    y_p  = y_p[0]
                    y_p_ = (y_p-0.5)*np.exp(n_p)/5000.0*256.0
                    y_hat = clf.predict(np.hstack((np.ones((t_.size, 1)), t_.reshape((-1, 1)))))
                    y_pi  = get_prediction_interval(y0_, y_hat, t_, time_+1+t_[-1])

                    new_en_loca_n.append([x_p_, y_p_, np.exp(n_p), x_pi, y_pi, np.exp(n_pi), 1, 1, 0])
                    en_loca_xy[bs, -1, 44, 0] = x_p
                    en_loca_xy[bs, -1, 44, 1] = y_p

                new_en_loca_n        = torch.from_numpy(np.array(new_en_loca_n))
                xt                   = torch.cat((en_loca_xy[:, -1, :, :].view(BS, 90), (new_en_loca_n.float()).view(BS, 9)), 1)

        return xt

    
    def get_uv_distance2(self, t_uv, d_uv):
        t_uv         = torch.from_numpy(t_uv).cuda().float()
        d_uv         = torch.from_numpy(d_uv).cuda().float()
        t_uv         = t_uv.repeat(d_uv.shape[0], 1, 1, 1)
        
        d_mask       = d_uv[:, 3:, :, :]>0.0
        t_mask       = t_uv[:, 3:, :, :]>0.0
        
            
        mask         = torch.logical_and(d_mask, t_mask)
        mask         = mask.repeat(1, 4, 1, 1)
        mask_        = torch.logical_not(mask)

        t_uv[mask_]                           = 0.0
        t_uv[:, 3:, :, :][mask_[:, 3:, :, :]] = -1.0
        t_uv[:, 3:, :, :][mask[:, 3:, :, :]]  = 1.0

        d_uv[mask_]                           = 0.0
        d_uv[:, 3:, :, :][mask_[:, 3:, :, :]] = -1.0
        d_uv[:, 3:, :, :][mask[:, 3:, :, :]]  = 1.0

        with torch.no_grad():
            t_emb    = self.HMAR.autoencoder_hmar(t_uv, en=True)
            d_emb    = self.HMAR.autoencoder_hmar(d_uv, en=True)
        
        t_emb        = t_emb.view(t_emb.shape[0], -1)/10**3
        d_emb        = d_emb.view(d_emb.shape[0], -1)/10**3
                
        return t_emb.cpu().numpy(), d_emb.cpu().numpy(), torch.sum(mask).cpu().numpy()/4/256/256/2
