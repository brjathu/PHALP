import numpy as np
import torch
import torch.nn as nn
from yacs.config import CfgNode as CN

from phalp.models.backbones import resnet
from phalp.models.heads.apperence_head import TextureHead
from phalp.models.heads.encoding_head import EncodingHead
from phalp.models.heads.mesh import *
from phalp.models.heads.smpl_head import SMPLHead
from phalp.models.joint_mapper import JointMapper, smpl_to_openpose
from phalp.models.pose_transformer import Pose_transformer
from phalp.models.smplx import create
from phalp.utils.utils import *
from phalp.utils.utils import perspective_projection


class HMAR(nn.Module):
    
    def __init__(self, cfg):
        super(HMAR, self).__init__()
       
        self.cfg = cfg

        nz_feat  = 512; tex_size = 6
        img_H    = 256; img_W    = 256
        
        texture_file         = np.load(self.cfg.SMPL.TEXTURE)
        self.faces_cpu       = texture_file['smpl_faces'].astype('uint32')
        
        vt                   = texture_file['vt']
        ft                   = texture_file['ft']
        uv_sampler           = compute_uvsampler(vt, ft, tex_size=6)
        uv_sampler           = torch.tensor(uv_sampler, dtype=torch.float)
        uv_sampler           = uv_sampler.unsqueeze(0)

        self.F               = uv_sampler.size(1)   
        self.T               = uv_sampler.size(2) 
        self.uv_sampler      = uv_sampler.view(-1, self.F, self.T*self.T, 2)
        self.backbone        = resnet(pretrained=True, num_layers=self.cfg.MODEL.BACKBONE.NUM_LAYERS, cfg=self.cfg)
        self.texture_head    = TextureHead(self.uv_sampler, self.cfg, img_H=img_H, img_W=img_W)
        self.encoding_head   = EncodingHead(cfg=self.cfg, img_H=img_H, img_W=img_W) 
    
        smpl_params  = {k.lower(): v for k,v in dict(cfg.SMPL).items()}
        joint_mapper = JointMapper(smpl_to_openpose(model_type=cfg.SMPL.MODEL_TYPE))
        self.smpl    = create(**smpl_params,
                                  batch_size=1,
                                  joint_mapper = joint_mapper,
                                  create_betas=False,
                                  create_body_pose=False,
                                  create_global_orient=False,
                                  create_left_hand_pose=False,
                                  create_right_hand_pose=False,
                                  create_expression=False,
                                  create_leye_pose=False,
                                  create_reye_pose=False,
                                  create_jaw_pose=False,
                                  create_transl=False)
        
        self.smpl_head            = SMPLHead(cfg)
        self.smpl_head.pool       = 'pooled'
        self.device               = "cuda"
        
        if("P" in self.cfg.phalp.predict):
            self.pose_transformer     = Pose_transformer(self.cfg)
            checkpoint_file = torch.load("_DATA/hmmr_v2_weights.pt")
                
            state_dict_filt = {k[11:]: v for k, v in checkpoint_file['model'].items() if ("relational" in k)}  
            self.pose_transformer.relational.load_state_dict(state_dict_filt, strict=True)

            state_dict_filt = {k[18:]: v for k, v in checkpoint_file['model'].items() if ("smplx_head_future" in k)}  
            self.pose_transformer.smpl_head_prediction.load_state_dict(state_dict_filt, strict=False)


    def forward(self, x):
        feats, skips    = self.backbone(x)
        flow            = self.texture_head(skips)
        uv_image        = self.flow_to_texture(flow, x)
        
        pose_embeddings = feats.max(3)[0].max(2)[0]
        pose_embeddings = pose_embeddings.view(x.size(0),-1)
        with torch.no_grad():
            pred_smpl_params, _, _ = self.smpl_head(pose_embeddings)

        out = {
            "uv_image"  : uv_image,
            "flow"      : flow,
            "pose_emb"  : pose_embeddings,
            "pose_smpl" : pred_smpl_params,
        }
        return out    
    
    def flow_to_texture(self, flow_map, img_x):
        batch_size = flow_map.size(0)
        flow_map   = flow_map.permute(0,2,3,1)
        uv_images  = torch.nn.functional.grid_sample(img_x, flow_map)
        return uv_images
    
    def autoencoder_hmar(self, x, en=True):
        if(en==True):
            if(self.cfg.phalp.encode_type=="3c"):
                return self.encoding_head(x[:, :3, :, :], en=en)
            else:
                return self.encoding_head(x, en=en)
        else:
            return self.encoding_head(x, en=en)

    
    def get_3d_parameters(self, pose_embeddings, color, center=np.array([128, 128]), img_size = 256, scale = None, location=None, texture=None, image=None, use_image=False, render=True, engine="NMR"):
        
        if(scale is not None): pass
        else: scale = np.ones((pose_embeddings.size(0), 1))*256

        with torch.no_grad():
            pred_smpl_params, pred_cam, _ = self.smpl_head(pose_embeddings[:, 2048:].float())

        batch_size             = pose_embeddings.shape[0]
        dtype                  = pred_cam.dtype
        focal_length           = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=self.device, dtype=dtype)
 
        smpl_output            = self.smpl(**{k: v.float() for k,v in pred_smpl_params.items()}, pose2rot=False)
        pred_vertices          = smpl_output.vertices
        pred_joints            = smpl_output.joints
        
        if(location is not None):
            pred_cam_t         = torch.tensor(location*1.0, dtype=dtype, device=self.device) #location
        else:
            pred_cam_t         = torch.stack([pred_cam[:,1], pred_cam[:,2], 2*focal_length[:, 0]/(pred_cam[:,0]*torch.tensor(scale[:, 0], dtype=dtype, device=self.device) + 1e-9)], dim=1)
            pred_cam_t[:, :2] += torch.tensor(center-img_size/2., dtype=dtype, device=self.device) * pred_cam_t[:, [2]] / focal_length

        # transform vertices to world coordinates
        pred_cam_t_bs         = pred_cam_t.unsqueeze(1).repeat(1, pred_vertices.size(1), 1)
        verts                 = pred_vertices + pred_cam_t_bs

        mask_model = []
        loc_       = 0
        zeros_  = torch.zeros(batch_size, 1, 3).cuda()
        pred_joints = torch.cat((pred_joints, zeros_), 1)

        camera_center          = torch.zeros(batch_size, 2)
        pred_keypoints_2d_smpl = perspective_projection(pred_joints, rotation=torch.eye(3,).unsqueeze(0).expand(batch_size, -1, -1).cuda(),
                                                        translation=pred_cam_t.cuda(),
                                                        focal_length=focal_length / img_size,
                                                        camera_center=camera_center.cuda())  

        pred_keypoints_2d_smpl = (pred_keypoints_2d_smpl+0.5)*img_size

        return pred_smpl_params, pred_keypoints_2d_smpl, pred_joints, pred_cam_t