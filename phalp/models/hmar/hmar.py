import numpy as np
import torch
import torch.nn as nn

from phalp.models.backbones import resnet
from phalp.models.heads.apperence_head import TextureHead
from phalp.models.heads.encoding_head import EncodingHead
from phalp.models.heads.smpl_head import SMPLHead
from phalp.utils.smpl_utils import SMPL
from phalp.utils.utils import compute_uvsampler, perspective_projection


class HMAR(nn.Module):
    
    def __init__(self, cfg):
        super(HMAR, self).__init__()
       
        self.cfg = cfg

        nz_feat, tex_size    = 512, 6
        img_H, img_W         = 256, 256
        
        texture_file         = np.load(self.cfg.SMPL.TEXTURE)
        self.faces_cpu       = texture_file['smpl_faces'].astype('uint32')
        
        vt                   = texture_file['vt']
        ft                   = texture_file['ft']
        uv_sampler           = compute_uvsampler(vt, ft, tex_size=tex_size)
        uv_sampler           = torch.tensor(uv_sampler, dtype=torch.float)
        uv_sampler           = uv_sampler.unsqueeze(0)

        self.F               = uv_sampler.size(1)   
        self.T               = uv_sampler.size(2) 
        self.uv_sampler      = uv_sampler.view(-1, self.F, self.T*self.T, 2)
        self.backbone        = resnet(pretrained=True, num_layers=self.cfg.MODEL.BACKBONE.NUM_LAYERS, cfg=self.cfg)
        self.texture_head    = TextureHead(self.uv_sampler, self.cfg, img_H=img_H, img_W=img_W)
        self.encoding_head   = EncodingHead(cfg=self.cfg, img_H=img_H, img_W=img_W) 

        smpl_cfg             = {k.lower(): v for k,v in dict(cfg.SMPL).items()}
        self.smpl            = SMPL(**smpl_cfg)
        
        self.smpl_head       = SMPLHead(cfg, 
                                        input_dim=cfg.MODEL.SMPL_HEAD.IN_CHANNELS,
                                        pool='pooled')
        
    def load_weights(self, path):
        checkpoint_file = torch.load(path)
        state_dict_filt = {}
        for k, v in checkpoint_file['model'].items():
            if ("encoding_head" in k or "texture_head" in k or "backbone" in k or "smplx_head" in k): 
                state_dict_filt.setdefault(k[5:].replace("smplx", "smpl"), v)
        self.load_state_dict(state_dict_filt, strict=False)


    def forward(self, x):
        feats, skips    = self.backbone(x)
        flow            = self.texture_head(skips)
        uv_image        = self.flow_to_texture(flow, x)
        
        pose_embeddings = feats.max(3)[0].max(2)[0]
        pose_embeddings = pose_embeddings.view(x.size(0),-1)
        with torch.no_grad():
            pred_smpl_params, pred_cam, _ = self.smpl_head(pose_embeddings)

        out = {
            "uv_image"  : uv_image, # raw uv_image
            "uv_vector" : self.process_uv_image(uv_image), # preprocessed uv_image for the autoencoder
            "flow"      : flow,
            "pose_emb"  : pose_embeddings,
            "pose_smpl" : pred_smpl_params,
            "pred_cam"  : pred_cam,
        }
        return out    
    
    def process_uv_image(self, uv_image):
        uv_mask         = uv_image[:, 3:, :, :]
        uv_image        = uv_image[:, :3, :, :]/5.0
        zeros_          = uv_mask==0
        ones_           = torch.logical_not(zeros_)
        zeros_          = zeros_.repeat(1, 3, 1, 1)
        ones_           = ones_.repeat(1, 3, 1, 1)
        uv_image[zeros_]= 0.0
        uv_mask[zeros_[:, :1, :, :]] = -1.0
        uv_mask[ones_[:, :1, :, :]]  = 1.0
        uv_vector       = torch.cat((uv_image, uv_mask), 1)
        
        return uv_vector
    
    def flow_to_texture(self, flow_map, img_x):
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

    
    def get_3d_parameters(self, pred_smpl_params, pred_cam, center=np.array([128, 128]), img_size = 256, scale = None, ):
        
        if(scale is not None): 
            pass
        else: 
            scale = np.ones((pred_cam.size(0), 1))*256

        batch_size             = pred_cam.shape[0]
        dtype                  = pred_cam.dtype
        device                 = pred_cam.device
        focal_length           = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=dtype)
 
        smpl_output            = self.smpl(**{k: v.float() for k,v in pred_smpl_params.items()}, pose2rot=False)
        pred_joints            = smpl_output.joints

        pred_cam_t         = torch.stack([pred_cam[:,1], pred_cam[:,2], 2*focal_length[:, 0]/(pred_cam[:,0]*torch.tensor(scale[:, 0], dtype=dtype, device=device) + 1e-9)], dim=1)
        pred_cam_t[:, :2] += torch.tensor(center-img_size/2., dtype=dtype, device=device) * pred_cam_t[:, [2]] / focal_length

        zeros_  = torch.zeros(batch_size, 1, 3).to(device)
        pred_joints = torch.cat((pred_joints, zeros_), 1)

        camera_center          = torch.zeros(batch_size, 2)
        pred_keypoints_2d_smpl = perspective_projection(pred_joints, rotation=torch.eye(3,).unsqueeze(0).expand(batch_size, -1, -1).to(device),
                                                        translation=pred_cam_t.to(device),
                                                        focal_length=focal_length / img_size,
                                                        camera_center=camera_center.to(device))  

        pred_keypoints_2d_smpl = (pred_keypoints_2d_smpl+0.5)*img_size

        return pred_smpl_params, pred_keypoints_2d_smpl, pred_joints, pred_cam_t