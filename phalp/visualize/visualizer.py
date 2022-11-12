import numpy as np
import torch
import torch.nn as nn
from yacs.config import CfgNode as CN

from phalp.models.heads.mesh import *
from phalp.utils.utils import *
from phalp.utils.utils import perspective_projection
from phalp.visualize.py_renderer import Renderer


class Visualizer(nn.Module):
    
    def __init__(self, cfg, hmar):
        super(Visualizer, self).__init__()
        
        self.cfg = cfg
        self.hmar = hmar
        self.device = 'cuda'
        texture_file = np.load(self.cfg.SMPL.TEXTURE)
        self.faces_cpu = texture_file['smpl_faces'].astype('uint32')

        self.render = Renderer(focal_length=self.cfg.EXTRA.FOCAL_LENGTH, img_res=256, faces=self.faces_cpu)
        self.render_size = 256
        
    def render_single_frame(self, pose_embeddings, color, center=np.array([128, 128]), img_size = 256, scale = None, location=None, texture=None, image=None, use_image=False, render=True, engine="NMR"):
        
        if(scale is not None): pass
        else: scale = np.ones((pose_embeddings.size(0), 1))*256

        with torch.no_grad():
            pred_smpl_params, pred_cam, _ = self.hmar.smpl_head(pose_embeddings[:, 2048:].float())

        batch_size             = pose_embeddings.shape[0]
        dtype                  = pred_cam.dtype
        focal_length           = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=self.device, dtype=dtype)
 
        smpl_output            = self.hmar.smpl(**{k: v.float() for k,v in pred_smpl_params.items()}, pose2rot=False)
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

        if(render):
            rgb_from_pred, validmask = self.render.visualize_all(pred_vertices.cpu().numpy(), pred_cam_t_bs.cpu().numpy(), color, image, use_image=use_image)
            return rgb_from_pred, validmask, pred_keypoints_2d_smpl, pred_joints, pred_cam_t
        else:
            return 0, 0, pred_keypoints_2d_smpl, pred_joints, pred_cam_t
        
    def reset_render(self, image_size):
        self.render = Renderer(focal_length=self.cfg.EXTRA.FOCAL_LENGTH, img_res=image_size, faces=self.faces_cpu)
        self.render_size = image_size




