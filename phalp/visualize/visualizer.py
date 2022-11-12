import copy

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from torchvision.utils import make_grid
from yacs.config import CfgNode as CN

from phalp.models.heads.mesh import *
from phalp.utils.utils import *
from phalp.utils.utils import (get_colors, numpy_to_torch_image,
                               perspective_projection)
from phalp.visualize.py_renderer import Renderer

RGB_tuples = get_colors()

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

    def reset_render(self, image_size):
        self.render = Renderer(focal_length=self.cfg.EXTRA.FOCAL_LENGTH, img_res=image_size, faces=self.faces_cpu)
        self.render_size = image_size
        
    def render_single_frame(self, pred_smpl_params, pred_cam_t, color, img_size = 256, image=None, use_image=False):
        
        dtype = pred_cam_t.dtype
        batch_size = len(pred_cam_t)
        pred_cam_t = torch.tensor(pred_cam_t, device=self.device) 
        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=self.device)
        
        pred_smpl_params = default_collate(pred_smpl_params)
        smpl_output = self.hmar.smpl(**{k: v.float().cuda() for k,v in pred_smpl_params.items()}, pose2rot=False)
        pred_vertices = smpl_output.vertices
        pred_joints = smpl_output.joints
        
        # transform vertices to world coordinates
        pred_cam_t_bs = pred_cam_t.unsqueeze(1).repeat(1, pred_vertices.size(1), 1)
        verts = pred_vertices + pred_cam_t_bs

        pred_joints = torch.cat((pred_joints, torch.zeros(batch_size, 1, 3).cuda()), 1)
        pred_keypoints_2d_smpl = perspective_projection(pred_joints, rotation=torch.eye(3,).unsqueeze(0).expand(batch_size, -1, -1).cuda(),
                                                        translation=pred_cam_t.cuda(),
                                                        focal_length=focal_length / img_size,
                                                        camera_center=torch.zeros(batch_size, 2).cuda())  

        pred_keypoints_2d_smpl = (pred_keypoints_2d_smpl+0.5)*img_size

        rgb_from_pred, validmask = self.render.visualize_all(pred_vertices.cpu().numpy(), pred_cam_t_bs.cpu().numpy(), color, image, use_image=use_image)
        
        return rgb_from_pred, validmask, pred_keypoints_2d_smpl, pred_joints, pred_cam_t

    def render_video(self, final_visuals_dic):
        
        t_           = final_visuals_dic['time']
        cv_image     = final_visuals_dic['frame']
        tracked_ids  = final_visuals_dic["tid"]
        tracked_time = final_visuals_dic["tracked_time"]
        
        tracked_smpl = final_visuals_dic["smpl"]
        tracked_cameras = final_visuals_dic["camera"]

        img_height, img_width, _      = cv_image.shape
        new_image_size                = max(img_height, img_width)
    
        new_image_size_x              = self.cfg.render.res*self.cfg.render.up_scale
        ratio                         = 1.0*self.cfg.render.res/max(img_height, img_width)*self.cfg.render.up_scale
        
        delta_w                       = new_image_size - img_width
        delta_h                       = new_image_size - img_height
        top, bottom, left, right      = delta_h//2, delta_h-(delta_h//2), delta_w//2, delta_w-(delta_w//2)
        resized_image                 = cv2.copyMakeBorder(cv_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        resized_image_bbox            = copy.deepcopy(resized_image)
        resized_image_small           = cv2.resize(resized_image, (self.cfg.render.res*self.cfg.render.up_scale, self.cfg.render.res*self.cfg.render.up_scale))
        scale_                        = self.cfg.video.output_resolution/img_width
        frame_size                    = (self.cfg.video.output_resolution, int(img_height*(scale_)))

        rendered_image_1              = numpy_to_torch_image(np.array(resized_image_bbox)/255.)
        rendered_image_1x             = numpy_to_torch_image(np.array(resized_image_bbox)/255.)
        
        if(len(tracked_ids)>0):
            tracked_time              = np.array(tracked_time)
            tracked_smpl              = np.array(tracked_smpl)
            tracked_cameras           = np.array(tracked_cameras)
            tracked_cameras[:, 2]     = tracked_cameras[:, 2]/self.cfg.render.up_scale
        
            if "HUMAN" in self.cfg.render.type:
                ids_x = tracked_time==0
            elif "GHOST" in self.cfg.render.type:
                ids_x = tracked_time>=0

            tracked_ids_x             = np.array(tracked_ids)
            tracked_ids_x             = tracked_ids_x[ids_x]        
            rendered_image_1x         = rendered_image_1x[:, :, top:top+img_height, left:left+img_width]

            resized_image_small             = cv2.resize(resized_image, (self.cfg.render.res*self.cfg.render.up_scale, self.cfg.render.res*self.cfg.render.up_scale))
            if(len(tracked_ids_x)>0):
                
                rendered_image_3, valid_mask, _, _, _  = self.render_single_frame(
                                                                                    tracked_smpl[ids_x],
                                                                                    tracked_cameras[ids_x],
                                                                                    np.array(RGB_tuples[list(tracked_ids_x)])/255.0, 
                                                                                    img_size   = new_image_size_x, 
                                                                                    image      = (0*resized_image_small)/255.0, 
                                                                                    use_image  = True,
                                                                                    )

                rendered_image_3           = cv2.resize(rendered_image_3, (max(img_height, img_width), max(img_height, img_width)))
                rendered_image_3           = numpy_to_torch_image(np.array(rendered_image_3))[:, :, top:top+img_height, left:left+img_width]

                valid_mask                 = np.repeat(valid_mask, 3, 2)
                valid_mask                 = np.array(valid_mask, dtype=int)
                valid_mask                 = np.array(valid_mask, dtype=float)
                valid_mask                 = cv2.resize(valid_mask, (max(img_height, img_width), max(img_height, img_width)))
                valid_mask                 = numpy_to_torch_image(np.array(valid_mask))[:, :, top:top+img_height, left:left+img_width]

                loc_b = valid_mask==1
                rendered_image_5x             = copy.deepcopy(rendered_image_1x)#*0 + 1
                rendered_image_5x[loc_b]      = 0
                rendered_image_3[torch.logical_not(loc_b)] = 0
                rendered_image_3              = rendered_image_3 + rendered_image_5x
                
                
            else:
                rendered_image_3              = copy.deepcopy(rendered_image_1x)

        else:
            rendered_image_3              = copy.deepcopy(rendered_image_1x)
            rendered_image_1              = rendered_image_1[:, :, top:top+img_height, left:left+img_width]

        grid_img = make_grid(torch.cat([rendered_image_3], 0), nrow=10)
        grid_img = grid_img[[2,1,0], :, :]
        ndarr    = grid_img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        cv_ndarr = cv2.resize(ndarr, frame_size)
        cv2.putText(cv_ndarr, str(t_), (20,20), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,255,255))

        return cv_ndarr, frame_size


