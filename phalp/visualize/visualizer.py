import copy

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data._utils.collate import default_collate
from torchvision.utils import make_grid

from phalp.utils.utils import (get_colors, numpy_to_torch_image,
                               perspective_projection)
from phalp.visualize.py_renderer import Renderer


def rect_with_opacity(image, top_left, bottom_right, fill_color, fill_opacity):
    with_fill = image.copy()
    with_fill = cv2.rectangle(with_fill, top_left, bottom_right, fill_color, cv2.FILLED)
    return cv2.addWeighted(with_fill, fill_opacity, image, 1 - fill_opacity, 0, image)

class Visualizer(nn.Module):
    
    def __init__(self, cfg, hmar):
        super(Visualizer, self).__init__()
        
        self.cfg = cfg
        self.hmar = hmar
        self.device = 'cuda'
        if(not(self.cfg.render.head_mask)):
            texture_file = np.load(self.cfg.SMPL.TEXTURE)
            self.faces_cpu = texture_file['smpl_faces'].astype('uint32')
        else:
            texture_file = np.load(self.cfg.render.head_mask_path)
            self.faces_cpu = texture_file.astype('uint32')
            
        self.render = Renderer(focal_length=self.cfg.EXTRA.FOCAL_LENGTH, img_res=256, faces=self.faces_cpu)
        self.render_size = 256
        
        self.colors = get_colors(pallette=self.cfg.render.colors)

        if(self.cfg.render.blur_faces):
            try:
                from facenet_pytorch import MTCNN
            except:
                raise ValueError('''
                Please install facenet_pytorch to use blur_faces option. 
                `pip install facenet_pytorch` or `pip install -e . [blur]`''')
            self.face_detector = MTCNN(keep_all=True, device="cuda")

    def reset_render(self, image_size):
        del self.render
        self.render = None
        self.render = Renderer(focal_length=self.cfg.EXTRA.FOCAL_LENGTH, img_res=image_size, faces=self.faces_cpu, 
                               metallicFactor=self.cfg.render.metallicfactor, roughnessFactor=self.cfg.render.roughnessfactor)
        self.render_size = image_size      
        
    def render_single_frame(self, pred_smpl_params, pred_cam_t, color, img_size = 256, image=None, use_image=False):
                
        pred_smpl_params = default_collate(pred_smpl_params)
        smpl_output = self.hmar.smpl(**{k: v.float().cuda() for k,v in pred_smpl_params.items()}, pose2rot=False)
        pred_vertices = smpl_output.vertices.cpu()

        # a1 = pred_vertices[0, :, 2]>0.1
        # a2 = a1.nonzero()
        # new_faces = []
        # for f_ in self.faces_cpu:
        #     if f_[0] in a2 and f_[1] in a2 and f_[2] in a2:
        #         new_faces.append(f_)
        # new_faces = np.array(new_faces)
        # self.render.faces = new_faces

        pred_cam_t = torch.tensor(pred_cam_t, device=self.device) 
        pred_cam_t_bs = pred_cam_t.unsqueeze(1).repeat(1, pred_vertices.size(1), 1)

        rgb_from_pred, validmask = self.render.visualize_all(pred_vertices.numpy(), pred_cam_t_bs.cpu().numpy(), color, image, use_image=use_image)
        
        return rgb_from_pred, validmask
    
    def draw_text(self, image, text, xy, bg_color=(255, 255, 255)):

        # get top corner of mask
        x, y = xy[0], xy[1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        ((txt_w, txt_h), _) = cv2.getTextSize(text, font, font_scale, 1)
        
        # Place text background.
        back_tl = int(x), int(y)
        back_br = int(x) + int(txt_w), int(y) + int(1.3 * txt_h)
        txt_tl = int(x), int(y) + int(1 * txt_h)
        
        # draw text on top of mask
        image = rect_with_opacity(image, back_tl, back_br, bg_color, 0.6)
        
        cv2.putText(image, text, txt_tl, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

        return image

    def visualize_mask(self, image, mask, bbox, color, text, 
                 alpha=0.5, show_border=True, border_alpha=0.8, 
                 border_thick=2, border_color=None):
        """Visualizes a single binary mask."""
        
        # draw border with bbox
        cv_color = np.array([color[2],color[1],color[0]])
        try:
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), cv_color, border_thick)
        except:
            image = image.astype(np.uint8)
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), cv_color, border_thick)
        
        image = image.astype(np.float32)
        mask = mask[:, :, None]
        idx = np.nonzero(mask)

        if ("MASK" in self.cfg.render.type):
            image[idx[0], idx[1], :] *= 1.0 - alpha
            image[idx[0], idx[1], :] += [alpha * x for x in cv_color]
            
            if border_alpha == 0:
                return

            if border_color is None:
                border_color = [x * 0.5 for x in cv_color]
            if isinstance(border_color, np.ndarray):
                border_color = border_color.tolist()

            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if border_alpha < 1:
                with_border = image.copy()
                cv2.drawContours(with_border, contours, -1, border_color, border_thick, cv2.LINE_AA)
                image = ((1 - border_alpha) * image + border_alpha * with_border)
            else:
                cv2.drawContours(image, contours, -1, border_color, border_thick, cv2.LINE_AA)
        
        image = self.draw_text(image, text, [bbox[0], bbox[1]])
        
        return image.astype(np.uint8)
    
    def visualize_labels_bbox(self, image, labels, bbox, color, track_ids, tracked_smpl, tracked_camera):
        
        # convert to numpy
        grid_img = make_grid(image, nrow=10)
        grid_img = grid_img[[2,1,0], :, :]
        ndarr    = grid_img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        cv_ndarr = cv2.resize(ndarr, ndarr.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

        # draw bounding boxes and print labels
        for i, box in enumerate(bbox):
            cv_color = np.array([color[i][2],color[i][1],color[i][0]])*255
            cv2.rectangle(cv_ndarr, (int(box[0]), int(box[1])), (int(box[0]+box[2]), int(box[1]+box[3])), cv_color, 2)
            for j, label in enumerate(labels[track_ids[i]][::-1]):
                 cv_ndarr = self.draw_text(cv_ndarr, label, [box[0], box[1]-(j+1)*20])

        return numpy_to_torch_image(cv_ndarr)/255.
    
    def visualize_labels_arrow(self, image, labels, bbox, color, track_ids, tracked_smpl, tracked_camera):

        # convert to numpy
        grid_img = make_grid(image, nrow=10)
        grid_img = grid_img[[2,1,0], :, :]
        ndarr    = grid_img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        cv_ndarr = cv2.resize(ndarr, ndarr.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

        # show labels on the top near head.
        img_w, img_h = cv_ndarr.shape[1], cv_ndarr.shape[0]
        img_size = max(img_w, img_h)
        if(len(tracked_smpl)==0):
            return numpy_to_torch_image(cv_ndarr)/255.

        tracked_smpl = default_collate(tracked_smpl)
        smpl_output = self.hmar.smpl(**{k: v.float().cuda() for k,v in tracked_smpl.items()}, pose2rot=False)
        pred_joints = smpl_output.joints
        batch_size = pred_joints.shape[0]
        camera_center          = torch.zeros(batch_size, 2, device=self.device, dtype=pred_joints.dtype)
        focal_length           = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=self.device, dtype=pred_joints.dtype)
        pred_keypoints_2d_smpl = perspective_projection(pred_joints, rotation=torch.eye(3,).unsqueeze(0).expand(batch_size, -1, -1).cuda(),
                                                        translation=torch.from_numpy(tracked_camera).float().cuda(),
                                                        focal_length=focal_length / img_size,
                                                        camera_center=camera_center)  

        pred_keypoints_2d_smpl = (pred_keypoints_2d_smpl+0.5)*img_size
        pred_keypoints_2d_smpl[:, :, 0] -= (img_size - img_w) / 2
        pred_keypoints_2d_smpl[:, :, 1] -= (img_size - img_h) / 2

        # # draw keypoints
        if(self.cfg.render.show_keypoints):
            for i, box in enumerate(bbox):
                cv_color = np.array([color[i][2],color[i][1],color[i][0]])*255
                for j, keypoint in enumerate(pred_keypoints_2d_smpl[i]):
                    cv2.circle(cv_ndarr, (int(keypoint[0]), int(keypoint[1])), 2, cv_color, 2)
                    cv2.putText(cv_ndarr, str(j), (int(keypoint[0]), int(keypoint[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cv_color, 1, cv2.LINE_AA)

        # find the direction of the head and draw the label on the top of the head.
        for i, box in enumerate(bbox):
            cv_color = np.array([color[i][2],color[i][1],color[i][0]])*255
            v_head = pred_keypoints_2d_smpl[i][43] - pred_keypoints_2d_smpl[i][8]
            v_head_norm = v_head / np.linalg.norm(v_head.cpu().numpy())
            v_head_norm = v_head_norm.cpu().numpy()

            # draw line from head to 10% of the norm
            line_start = pred_keypoints_2d_smpl[i][43].cpu().numpy() + v_head_norm*0.1*np.linalg.norm(v_head.cpu().numpy())
            line_end = pred_keypoints_2d_smpl[i][43].cpu().numpy() + v_head_norm*0.2*np.linalg.norm(v_head.cpu().numpy())
            cv2.line(cv_ndarr, (int(line_start[0]), int(line_start[1])), (int(line_end[0]), int(line_end[1])), cv_color, 2)

            # draw text
            for j, label in enumerate(labels[track_ids[i]][::-1]):
                 cv_ndarr = self.draw_text(cv_ndarr, label, [line_end[0]-50, line_end[1]-(j+1)*20], cv_color)

        return numpy_to_torch_image(cv_ndarr)/255.

    def tile_texture(self, image_padded, uv_maps, tracked_ids_x, img_height, img_width, top, left):
        rendered_image_tex = numpy_to_torch_image(np.array(image_padded)/255.)*0.0
        try:
            mask_valid       = np.load("_DATA/fmap_256.npy")
            mask_valid       = mask_valid>0
            mask_valid_      = np.logical_not(mask_valid)
            uv_maps          = uv_maps[ids_x, :, :, :]
            tracked_ids_x    = np.sort(tracked_ids_x)
            for i_, track_id in enumerate(tracked_ids_x):
                if(i_>7): continue
                uv_x0 = (img_height//2)*(i_//4) + top
                uv_x1 = (img_height//2)*(i_//4) + (img_height//2) + top
                uv_y0 = (img_width//4)*(i_%4) + left
                uv_y1 = (img_width//4)*(i_%4) + (img_width//4) + left
                uvmap_x = uv_maps[i_].unsqueeze(0)*5.0  
                uvmap_x = uvmap_x * torch.tensor([0.229, 0.224, 0.225], device="cuda").reshape(1,3,1,1) 
                uvmap_x = uvmap_x + torch.tensor([0.485, 0.456, 0.406], device="cuda").reshape(1,3,1,1)
                uvmap_x[:, :, mask_valid_] = 1
                uvmap_x_ = F.interpolate(uvmap_x, size=(img_height//2, img_width//4))
                rendered_image_tex[0, :, uv_x0:uv_x1, uv_y0:uv_y1] = uvmap_x_[0]
            rendered_image_tex = rendered_image_tex[:, :, top:top+img_height, left:left+img_width]
            rendered_image_tex = rendered_image_tex[:, [2,1,0], :, :] 
        except:
            pass
        return rendered_image_tex

    def blur_faces(self, image):
        
        rgb  = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        boxes, _ = self.face_detector.detect(rgb)

        # If no faces are detected then return original img
        if boxes is None:
            return image
        
        # Draw rectangle around the faces which is our region of interest (ROI)
        boxes    = np.array(boxes).astype(int)
        for (x, y, w, h) in boxes:
            x = max(0, x)
            y = max(0, y)
            w = min(image.shape[1], w)
            h = min(image.shape[0], h)
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = image[y:h, x:w]
            # applying a gaussian blur over this new rectangle area
            roi = cv2.GaussianBlur(roi, (23, 23), 30)

            # impose this blurred image on original image to get final image
            image[y:y+roi.shape[0], x:x+roi.shape[1]] = roi
        
        return image


    def render_video(self, final_visuals_dic):
        
        t_           = final_visuals_dic['time']
        shot_        = final_visuals_dic['shot']
        cv_image     = final_visuals_dic['frame']
        tracked_ids  = final_visuals_dic["tid"]
        tracked_time = final_visuals_dic["tracked_time"]
        
        tracked_mask = final_visuals_dic["mask"]
        tracked_bbox = final_visuals_dic["bbox"]
        
        tracked_smpl = final_visuals_dic["smpl"]
        tracked_cameras = final_visuals_dic["camera"]    

        NUM_PANELS    = 1
        PANEL_TEXTURE = False

        if "TEX_S" in self.cfg.render.type:
            tracked_appe = final_visuals_dic["uv"]
            NUM_PANELS   = 2
            PANEL_TEXTURE = True
        elif "TEX_P" in self.cfg.render.type:
            tracked_appe = final_visuals_dic["prediction_uv"]
            NUM_PANELS   = 2
            PANEL_TEXTURE = True
        else:
            tracked_appe = None
        
        # blur faces if needed
        if(self.cfg.render.blur_faces):
            cv_image = self.blur_faces(cv_image)

        img_height, img_width, _      = cv_image.shape
        new_image_size                = max(img_height, img_width)
        render_image_size             = self.cfg.render.res*self.cfg.render.up_scale
        ratio_                        = self.cfg.render.res*self.cfg.render.up_scale/new_image_size
        
        delta_w                       = new_image_size - img_width
        delta_h                       = new_image_size - img_height
        top, bottom, left, right      = delta_h//2, delta_h-(delta_h//2), delta_w//2, delta_w-(delta_w//2)
        top_, bottom_, left_, right_  = int(top*ratio_), int(bottom*ratio_), int(left*ratio_), int(right*ratio_)
        img_height_, img_width_       = int(img_height*ratio_), int(img_width*ratio_)

        image_padded                  = cv2.copyMakeBorder(cv_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        image_resized                 = cv2.resize(image_padded, (self.cfg.render.res*self.cfg.render.up_scale, self.cfg.render.res*self.cfg.render.up_scale))
        scale_                        = self.cfg.render.output_resolution/img_width
        frame_size                    = (self.cfg.render.output_resolution*NUM_PANELS, int(img_height*(scale_)))
        image_resized_rgb             = numpy_to_torch_image(np.array(image_resized)/255.)
        
        if(len(tracked_ids)>0):
            tracked_time              = np.array(tracked_time)
            tracked_smpl              = np.array(tracked_smpl)
            tracked_mask              = np.array(tracked_mask)
            tracked_bbox              = np.array(tracked_bbox)
            tracked_cameras           = np.array(tracked_cameras)
            tracked_cameras[:, 2]     = tracked_cameras[:, 2]/self.cfg.render.up_scale
        
            if "HUMAN" in self.cfg.render.type:
                ids_x = tracked_time==0
            elif "TRACKID" in self.cfg.render.type:
                ids_x = np.logical_and(tracked_time==0, np.array(tracked_ids)==int(self.cfg.render.type.split("TRACKID_")[1].split("_")[0]))
            elif "GHOST" in self.cfg.render.type:
                ids_x = tracked_time<=0
                ids_g = tracked_time[ids_x]==-1
            else:
                raise ValueError("Unknown render type")

            tracked_ids_x             = np.array(tracked_ids)
            tracked_ids_x             = tracked_ids_x[ids_x]        
            tracked_colors            = np.array(self.colors[list(tracked_ids_x)])/255.0

            # if "GHOST" in self.cfg.render.type:
            #     tracked_colors[ids_g] = tracked_colors[ids_g]*0.5

            if PANEL_TEXTURE:
                tracked_appe              = np.array(tracked_appe)
                tracked_appe              = torch.from_numpy(tracked_appe).float().cuda()
                uv_maps                   = tracked_appe[:, :3, :, :]

            if(len(tracked_ids_x)>0):
                
                if "MESH" in self.cfg.render.type:
                    rendered_image_final, valid_mask  = self.render_single_frame(
                                                                            tracked_smpl[ids_x],
                                                                            tracked_cameras[ids_x],
                                                                            tracked_colors, 
                                                                            img_size   = render_image_size, 
                                                                            image      = (0*image_resized)/255.0, 
                                                                            use_image  = True,
                                                                            )

                    rendered_image_final = numpy_to_torch_image(np.array(rendered_image_final))

                    valid_mask = np.repeat(valid_mask, 3, 2)
                    valid_mask = np.array(valid_mask, dtype=float)
                    valid_mask = numpy_to_torch_image(np.array(valid_mask))
                    
                    rendered_image_final = valid_mask*rendered_image_final + (1-valid_mask)*image_resized_rgb
                    rendered_image_final = rendered_image_final[:, :, top_:top_+img_height_, left_:left_+img_width_]

                if "MASK" in self.cfg.render.type or "BBOX" in self.cfg.render.type:
                    seg_mask = tracked_mask[ids_x]
                    seg_bbox = tracked_bbox[ids_x]
                    for i, tr in enumerate(tracked_ids_x):
                        seg_mask_ = mask_utils.decode(seg_mask[i][0])
                        seg_bbox_ = seg_bbox[i]
                        cv_image = self.visualize_mask(cv_image, seg_mask_, seg_bbox_, 
                                                       color=np.array(self.colors[tr]), 
                                                       text="track id : " + str(tr)
                                                       )
                    rendered_image_final = numpy_to_torch_image(np.array(cv_image)/255.)
            else:
                rendered_image_final = copy.deepcopy(image_resized_rgb)
                rendered_image_final = rendered_image_final[:, :, top_:top_+img_height_, left_:left_+img_width_]
        else:
            rendered_image_final = copy.deepcopy(image_resized_rgb)
            rendered_image_final = rendered_image_final[:, :, top_:top_+img_height_, left_:left_+img_width_]
        
        # render additional labels if any
        if("label" in final_visuals_dic.keys()):
            tracked_labels = final_visuals_dic["label"]
            rendered_image_final = self.visualize_labels_arrow(rendered_image_final, tracked_labels, tracked_bbox[ids_x]*ratio_, 
                                                         tracked_colors, tracked_ids_x, tracked_smpl[ids_x], tracked_cameras[ids_x])

        if PANEL_TEXTURE:
            rendered_image_tex = self.tile_texture(image_resized, uv_maps, tracked_ids_x, img_height_, img_width_, top, left)
            grid_img = make_grid(torch.cat([rendered_image_final, rendered_image_tex], 0), nrow=10)
        else:
            grid_img = make_grid(rendered_image_final, nrow=10)

        grid_img = grid_img[[2,1,0], :, :]
        ndarr    = grid_img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        cv_ndarr = cv2.resize(ndarr, frame_size)
        cv2.putText(cv_ndarr, str(t_), (20,40), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,255,255))
        if(shot_==1):
            cv2.putText(cv_ndarr, "SHOT", (20,80), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,255,255))
            cv2.rectangle(cv_ndarr, (0,0), (frame_size[0], frame_size[1]), (0,0,255), 5)
            
        return cv_ndarr, frame_size


