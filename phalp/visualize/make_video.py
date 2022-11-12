import copy

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from phalp.utils.utils import get_colors, numpy_to_torch_image

RGB_tuples = get_colors()

def render_frame_main_online(cfg, phalp_tracker, image_name, final_visuals_dic, dataset, number_of_windows=4, track_id=0):

    t_           = final_visuals_dic['time']
    cv_image     = final_visuals_dic['frame']
    tracked_ids  = final_visuals_dic["tid"]
    tracked_bbox = final_visuals_dic["bbox"]
    tracked_loca = final_visuals_dic["prediction_loca"]
    tracked_pose = [final_visuals_dic["prediction_pose"], final_visuals_dic["pose"]]
    tracked_appe = [final_visuals_dic["uv"], final_visuals_dic["prediction_uv"]]
    tracked_time = final_visuals_dic["tracked_time"]
    
    
    number_of_windows = 1
    res               = 1440

    img_height, img_width, _      = cv_image.shape
    new_image_size                = max(img_height, img_width)
 
    if(phalp_tracker.visualizer.render_size!=cfg.render.res*cfg.render.up_scale):
        phalp_tracker.visualizer.reset_render(cfg.render.res*cfg.render.up_scale)
    
    new_image_size_x              = cfg.render.res*cfg.render.up_scale
    ratio                         = 1.0*cfg.render.res/max(img_height, img_width)*cfg.render.up_scale
    
    delta_w                       = new_image_size - img_width
    delta_h                       = new_image_size - img_height
    top, bottom, left, right      = delta_h//2, delta_h-(delta_h//2), delta_w//2, delta_w-(delta_w//2)
    resized_image                 = cv2.copyMakeBorder(cv_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    resized_image_bbox            = copy.deepcopy(resized_image)
    resized_image_small           = cv2.resize(resized_image, (cfg.render.res*cfg.render.up_scale, cfg.render.res*cfg.render.up_scale))
    scale_                        = res/img_width
    frame_size                    = (number_of_windows*res, int(img_height*(scale_)))

    rendered_image_1              = numpy_to_torch_image(np.array(resized_image_bbox)/255.)
    rendered_image_1x             = numpy_to_torch_image(np.array(resized_image_bbox)/255.)
    
    if(len(tracked_ids)>0):
        tracked_time              = np.array(tracked_time)
        tracked_pose_single       = np.array(tracked_pose[1])
        tracked_pose              = np.array(tracked_pose[0])
        tracked_pose_single       = torch.from_numpy(tracked_pose_single).cuda()
        tracked_pose              = torch.from_numpy(tracked_pose).cuda()
        tracked_bbox              = np.array(tracked_bbox)
        tracked_center            = tracked_bbox[:, :2] + tracked_bbox[:, 2:]/2.0 + [left, top]
        tracked_scale             = np.max(tracked_bbox[:, 2:], axis=1)
        tracked_loca              = np.array(tracked_loca)
        
        tracked_loca_xy           = tracked_loca[:, :90]
        tracked_loca_xy           = np.reshape(tracked_loca_xy, (-1,45,2))
        tracked_loca_xy           = tracked_loca_xy[:, 44, :]*new_image_size

        tracked_loca              = tracked_loca[:, 90:93]
        tracked_loca[:, 2]       /= cfg.render.up_scale
        
        if "HUMAN" in cfg.render.type:
            ids_x                 = tracked_time==0
        elif "GHOST" in cfg.render.type:
            ids_x                 = tracked_time>-100

        tracked_ids_x             = np.array(tracked_ids)
        tracked_ids_x             = tracked_ids_x[ids_x]

        tracked_appe_single       = np.array(tracked_appe[0])
        tracked_appe              = np.array(tracked_appe[1])
        tracked_appe_single       = torch.from_numpy(tracked_appe_single).float().cuda()
        tracked_appe              = torch.from_numpy(tracked_appe).float().cuda()
        uv_maps                   = tracked_appe_single[:, :3, :, :]
        scale_x                   = tracked_scale
        scale_x                   = np.reshape(scale_x, (len(scale_x), 1))
        
        rendered_image_1x             = rendered_image_1x[:, :, top:top+img_height, left:left+img_width]

        if(phalp_tracker.visualizer.render_size!=cfg.render.res*cfg.render.up_scale):
            phalp_tracker.visualizer.reset_render(cfg.render.res*cfg.render.up_scale)
        resized_image_small             = cv2.resize(resized_image, (cfg.render.res*cfg.render.up_scale, cfg.render.res*cfg.render.up_scale))
        if(len(tracked_ids_x)>0):
            if("SMOOTH" in cfg.render.type):
                rendered_image_3, valid_mask, _, _, _  = phalp_tracker.visualizer.render_single_frame(tracked_pose[ids_x, :], 
                                                                                    np.array(RGB_tuples[list(tracked_ids_x)])/255.0, 
                                                                                    img_size   = new_image_size_x,
                                                                                    location   = torch.from_numpy(tracked_loca[ids_x, :]).cuda(),
                                                                                    image      = (0*resized_image_small)/255.0, 
                                                                                    render     = True, 
                                                                                    use_image  = True,
                                                                                    engine     = "PYR" )
                
            else:
                rendered_image_3, valid_mask, _, _, _  = phalp_tracker.visualizer.render_single_frame(tracked_pose_single[ids_x, :], 
                                                                                    np.array(RGB_tuples[list(tracked_ids_x)])/255.0, 
                                                                                    center     = ratio*tracked_center[ids_x, :], 
                                                                                    img_size   = new_image_size_x, 
                                                                                    scale      = ratio*scale_x[ids_x, :], 
                                                                                    image      = (0*resized_image_small)/255.0, 
                                                                                    render     = True, 
                                                                                    use_image  = True,
                                                                                    engine     = "PYR" )
            
            
            
            
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


