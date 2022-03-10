import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.projects import point_rend

import os
import traceback
import warnings
import json
import copy
import argparse
import joblib
import cv2
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
from yacs.config import CfgNode as CN
from pytube import YouTube

from utils.make_video import render_frame_main, render_frame_main_online
from utils.utils import str2bool, get_colors
from utils.utils import FrameExtractor

from deep_sort_ import nn_matching
from deep_sort_.detection import Detection
from deep_sort_.tracker import Tracker

from PHALP import PHALP_tracker
from utils.utils_dataset import process_image, process_mask

warnings.filterwarnings("ignore")

RGB_tuples = get_colors()
    
def test_tracker_online(opt, phalp_tracker, checkpoint=None):

    try:
        os.system("mkdir out/" + opt.storage_folder)
        os.system("mkdir out/" + opt.storage_folder + "/results")        
    except: pass


    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))   
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST   = 0.4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    predictor         = DefaultPredictor(cfg)

    phalp_tracker.eval()
    phalp_tracker.HMAR.reset_nmr(opt.res)    

    metric           = nn_matching.NearestNeighborDistanceMetric(opt, opt.hungarian_th, opt.past_lookback)
    tracker          = Tracker(opt, metric, max_age=opt.max_age_track, n_init=opt.n_init, phalp_tracker=phalp_tracker, dims=[4096, 4096, 99])  

    try: 

        list_of_frames    = np.sort([i for i in os.listdir(opt.base_path + "/" + opt.video_seq[0]) if ".jpg" in i])
        tracked_frames    = []
        final_results_dic = {}
        final_visuals_dic = {}        
        video_created     = 0
        for t_, frame_name in tqdm(enumerate(list_of_frames)):
            
            ##### detection part
            im                            = cv2.imread(opt.base_path + "/" + opt.video_seq[0] + "/" + frame_name)
            img_height, img_width, _      = im.shape
            new_image_size                = max(img_height, img_width)
            delta_w                       = new_image_size - img_width
            delta_h                       = new_image_size - img_height
            top, bottom                   = delta_h//2, delta_h-(delta_h//2)
            left, right                   = delta_w//2, delta_w-(delta_w//2)
            
            outputs                       = predictor(im)
            instances                     = outputs['instances']
            v                             = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            v                             = v.draw_instance_predictions(instances.to("cpu"))
            
            centers_, scales_, bboxes_, masks_, confs_, classes_ = [], [], [], [], [], []
            pred_bbox                     = instances.pred_boxes.tensor.cpu().numpy()
            pred_classes                  = instances.pred_classes.cpu().numpy()
            pred_masks                    = instances.pred_masks.cpu().numpy()
            pred_scores                   = instances.scores.cpu().numpy()
            
            idx_                          = pred_classes==0
            full_embedding                = []
            uv_vector_list                = []
            detections                    = []
            
            h_th = 100; w_th = 50
            for bbox, class_id, mask, score in zip(pred_bbox[idx_], pred_classes[idx_], pred_masks[idx_], pred_scores[idx_]):
                if score < opt.low_th_c or bbox[2]-bbox[0]<w_th or bbox[3]-bbox[1]<h_th: continue

                mask_a              = mask.astype(int)*255
                mask_a              = np.expand_dims(mask_a, 2)
                mask_a              = np.repeat(mask_a, 3, 2)
                
                ##### HMAR part
                center_             = np.array([(bbox[2] + bbox[0])/2, (bbox[3] + bbox[1])/2])
                scale_              = np.array([(bbox[2] - bbox[0]), (bbox[3] - bbox[1])])
                mask_tmp            = process_mask(mask_a.astype(np.uint8), center_, 1.0*np.max(scale_))
                image_tmp           = process_image(im, center_, 1.0*np.max(scale_))
                masked_image        = torch.cat((image_tmp, mask_tmp[:1, :, :]), 0)
                ratio               = 1.0/int(new_image_size)*opt.res
                
                with torch.no_grad():
                    
                    hmar_out        = phalp_tracker.HMAR(masked_image.unsqueeze(0).cuda()) 
                    uv_image        = hmar_out['uv_image'][:, :3, :, :]/5.0
                    uv_mask         = hmar_out['uv_image'][:, 3:, :, :]
                    zeros_          = uv_mask==0
                    ones_           = torch.logical_not(zeros_)
                    zeros_          = zeros_.repeat(1, 3, 1, 1)
                    ones_           = ones_.repeat(1, 3, 1, 1)
                    uv_image[zeros_]= 0.0
                    uv_mask[zeros_[:, :1, :, :]] = -1.0
                    uv_mask[ones_[:, :1, :, :]]  = 1.0
                    uv_vector       = torch.cat((uv_image, uv_mask), 1)
                    pose_embedding  = hmar_out['pose_emb']
                    appe_embedding  = phalp_tracker.HMAR.autoencoder_hmar(uv_vector, en=True)
                    appe_embedding  = appe_embedding.view(1, -1)
                    rendered_image, mask_, pred_joints_2d, pred_joints, pred_cam  = phalp_tracker.HMAR.render_3d(torch.cat((pose_embedding, pose_embedding), 1),
                                                                                                       np.array([[1.0, 0, 0]]),
                                                                                                       center=(center_ + [left, top])*ratio,
                                                                                                       img_size=opt.res,
                                                                                                       scale=np.reshape(np.array([max(scale_)]), (1, 1))*ratio,
                                                                                                       texture=uv_vector[:, :3, :, :]*5.0, render=False)

                a = pred_joints_2d.reshape(-1,)/256
                a.contiguous()
                b = pred_cam.view(-1,)
                b.contiguous()
                loca_embedding  = torch.cat((a, b, b, b), 0)
                
                appe_emb_ = appe_embedding[0].cpu()
                pose_emb_ = pose_embedding[0].cpu()
                loca_emb_ = loca_embedding.cpu()

                full_embedding = torch.cat((appe_emb_, pose_emb_, pose_emb_, loca_emb_), 0)
                
                detection_data   = {
                                      "bbox"            : [bbox[0], bbox[1], (bbox[2] - bbox[0]), (bbox[3] - bbox[1])],
                                      "conf_c"          : score, 
                                      "embedding"       : full_embedding, 
                                      "uv_vector"       : uv_vector[0].cpu(), 
                                      "center"          : center_,
                                      "scale"           : scale_,
                                      "size"            : [img_height, img_width],
                                      "img_name"        : frame_name,
                                      "mask_name"       : frame_name,
                                      "ground_truth"    : 1,
                                      "time"            : t_,
                                   }
                a = Detection(detection_data)
                detections.append(a)

            tracker.predict()
            matches = tracker.update(detections, t_, frame_name, 0)
            

            tracked_ids = [];         tracked_bbox = [];     tracked_appe_single = [];     tracked_appe = []
            tracked_pose_single = []; tracked_pose = [];     tracked_loca = [];            tracked_time = []
            tracked_ids_ = [];        tracked_bbox_ = [];    tracked_mask_ = []
            for tracks_ in tracker.tracks:
                if(frame_name not in tracked_frames): tracked_frames.append(frame_name)
                if(not(tracks_.is_confirmed())): continue
                track_id       = tracks_.track_id
                bbox_          = tracks_.phalp_bbox[-1]   
                tracked_ids.append(track_id)
                tracked_bbox.append([bbox_[0], bbox_[1], bbox_[2], bbox_[3]])

                if(opt.render): 
                    tracked_appe_single.append(tracks_.phalp_uv_map); tracked_appe.append(tracks_.phalp_uv_predicted); 
                    tracked_pose_single.append(tracks_.phalp_pose_features[-1]); tracked_pose.append(tracks_.phalp_pose_predicted); 
                    tracked_loca.append(tracks_.phalp_loca_predicted); tracked_time.append(tracks_.time_since_update); 

                if(tracks_.time_since_update==0):
                    tracked_ids_.append(track_id)
                    tracked_bbox_.append([bbox_[0], bbox_[1], bbox_[2], bbox_[3]])
                    tracked_mask_.append(tracks_.detection_data[-1]['mask_name'])
                    if(tracks_.hits==opt.n_init):
                        for ia in range(opt.n_init-1):
                            try:
                                final_results_dic[tracked_frames[-2-ia]][0].append(track_id)
                                final_results_dic[tracked_frames[-2-ia]][1].append(tracks_.phalp_bbox[-2-ia])

                                if(opt.render): 
                                    final_visuals_dic[tracked_frames[-2-ia]][0].append(track_id)
                                    final_visuals_dic[tracked_frames[-2-ia]][1].append(tracks_.phalp_bbox[-2-ia])
                                    final_visuals_dic[tracked_frames[-2-ia]][3].append(tracks_.phalp_loca_predicted_[-2-ia])
                                    final_visuals_dic[tracked_frames[-2-ia]][4][0].append(tracks_.phalp_pose_predicted_[-2-ia])
                                    final_visuals_dic[tracked_frames[-2-ia]][4][1].append(tracks_.phalp_pose_features[-2-ia])
                                    final_visuals_dic[tracked_frames[-2-ia]][5][0].append(tracks_.phalp_uv_map_[-2-ia])
                                    final_visuals_dic[tracked_frames[-2-ia]][5][1].append(tracks_.phalp_uv_predicted_[-2-ia])
                                    final_visuals_dic[tracked_frames[-2-ia]][6].append(0)
                            except:
                                final_results_dic.setdefault(tracked_frames[-2-ia], [[track_id], tracks_.phalp_bbox[-2-ia], t_-ia-1]) 
                                if(opt.render): 
                                    final_visuals_dic.setdefault(tracked_frames[-2-ia], 
                                                                 [[track_id], tracks_.phalp_bbox[-2-ia], t_-ia-1, tracks_.phalp_loca_predicted_[-2-ia], [tracks_.phalp_pose_predicted_[-2-ia], 
                                                                 tracks_.phalp_pose_features[-2-ia]], [tracks_.phalp_uv_map_[-2-ia], tracks_.phalp_uv_predicted_[-2-ia]], [0]]) 



            if(frame_name in final_results_dic.keys() or frame_name in final_visuals_dic.keys()): print("Error!") # error check
            else: 
                final_results_dic.setdefault(frame_name, [tracked_ids_, tracked_bbox_, t_]) 
                if(opt.render): final_visuals_dic.setdefault(frame_name, [tracked_ids, tracked_bbox, t_, tracked_loca, [tracked_pose, tracked_pose_single], [tracked_appe, tracked_appe_single], tracked_time]) 

            if(opt.render):
                frame_key = frame_name
                rendered_, f_size  = render_frame_main_online(opt, phalp_tracker, final_visuals_dic[frame_key][2], opt.track_dataset, ["", ""], im, 
                                                       frame_key, final_visuals_dic[frame_key][0], final_visuals_dic[frame_key][1], final_visuals_dic[frame_key][3], 
                                                       final_visuals_dic[frame_key][4], final_visuals_dic[frame_key][5], final_visuals_dic[frame_key][6],  
                                                       number_of_windows=4, downsample=opt.downsample, storage_folder="out/" + opt.storage_folder + "/_TEMP/", track_id=-100)      
                if(video_created==0):
                    video_created  = 1
                    fourcc         = cv2.VideoWriter_fourcc(*'mp4v')
                    video_file     = cv2.VideoWriter("out/" + opt.storage_folder + "/PHALP_" + opt.track_dataset + "_" + "online" + ".mp4", fourcc, 15, frameSize=f_size)
                video_file.write(rendered_)
                
        if(opt.render):video_file.release()

    except Exception as e: 
        print(e)
        print(traceback.format_exc())     
    

if __name__ == '__main__':
    
    parser_demo = argparse.ArgumentParser(description='Demo')
    parser_demo.add_argument('--track_dataset', type=str, default='demo')
    opt         = parser_demo.parse_args()

    os.system("mkdir "  + "_DATA/detections/" )
    os.system("mkdir "  + "_DATA/embeddings/")
    os.system("mkdir "  + "_DATA/out/")
    
    # ########## Youtube Demo videos
    if(opt.track_dataset=="demo"):
        track_dataset    = "demo"
        links            = ['xEH_5T9jMVU'] 
        videos           = ["youtube_"+str(i) for i,j in enumerate(links)]
        base_path_frames = "_DATA/DEMO/frames/youtube/"

    os.system("mkdir "  + "_DATA/detections/" + track_dataset)
        
        
        
    for vid, video in enumerate(videos):        
        if(track_dataset=="demo"):
            os.system("rm -rf " + base_path_frames + video)
            os.system("mkdir "  + base_path_frames + video)
            print('https://www.youtube.com/watch?v=' + links[vid])
            youtube_video = YouTube('https://www.youtube.com/watch?v=' + links[vid])
            print(f'Title: {youtube_video.title}')
            print(f'Duration: {youtube_video.length / 60:.2f} minutes')
            # print(youtube_video.streams.all())
            youtube_video.streams.get_by_itag(136).download(output_path = base_path_frames + video, filename="youtube.mp4")
            fe = FrameExtractor(base_path_frames + video + "/youtube.mp4")
            print(fe.n_frames)
            print(fe.get_video_duration())
            fe.extract_frames(every_x_frame=1, img_name='', dest_path=base_path_frames + video + "/", frames=[0, 130000])

        os.system("rm -rf " + "_DATA/detections/" + track_dataset + "/" + video)
        os.system("mkdir "  + "_DATA/detections/" + track_dataset + "/" + video)
        frames_path            = base_path_frames + video
        detections_path        = "_DATA/detections/" + track_dataset + "/" + video + "/"

        opt.version            = "v1"  
        opt.track_dataset      = track_dataset
        opt.predict            = "TLP"
        opt.base_path          = base_path_frames 
        opt.mask_path          = detections_path
        opt.storage_folder     = "Videos_results"
        opt.distance_type      = "EQ_A"
        
        opt.track_history      = 7
        opt.low_th_c           = 0.9
        opt.alpha              = 0.1
        opt.hungarian_th       = 100
        opt.max_age_track      = 24
        opt.n_init             = 5
        opt.max_ids            = 50
        opt.window             = 1
        opt.batch_id           = -1
        opt.video_seq          = [video]
        
        opt.render_type        = "HUMAN_HEAD_FAST"
        opt.render             = True
        opt.res                = 256
        opt.render_up_scale    = 2
        opt.downsample         = 1
        
        opt.verbose            = False
        opt.use_gt             = False
        opt.encode_type        = "3c"
        opt.past_lookback      = 1
        opt.cva_type           = "least_square"
        opt.mask_type          = "feat"
        
        phalp_tracker          = PHALP_tracker(opt)
        phalp_tracker.cuda()
        phalp_tracker.eval()
        phalp_tracker.HMAR.reset_nmr(256)

        test_tracker_online(opt, phalp_tracker)

        
        
        