import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

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

from tqdm import tqdm
from yacs.config import CfgNode as CN
from utils.make_video import render_frame_main
from utils.utils import str2bool, get_colors
 
from deep_sort_ import nn_matching
from deep_sort_.detection import Detection
from deep_sort_.tracker import Tracker

from PHALP import PHALP_tracker
from evaluate_PHALP import evaluate_trackers

warnings.filterwarnings("ignore")

RGB_tuples = get_colors()
    
def test_tracker(opt, phalp_tracker, checkpoint=None):
    phalp_tracker.eval()
    video_seq    = np.load(opt.video_seq) if("npy" in opt.video_seq) else [opt.video_seq]
    
    os.makedirs("out/", exist_ok=True)
    os.makedirs("out/"+opt.storage_folder, exist_ok=True)
    os.makedirs("out/"+opt.storage_folder+"/results", exist_ok=True)      

    for video_file_name in tqdm(video_seq):

        try: 
            if(opt.use_gt):  track = joblib.load('_DATA/embeddings/v1_gt_' + str(video_file_name) + '.pickle') 
            else:            track = joblib.load('_DATA/embeddings/v1_'    + str(video_file_name) + '.pickle')
     
            final_results_dic    = {}
            final_visuals_dic    = {}
            sequence             = track.keys()
            for video in sequence:
                metric           = nn_matching.NearestNeighborDistanceMetric(opt, opt.hungarian_th, opt.past_lookback)
                tracker          = Tracker(opt, metric, max_age=opt.max_age_track, n_init=opt.n_init, phalp_tracker=phalp_tracker, dims=[4096, 4096, 99])  

                frame_list       = sorted(list(track[video].keys()))
                frame_list       = frame_list[:]
                frame_length     = len(frame_list)
                max_ids          = opt.max_ids
                person_id        = torch.zeros(frame_length, max_ids) + -1
                center           = torch.zeros(frame_length, max_ids, 2)
                scale            = torch.zeros(frame_length, max_ids, 2) 
                bbox             = torch.zeros(frame_length, max_ids, 4) 
                conf_c           = torch.zeros(frame_length, max_ids, 1) 
                pose_emb         = torch.zeros(frame_length, max_ids, 4096) 
                appe_emb         = torch.zeros(frame_length, max_ids, 4096) 
                loca_emb         = torch.zeros(frame_length, max_ids, 99) + 1
                uv_vector        = torch.zeros(frame_length, max_ids, 4, 256, 256) 
                frame_size       = torch.zeros(frame_length, max_ids, 2) 
                image_name       = np.zeros((frame_length, max_ids, 1), dtype=object) 
                mask_name        = np.zeros((frame_length, max_ids, 1), dtype=object) 
                ground_truth     = np.zeros((frame_length, max_ids, 1)) 
                shots            = np.zeros((frame_length,)) 

                for frame_idx, frame in enumerate(frame_list): 
                    idx          = 0
                    for idx_ in range(len(track[video][frame])):
                        person_                                 = track[video][frame][idx_+1]
                        if(person_['score']>opt.low_th_c ):
                            person_id[frame_idx, idx]           = 0 
                            pose_emb[frame_idx, idx, 2048:]     = torch.from_numpy(person_['pose_embedding'])
                            appe_emb[frame_idx, idx, :]         = torch.from_numpy(person_['appe_embedding'])
                            loca_emb[frame_idx, idx, :93]       = torch.from_numpy(person_['loca_embedding'])
                            conf_c[frame_idx, idx, :]           = torch.from_numpy(np.array(person_['score']))
                            bbox[frame_idx, idx, :]             = torch.from_numpy(np.array([person_['bbox'][0], person_['bbox'][1], person_['bbox'][0]+person_['bbox'][2], person_['bbox'][1]+person_['bbox'][3]]))
                            center[frame_idx, idx, :]           = torch.from_numpy(np.array(person_['center']))
                            scale[frame_idx, idx, :]            = torch.from_numpy(np.array(person_['scale']))
                            uv_vector[frame_idx, idx, :, :, :]  = torch.from_numpy(person_['uv_vector'])
                            frame_size[frame_idx, idx, :]       = torch.from_numpy(np.array(person_['image_size']))
                            mask_name[frame_idx, idx, :]        = np.array([person_['mask_name']])
                            image_name[frame_idx, idx, :]       = np.array([person_['image_name']])
                            ground_truth[frame_idx, idx, :]     = np.array([person_['gt']])
                            try: shots[frame_idx]               = person_['shot']
                            except: shots[frame_idx]            = 0
                            idx += 1
                            if(idx>=opt.max_ids): break

                BS, T, P   = 1, frame_length, max_ids
                window     = frame_length//opt.window            
                embeddings = torch.cat((appe_emb, pose_emb, loca_emb), 2) 

                opt.shot = 0
                tracked_frames = []
                for w_ in range(frame_length//window):
                    for t in range(window):

                        if(opt.verbose):       print("t : ", t, video_file_name, opt.storage_folder)
                        t_                     = t+w_*window
                        loc_                   = np.where(person_id[w_*window:(w_+1)*window][t]!=-1)[0]
                        detections             = []; tracked_appe_single    = []; tracked_time = []; tracked_mask_ = [];
                        tracked_ids            = []; tracked_bbox           = []; tracked_ids_ = [];
                        tracked_appe           = []; tracked_pose           = []; tracked_bbox_= [];
                        tracked_loca           = []; tracked_pose_single    = []; tracked_feat_= [];
                        for m in range(len(loc_)):
                            bbox_candidate     = bbox[w_*window:(w_+1)*window][t][loc_][m]
                            w                  = bbox_candidate[2] - bbox_candidate[0]
                            h                  = bbox_candidate[3] - bbox_candidate[1]
                            opt.shot = shots[t_-1] if(t_>0) else 0
                            if(opt.track_dataset=="mupots"): th_h = 200
                            else:                            th_h = 100
                            if(h>th_h and w>50 ):
                                detection_data   = {
                                                      "bbox"            : [bbox_candidate[0], bbox_candidate[1], w, h],
                                                      "conf_c"          : conf_c[w_*window:(w_+1)*window][t][loc_][m], 
                                                      "embedding"       : embeddings[w_*window:(w_+1)*window][t][loc_][m], 
                                                      "uv_vector"       : uv_vector[w_*window:(w_+1)*window][t][loc_][m], 
                                                      "center"          : center[w_*window:(w_+1)*window][t][loc_][m],
                                                      "scale"           : scale[w_*window:(w_+1)*window][t][loc_][m],
                                                      "size"            : frame_size[w_*window:(w_+1)*window][t][loc_][m],
                                                      "img_name"        : image_name[w_*window:(w_+1)*window][t][loc_][m],
                                                      "mask_name"       : mask_name[w_*window:(w_+1)*window][t][loc_][m],
                                                      "ground_truth"    : ground_truth[w_*window:(w_+1)*window][t][loc_][m],
                                                      "time"            : t_,
                                                   }
                                detections.append(Detection(detection_data))

                        tracker.predict()
                        matches = tracker.update(detections, t_, frame_list[t_], opt.shot)



                        for tracks_ in tracker.tracks:
                            if(frame_list[t_] not in tracked_frames): tracked_frames.append(frame_list[t_])
                            if(not(tracks_.is_confirmed())): continue
                            track_id       = tracks_.track_id
                            bbox_          = tracks_.phalp_bbox[-1]   
                            tracked_ids.append(track_id)
                            tracked_bbox.append([bbox_[0], bbox_[1], bbox_[2], bbox_[3]])

                            if(opt.render): 
                                tracked_appe_single.append(tracks_.phalp_uv_map); tracked_appe.append(tracks_.phalp_uv_predicted); # phalp_uv_predicted
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



                        if(frame_list[t_] in final_results_dic.keys() or frame_list[t_] in final_visuals_dic.keys()): print("Error!") 
                        else: 
                            final_results_dic.setdefault(frame_list[t_], [tracked_ids_, tracked_bbox_, t_]) 
                            if(opt.render): final_visuals_dic.setdefault(frame_list[t_], [tracked_ids, tracked_bbox, t_, tracked_loca, [tracked_pose, tracked_pose_single], [tracked_appe, tracked_appe_single], tracked_time]) 


                frame_tracked = final_results_dic.keys()
                for t_, frame_ in enumerate(frame_list):
                    if(frame_ not in frame_tracked): final_results_dic.setdefault(frame_, [[], [[]], t_]) 

                joblib.dump(final_results_dic, "out/" + opt.storage_folder + "/results/" + video_file_name + ".pkl")
                joblib.dump(tracker.tracked_cost, "out/" + opt.storage_folder + "/results/" + video_file_name + "_distance" + ".pkl")

                if(opt.render):
                    t_ = 0
                    for frame_key in tqdm(final_visuals_dic.keys()):
                        rendered_, f_size  = render_frame_main(opt, phalp_tracker, final_visuals_dic[frame_key][2], opt.track_dataset, [opt.base_path, opt.mask_path], video, 
                                                               frame_key, final_visuals_dic[frame_key][0], final_visuals_dic[frame_key][1], final_visuals_dic[frame_key][3], 
                                                               final_visuals_dic[frame_key][4], final_visuals_dic[frame_key][5], final_visuals_dic[frame_key][6],  
                                                               number_of_windows=4, downsample=opt.downsample, storage_folder="out/" + opt.storage_folder + "/_TEMP/", track_id=-100)      
                        if(t_==0):
                            fourcc         = cv2.VideoWriter_fourcc(*'mp4v')
                            video_file     = cv2.VideoWriter("out/" + opt.storage_folder + "/" + video + ".mp4", fourcc, 15, frameSize=f_size)
                        video_file.write(rendered_)
                        t_ += 1
                    video_file.release()

        except Exception as e: 
            print(e)
            print(traceback.format_exc())     
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PHALP_pixel Tracker')
    parser.add_argument('--batch_id', type=int, default='-1')
    parser.add_argument('--track_dataset', type=str, default='posetrack')
    parser.add_argument('--predict', type=str, default='APL')
    parser.add_argument('--storage_folder', type=str, default='Videos_v20.000')
    parser.add_argument('--distance_type', type=str, default='A5')
    parser.add_argument('--use_gt', type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--low_th_c', type=float, default=0.9)
    parser.add_argument('--hungarian_th', type=float, default=100.0)
    parser.add_argument('--track_history', type=int, default=7)
    parser.add_argument('--max_age_track', type=int, default=20)
    parser.add_argument('--n_init',  type=int, default=5)
    parser.add_argument('--max_ids', type=int, default=50)
    parser.add_argument('--window',  type=int, default=1)
    parser.add_argument('--verbose', type=str2bool, nargs='?', const=True, default=False)
    
    parser.add_argument('--base_path', type=str)
    parser.add_argument('--mask_path', type=str)
    parser.add_argument('--video_seq', type=str, default='_DATA/posetrack/list_videos_val.npy')
    parser.add_argument('--version', type=str, default='v1')

    parser.add_argument('--render', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--render_type', type=str, default='HUMAN_HEAD_FAST')
    parser.add_argument('--render_up_scale', type=int, default=2)
    parser.add_argument('--res', type=int, default=256)
    parser.add_argument('--downsample',  type=int, default=1)
    
    parser.add_argument('--encode_type', type=str, default='3c')
    parser.add_argument('--cva_type', type=str, default='least_square')
    parser.add_argument('--past_lookback', type=int, default=1)

    
    
    opt                   = parser.parse_args()
    phalp_tracker         = PHALP_tracker(opt)
    phalp_tracker.cuda()
    phalp_tracker.eval()
    phalp_tracker.HMAR.reset_nmr(512)
    
    test_tracker(opt, phalp_tracker)

