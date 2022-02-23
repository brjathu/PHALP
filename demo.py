from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import torchvision.models as models

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.projects import point_rend

import os
import json
import copy
import shutil
import heapq
import argparse
import pickle
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from yacs.config import CfgNode as CN
import random
import glob
import time
from pytube import YouTube


from utils.utils_dataset import process_image, process_mask
from utils.utils import FrameExtractor
from PHALP import PHALP_tracker
from test_tracker import test_tracker



def run_detection(image_path, detections_path, num_frames=-1, class_list=[0]):

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))   
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    # cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST   = 0.7
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    predictor         = DefaultPredictor(cfg)

    
    files             = glob.glob(os.path.join(image_path, '*.jpg'))
    files.sort()
    files             = files[:num_frames] if num_frames>0 else files
    for f in tqdm(files):
        im = cv2.imread(f)
        outputs = predictor(im)
        instances = outputs['instances']
        out_npz = os.path.join(detections_path, '%s.npz' % f.split('/')[-1][:-4])
        np.savez(out_npz, classes=instances.pred_classes.cpu().numpy(), scores=instances.scores.cpu().numpy(), boxes=instances.pred_boxes.tensor.cpu().numpy(), masks=instances.pred_masks.cpu().numpy())

    # extract masks
    npzs = glob.glob(os.path.join(detections_path, '*.npz'))
    npzs = [i for i in npzs if ("detections.npz" not in i)]
    npzs.sort()
    max_count = 0
    for npz_i in npzs:
        npz   = np.load(npz_i)
        count = 0
        for i in range(npz['classes'].shape[0]):
            if not(npz['classes'][i] in class_list):
                continue
            cv2.imwrite(os.path.join(detections_path, '%s_%02d.png' % (npz_i.split('/')[-1][:-4], count)), npz['masks'][i].astype(int)*255)
            count = count + 1
        max_count = max(count, max_count)

    imgnames_, masknames_, centers_, scales_, instances_, confs_, classes_ = [], [], [], [], [], [], []
    instances        = -np.ones([1, len(npzs), max_count]).astype(int)
    groups           = -np.ones([1, len(npzs)]).astype(int)
    counter_instance = 0

    for fi, npz in enumerate(npzs):
        npz_contents = np.load(npz)
        ui = 0
        for bbox, class_id, mask, score in zip(npz_contents['boxes'], npz_contents['classes'], npz_contents['masks'], npz_contents['scores']):
            if not(class_id in class_list): 
                continue
            imgname              = '%s.jpg' % npz.split('/')[-1][:-4] 
            maskname             = '%s_%02d.png' % (npz.split('/')[-1][:-4], ui) 
            center               = [(bbox[2] + bbox[0])/2, (bbox[3] + bbox[1])/2]
            scale                = [1.2*(bbox[2] - bbox[0]), 1.2*(bbox[3] - bbox[1])]
            instances[0, fi, ui] = counter_instance
            counter_instance     = counter_instance + 1
            ui                   = ui + 1

            imgnames_.append(imgname); masknames_.append(maskname); centers_.append(center); scales_.append(scale); confs_.append(score); classes_.append(class_id)

    video_npz = os.path.join(detections_path, 'detections.npz')
    np.savez(video_npz, imgname=imgnames_,  maskname=masknames_, center=centers_, scale=scales_, conf=confs_, class_id=classes_, instances=instances)
    

def run_hmar(video_path, detections_path, save_path):

    parser           = argparse.ArgumentParser(description='PHALP Tracker')
    parser.add_argument('--dataset', type=str, default='val')
    parser.add_argument('--batch_id', type=int, default='-1')

    opt              = parser.parse_args()
    opt.attributes   = "APL"
    opt.predict      = ""
    opt.mask_type    = "feat"
    opt.encode_type  = "3c"
    opt.res          = 256

    phalp_tracker      = PHALP_tracker(opt)
    phalp_tracker.cuda()
    phalp_tracker.eval()
    phalp_tracker.HMAR.reset_nmr(opt.res)

    video_data       = np.load(detections_path + "/detections.npz"); is_gt = False 
    video_seq        = video_data['instances']
    mask_path        = detections_path

    video_seq_       = []
    for video_id, video in enumerate(video_seq):

        track            = {}
        old_image_size   = 10
        frame_num        = 0
        
        for frame in tqdm(video):
            try:    shot = video_data['shots'][video_id][frame_num]; frame_num += 1
            except: shot = 0

            f_loc        = frame!=-1
            frame_ids    = frame[f_loc]
            if(len(frame_ids)==0): continue


            frame_ids_   = []
            frame_ids_gt = []
            for kl in frame_ids:
                frame_ids_.append(kl)
                gt_loc_  = np.where(frame==kl)[0]
                frame_ids_gt.append(gt_loc_[0])


            frame_name                    = video_data['imgname'][int(frame_ids[0])] 
            video_name                    = video_path.split("/")[-1]
            image                         = cv2.imread(video_path + "/" + frame_name)
            img_height, img_width, _      = image.shape
            new_image_size                = max(img_height, img_width)
            delta_w                       = new_image_size - img_width
            delta_h                       = new_image_size - img_height
            top, bottom                   = delta_h//2, delta_h-(delta_h//2)
            left, right                   = delta_w//2, delta_w-(delta_w//2)

            if(video_name in track.keys()): track[video_name][frame_name] = {}
            else:                           track[video_name] = {}; track[video_name][frame_name] = {}; video_seq_.append(video_name)

            for idx, det_person in enumerate(frame_ids):
                id_                 = int(det_person)
                id_gt               = int(frame_ids_gt[idx])
                center_             = video_data['center'][id_]
                scale_              = video_data['scale'][id_]
                conf_               = video_data['conf'][id_] if not(is_gt) else 1
                class_              = video_data['class_id'][id_]
                x1                  = center_[0] - scale_[0]/2.0
                y1                  = center_[1] - scale_[1]/2.0
                x2                  = center_[0] + scale_[0]/2.0
                y2                  = center_[1] + scale_[1]/2.0
                w                   = x2-x1; h = y2-y1
                center_             = np.array([(x1+x2)/2.0, (y1+y2)/2.0])
                scale_              = np.array([w, h])
                mask                = cv2.imread(detections_path + "/" + video_data['maskname'][id_])
                mask_tmp            = process_mask(mask, center_, 1.0*np.max(scale_))
                image_tmp           = process_image(image, center_, 1.0*np.max(scale_))
                masked_image        = torch.cat((image_tmp, mask_tmp[:1, :, :]), 0)
                ratio               = 1.0/int(new_image_size)*opt.res

                if(class_==0):
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
                        uv_image_re     = phalp_tracker.HMAR.autoencoder_hmar(appe_embedding.view(1, 16, 16, 16), en=False)
                        rendered_image, mask_, pred_joints_2d, pred_joints, pred_cam  = phalp_tracker.HMAR.render_3d(torch.cat((pose_embedding, pose_embedding), 1),
                                                                                                           np.array([[1.0, 0, 0]]),
                                                                                                           center=(center_ + [left, top])*ratio,
                                                                                                           img_size=opt.res,
                                                                                                           scale=np.reshape(np.array([max(scale_)]), (1, 1))*ratio,
                                                                                                           texture=uv_image_re[:, :3, :, :]*5.0, render=False)
                        pred_joints     = pred_joints[0]
                        pred_joints_2d  = pred_joints_2d[0]
                        pred_joints_2d  = pred_joints_2d.contiguous()
                        pred_cam_       = pred_cam.repeat(45, 1)
                        mask_image      = mask_[0]
                        loca_embedding  = torch.cat((pred_joints_2d.view(-1,)/opt.res, pred_cam.view(-1,)), 0)
                
                track[video_name][frame_name][idx+1]                   = {}
                track[video_name][frame_name][idx+1]['score']          = conf_
                track[video_name][frame_name][idx+1]['class']          = class_
                track[video_name][frame_name][idx+1]['gt']             = id_gt
                track[video_name][frame_name][idx+1]['bbox']           = np.array([x1, y1, w, h])
                track[video_name][frame_name][idx+1]['center']         = center_
                track[video_name][frame_name][idx+1]['scale']          = scale_
                track[video_name][frame_name][idx+1]['image_size']     = np.array([img_height, img_width])
                track[video_name][frame_name][idx+1]['image_name']     = frame_name
                track[video_name][frame_name][idx+1]['mask_name']      = video_data['maskname'][id_]
                track[video_name][frame_name][idx+1]['shot']           = shot

                track[video_name][frame_name][idx+1]['keypoints_3d']   = pred_joints.cpu().numpy()
                track[video_name][frame_name][idx+1]['keypoints_3t']   = pred_cam_.cpu().numpy()
                track[video_name][frame_name][idx+1]['keypoints_2d']   = pred_joints_2d.cpu().numpy()

                track[video_name][frame_name][idx+1]['pred_cam']       = pred_cam.cpu().numpy()
                track[video_name][frame_name][idx+1]['appe_embedding'] = appe_embedding[0].cpu().numpy()
                track[video_name][frame_name][idx+1]['pose_embedding'] = pose_embedding[0].cpu().numpy()
                track[video_name][frame_name][idx+1]['loca_embedding'] = loca_embedding.cpu().numpy()
                track[video_name][frame_name][idx+1]['uv_vector']      = uv_vector[0].cpu().numpy()

                
                
                
        list_of_frames_ = os.listdir(video_path)
        list_of_frames  = [f for f in list_of_frames_ if ".jpg" in f]

        for frame_ in list_of_frames:
            if(frame_ not in track[video_name].keys()):
                track[video_name][frame_]             = {}
                track[video_name][frame_][1]          = {}
                track[video_name][frame_][1]['score'] = -1


        if(len(track.keys())>0):
            with open(save_path + '/v1_' + video_name + '.pickle', 'wb') as handle:
                pickle.dump(track, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print("not saving!")



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
    
    # ########## PoseTrack
    if(opt.track_dataset=="posetrack"):
        track_dataset    = "posetrack"
        videos           = np.load("_DATA/posetrack/list_videos_val.npy")
        base_path_frames = "_DATA/posetrack/posetrack_data/images/val/"
    
    # ########## MuPoTs
    if(opt.track_dataset=="mupots"):
        track_dataset    = "mupots"
        videos           = np.load("_DATA/mupots/list_videos_val.npy")
        base_path_frames = "_DATA/mupots/mupots_data/images/val/"
    
    ########## MOT17
    if(opt.track_dataset=="mupots"):
        track_dataset    = "mot17"
        videos           = np.load("_DATA/mot17/list_videos_test.npy")
        base_path_frames = "_DATA/mot17/mot17_data/images/test/"

    os.system("mkdir "  + "_DATA/detections/" + track_dataset)
        
        
        
    for vid, video in enumerate(videos):        
        if(track_dataset=="demo"):
            os.system("rm -rf " + base_path_frames+video)
            os.system("mkdir " + base_path_frames+video)
            print('https://www.youtube.com/watch?v=' + links[vid])
            youtube_video = YouTube('https://www.youtube.com/watch?v=' + links[vid])
            print(f'Title: {youtube_video.title}')
            print(f'Duration: {youtube_video.length / 60:.2f} minutes')
            # print(youtube_video.streams.all())
            youtube_video.streams.get_by_itag(136).download(output_path = base_path_frames + video, filename="youtube.mp4")
            fe = FrameExtractor(base_path_frames + video + "/youtube.mp4")
            print(fe.n_frames)
            print(fe.get_video_duration())
            fe.extract_frames(every_x_frame=1, img_name='', dest_path=base_path_frames + video + "/", frames=[300, 400])

        os.system("rm -rf " + "_DATA/detections/" + track_dataset + "/" + video)
        os.system("mkdir "  + "_DATA/detections/" + track_dataset + "/" + video)
        frames_path            = base_path_frames + video
        detections_path        = "_DATA/detections/" + track_dataset + "/" + video + "/"
        save_path              = "_DATA/embeddings/"
        run_detection(frames_path, detections_path, num_frames=-1, class_list=[0])
        run_hmar(frames_path, detections_path, save_path)
        

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
        opt.video_seq          = video
        
        opt.render_type        = "HUMAN_HEAD"
        opt.render             = True
        opt.res                = 256
        opt.render_up_scale    = 2
        opt.downsample         = 1
        
        opt.verbose            = False
        opt.use_gt             = False
        opt.encode_type        = "3c"
        opt.past_lookback      = 1
        opt.cva_type           = "least_square"
    
        phalp_tracker          = PHALP_tracker(opt)
        phalp_tracker.cuda()
        phalp_tracker.eval()
        phalp_tracker.HMAR.reset_nmr(256)

        test_tracker(opt, phalp_tracker)
