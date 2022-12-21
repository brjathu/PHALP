import glob
import os
import time
import traceback
import warnings

warnings.filterwarnings('ignore')

import cv2
import gdown
import joblib
import numpy as np
import torch
import torch.nn as nn
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from phalp.external.deep_sort_ import nn_matching
from phalp.external.deep_sort_.detection import Detection
from phalp.external.deep_sort_.tracker import Tracker
from phalp.models.hmar import HMAR
from phalp.utils import get_pylogger
from phalp.utils.utils import (FrameExtractor, convert_pkl,
                               get_prediction_interval)
from phalp.utils.utils_dataset import process_image, process_mask
from phalp.utils.utils_detectron2 import DefaultPredictor_Lazy
from phalp.utils.utils_scenedetect import detect
from phalp.visualize.visualizer import Visualizer
from pycocotools import mask as mask_utils
from pytube import YouTube
from rich.progress import track
from scenedetect import AdaptiveDetector
from sklearn.linear_model import Ridge
from tqdm import tqdm

log = get_pylogger(__name__)

class PHALP(nn.Module):

    def __init__(self, cfg):
        super(PHALP, self).__init__()

        self.cfg = cfg
        self.device = torch.device(self.cfg.device)
        
        # download wights and configs from Google Drive
        self.cached_download_from_drive()
        
        # setup HMR, override this function to use your own model
        self.setup_hmr()
        
        # setup Detectron2, override this function to use your own model
        self.setup_detectron2()
        
        # create a visualizer
        if(self.cfg.render.enable):
            self.setup_visualizer()
        
        # move to device
        self.to(self.device)
        
        # train or eval
        self.train() if(self.cfg.train) else self.eval()
        
    def setup_hmr(self):
        log.info("Loading HMR model...")
        self.HMAR       = HMAR(self.cfg)
        checkpoint_file = torch.load('_DATA/hmar_v2_weights.pth')
        state_dict_filt = {}
        for k, v in checkpoint_file['model'].items():
            if ("encoding_head" in k or "texture_head" in k or "backbone" in k or "smplx_head" in k): 
                state_dict_filt.setdefault(k[5:].replace("smplx", "smpl"), v)
        self.HMAR.load_state_dict(state_dict_filt, strict=False) 

    def setup_detectron2(self):
        log.info("Loading Detection model...")
        self.detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        self.detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        self.detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        self.detector       = DefaultPredictor_Lazy(self.detectron2_cfg)
        self.class_names    = self.detector.metadata.get('thing_classes')
        
    def setup_deepsort(self):
        log.info("Setting up DeepSort...")
        metric  = nn_matching.NearestNeighborDistanceMetric(self.cfg, self.cfg.phalp.hungarian_th, self.cfg.phalp.past_lookback)
        self.tracker = Tracker(self.cfg, metric, max_age=self.cfg.phalp.max_age_track, n_init=self.cfg.phalp.n_init, phalp_tracker=self, dims=[4096, 4096, 99])  
        
    def setup_visualizer(self):
        log.info("Setting up Visualizer...")
        self.visualizer = Visualizer(self.cfg, self.HMAR)
        
    def track(self):
        
        eval_keys       = ['tracked_ids', 'tracked_bbox', 'tid', 'bbox', 'tracked_time']
        history_keys    = ['appe', 'loca', 'pose', 'uv'] if self.cfg.render.enable else []
        prediction_keys = ['prediction_uv', 'prediction_pose', 'prediction_loca'] if self.cfg.render.enable else []
        extra_keys_1    = ['center', 'scale', 'size', 'img_path', 'img_name', 'class_name', 'conf']
        extra_keys_2    = ['smpl', 'camera', '3d_joints', 'embedding', 'mask']
        history_keys    = history_keys + extra_keys_1 + extra_keys_2
        visual_store_   = eval_keys + history_keys + prediction_keys
        tmp_keys_       = ['uv', 'prediction_uv', 'prediction_pose', 'prediction_loca']
        
        # eval mode
        self.eval()
        
        # setup rendering and deep sort
        self.setup_deepsort()
        
        # process the source video and return a list of frames
        # source can be a video file, a youtube link or a image folder
        list_of_frames, additional_data = self.get_frames_from_source()
         
        # check if the video is already processed                                  
        if(not(self.cfg.overwrite) and os.path.isfile(self.cfg.video.output_dir + '/results/' + str(self.cfg.video_seq) + '.pkl')): return 0
        
        log.info("Saving tracks at : " + self.cfg.video.output_dir + '/results/' + str(self.cfg.video_seq))
        
        # create subfolders for saving additional results
        try:
            os.makedirs(self.cfg.video.output_dir + '/results', exist_ok=True)  
            os.makedirs(self.cfg.video.output_dir + '/_TMP', exist_ok=True)  
        except: 
            pass
        
    
        try: 
            
            list_of_frames = list_of_frames if self.cfg.phalp.start_frame==-1 else list_of_frames[self.cfg.phalp.start_frame:self.cfg.phalp.end_frame]
            list_of_shots = self.get_list_of_shots(list_of_frames)
            
            tracked_frames = []
            final_visuals_dic = {}

            for t_, frame_name in track(enumerate(list_of_frames), 
                                        description="Tracking : " + self.cfg.video_seq, 
                                        total=len(list_of_frames),
                                        disable=self.cfg.debug
                                        ):
                    
                if(self.cfg.render.enable):
                    # reset the renderer
                    self.visualizer.reset_render(self.cfg.render.res*self.cfg.render.up_scale)    
                
                image_frame               = cv2.imread(frame_name)
                img_height, img_width, _  = image_frame.shape
                new_image_size            = max(img_height, img_width)
                top, left                 = (new_image_size - img_height)//2, (new_image_size - img_width)//2,
                measurments               = [img_height, img_width, new_image_size, left, top]
                self.cfg.phalp.shot       = 1 if t_ in list_of_shots else 0

                ############ detection ##############
                pred_bbox, pred_masks, pred_scores, pred_classes, gt = self.get_detections(image_frame, frame_name, t_)
                
                ############ HMAR ##############
                detections = []
                for bbox, mask, score, cls_id, gt_id in zip(pred_bbox, pred_masks, pred_scores, pred_classes, gt):
                    if bbox[2]-bbox[0]<50 or bbox[3]-bbox[1]<100: continue
                    detection_data = self.get_human_features(image_frame, mask, bbox, score, frame_name, cls_id, t_, measurments, gt_id)
                    detections.append(Detection(detection_data))

                ############ tracking ##############
                self.tracker.predict()
                self.tracker.update(detections, t_, frame_name, self.cfg.phalp.shot)

                ############ record the results ##############
                final_visuals_dic.setdefault(frame_name, {'time': t_, 'shot': self.cfg.phalp.shot})
                if(self.cfg.render.enable): final_visuals_dic[frame_name]['frame'] = image_frame
                for key_ in visual_store_: final_visuals_dic[frame_name][key_] = []
                
                ############ record the track states ##############
                for tracks_ in self.tracker.tracks:
                    if(frame_name not in tracked_frames): tracked_frames.append(frame_name)
                    if(not(tracks_.is_confirmed())): continue
                    
                    track_id        = tracks_.track_id
                    track_data_hist = tracks_.track_data['history'][-1]
                    track_data_pred = tracks_.track_data['prediction']

                    final_visuals_dic[frame_name]['tid'].append(track_id)
                    final_visuals_dic[frame_name]['bbox'].append(track_data_hist['bbox'])
                    final_visuals_dic[frame_name]['tracked_time'].append(tracks_.time_since_update)

                    for hkey_ in history_keys:     final_visuals_dic[frame_name][hkey_].append(track_data_hist[hkey_])
                    for pkey_ in prediction_keys:  final_visuals_dic[frame_name][pkey_].append(track_data_pred[pkey_.split('_')[1]][-1])

                    if(tracks_.time_since_update==0):
                        final_visuals_dic[frame_name]['tracked_ids'].append(track_id)
                        final_visuals_dic[frame_name]['tracked_bbox'].append(track_data_hist['bbox'])
                        
                        if(tracks_.hits==self.cfg.phalp.n_init):
                            for pt in range(self.cfg.phalp.n_init-1):
                                track_data_hist_ = tracks_.track_data['history'][-2-pt]
                                track_data_pred_ = tracks_.track_data['prediction']
                                frame_name_      = tracked_frames[-2-pt]
                                final_visuals_dic[frame_name_]['tid'].append(track_id)
                                final_visuals_dic[frame_name_]['bbox'].append(track_data_hist_['bbox'])
                                final_visuals_dic[frame_name_]['tracked_ids'].append(track_id)
                                final_visuals_dic[frame_name_]['tracked_bbox'].append(track_data_hist_['bbox'])
                                final_visuals_dic[frame_name_]['tracked_time'].append(0)

                                for hkey_ in history_keys:    final_visuals_dic[frame_name_][hkey_].append(track_data_hist_[hkey_])
                                for pkey_ in prediction_keys: final_visuals_dic[frame_name_][pkey_].append(track_data_pred_[pkey_.split('_')[1]][-1])

                ############ save the video ##############
                if(self.cfg.render.enable and t_>=self.cfg.phalp.n_init):                    
                    d_ = self.cfg.phalp.n_init+1 if(t_+1==len(list_of_frames)) else 1
                    for t__ in range(t_, t_+d_):
                        frame_key = list_of_frames[t__-self.cfg.phalp.n_init]
                        rendered_, f_size = self.visualizer.render_video(final_visuals_dic[frame_key])      
                        if(t__-self.cfg.phalp.n_init in list_of_shots): cv2.rectangle(rendered_, (0,0), (f_size[0], f_size[1]), (0,0,255), 4)
                        if(t__-self.cfg.phalp.n_init==0):
                            file_name = self.cfg.video.output_dir + '/PHALP_' + str(self.cfg.video_seq) + '_'+ str(self.cfg.experiment_name) + '.mp4'
                            video_file = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, frameSize=f_size)
                        video_file.write(rendered_)
                        del final_visuals_dic[frame_key]['frame']
                        for tkey_ in tmp_keys_:  del final_visuals_dic[frame_key][tkey_] 

            joblib.dump(final_visuals_dic, self.cfg.video.output_dir + '/results/' + self.cfg.track_dataset + "_" + str(self.cfg.video_seq) + '.pkl')
            if(self.cfg.use_gt): joblib.dump(self.tracker.tracked_cost, self.cfg.video.output_dir + '/results/' + str(self.cfg.video_seq) + '_' + str(self.cfg.start_frame) + '_distance.pkl')
            if(self.cfg.render.enable): video_file.release()
            
        except Exception as e: 
            print(e)
            print(traceback.format_exc())     

    def get_frames_from_source(self):
    
        source_path = self.cfg.video.source
        
        # {key: frame name, value: {"bbox": None, "extra data": None}}
        # TODO: add this feature.
        additional_data = {}
        
        # check for youtube video
        if(source_path.startswith("https://") or source_path.startswith("http://")):
            video_name = source_path[-11:]
            img_path   = "_DEMO/" + video_name + "/img/"
            os.system("rm -rf " + "_DEMO/" + video_name)
            os.makedirs("_DEMO/" + video_name, exist_ok=True)    
            os.makedirs("_DEMO/" + video_name + "/img", exist_ok=True)    
            youtube_video = YouTube(source_path)
            log.info(f'Youtube Title: {youtube_video.title}')
            log.info(f'Video Duration: {youtube_video.length / 60:.2f} minutes')
            youtube_video.streams.get_by_itag(136).download(output_path = "_DEMO/" + video_name, filename="youtube.mp4")
            fe = FrameExtractor("_DEMO/" + video_name + "/youtube.mp4")
            log.info('Number of frames: ' + str(fe.n_frames))
            fe.extract_frames(every_x_frame=1, img_name='', dest_path= "_DEMO/" + video_name + "/img/", start_frame=self.cfg.video.start_frame, end_frame=self.cfg.video.end_frame)
            list_of_frames = sorted(glob.glob("_DEMO/" + video_name + "/img/*.jpg"))
    
        # extract frames from video
        elif(source_path.endswith(".mp4")):
            video_name = source_path.split('/')[-1].split('.')[0]
            img_path   = "_DEMO/" + video_name + "/img/"
            os.system("rm -rf " + "_DEMO/" + video_name)
            os.makedirs("_DEMO/" + video_name, exist_ok=True)    
            os.makedirs("_DEMO/" + video_name + "/img", exist_ok=True)    
            fe = FrameExtractor(source_path)
            log.info('Number of frames: ' + str(fe.n_frames))
            fe.extract_frames(every_x_frame=1, img_name='', dest_path= "_DEMO/" + video_name + "/img/", start_frame=self.cfg.video.start_frame, end_frame=self.cfg.video.end_frame)
            list_of_frames = sorted(glob.glob("_DEMO/" + video_name + "/img/*.jpg"))
        
        # read from image folder
        elif(os.path.isdir(source_path)):
            video_name = source_path.split('/')[-1]
            img_path   = source_path
            list_of_frames = sorted(glob.glob(source_path + "/*.jpg"))
            
        else:
            raise Exception("Invalid source path")
        
        # setup the video name and the root folder of the frames.
        self.cfg.video_seq = video_name
        self.cfg.base_path = img_path
        
        return list_of_frames, additional_data        

    def get_detections(self, image, frame_name, t_):

        if("mask" in self.cfg.phalp.detection_type):
            outputs     = self.detector(image)   
            instances   = outputs['instances']
            instances   = instances[instances.pred_classes==0]
            instances   = instances[instances.scores>self.cfg.phalp.low_th_c]

            pred_bbox   = instances.pred_boxes.tensor.cpu().numpy()
            pred_masks  = instances.pred_masks.cpu().numpy()
            pred_scores = instances.scores.cpu().numpy()
            pred_classes= instances.pred_classes.cpu().numpy()
            
        ground_truth = [1 for i in list(range(len(pred_scores)))]

        return pred_bbox, pred_masks, pred_scores, pred_classes, ground_truth

    def get_human_features(self, image, seg_mask, bbox, score, frame_name, cls_id, t_, measurments, gt=1):
        
        img_height, img_width, new_image_size, left, top = measurments                
        
        # Encode the mask for storing, borrowed from tao dataset
        # https://github.com/TAO-Dataset/tao/blob/master/scripts/detectors/detectron2_infer.py
        masks_decoded = np.array(np.expand_dims(seg_mask, 2), order='F', dtype=np.uint8)
        rles = mask_utils.encode(masks_decoded)
        for rle in rles: rle["counts"] = rle["counts"].decode("utf-8")
        
        seg_mask = seg_mask.astype(int)*255
        if(len(seg_mask.shape)==2):
            seg_mask        = np.expand_dims(seg_mask, 2)
            seg_mask        = np.repeat(seg_mask, 3, 2)
        
        center_             = np.array([(bbox[2] + bbox[0])/2, (bbox[3] + bbox[1])/2])
        scale_              = np.array([(bbox[2] - bbox[0]), (bbox[3] - bbox[1])])
        mask_tmp            = process_mask(seg_mask.astype(np.uint8), center_, 1.0*np.max(scale_))
        image_tmp           = process_image(image, center_, 1.0*np.max(scale_))
        masked_image        = torch.cat((image_tmp, mask_tmp[:1, :, :]), 0)
        ratio               = 1.0/int(new_image_size)*self.cfg.render.res

        with torch.no_grad():
            hmar_out        = self.HMAR(masked_image.unsqueeze(0).cuda()) 

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
            appe_embedding  = self.HMAR.autoencoder_hmar(uv_vector, en=True)
            appe_embedding  = appe_embedding.view(1, -1)
            pred_smpl_params, pred_joints_2d, pred_joints, pred_cam  = self.HMAR.get_3d_parameters(torch.cat((pose_embedding, pose_embedding), 1),
                                                                                               np.array([[1.0, 0, 0]]),
                                                                                               center=(center_ + [left, top])*ratio,
                                                                                               img_size=self.cfg.render.res,
                                                                                               scale=np.reshape(np.array([max(scale_)]), (1, 1))*ratio,
                                                                                               texture=uv_vector[:, :3, :, :]*5.0, render=False)
            
            pred_smpl_params = {k:v[0].cpu().numpy() for k,v in pred_smpl_params.items()}
            pred_joints_2d_ = pred_joints_2d.reshape(-1,)/self.cfg.render.res
            pred_cam_ = pred_cam.view(-1,)
            pred_joints_2d_.contiguous()
            pred_cam_.contiguous()
            loca_embedding  = torch.cat((pred_joints_2d_, pred_cam_, pred_cam_, pred_cam_), 0)
            
        full_embedding    = torch.cat((appe_embedding[0].cpu(), pose_embedding[0].cpu(), pose_embedding[0].cpu(), loca_embedding.cpu()), 0)

        detection_data = {
                              "bbox"            : np.array([bbox[0], bbox[1], (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]),
                              "mask"            : rles,
                              "conf"            : score, 
                              "appe"            : appe_embedding[0].cpu().numpy(), 
                              "pose"            : torch.cat((pose_embedding[0].cpu(), pose_embedding[0].cpu()), 0).cpu().numpy(), 
                              "loca"            : loca_embedding.cpu().numpy(), 
                              "embedding"       : full_embedding, 
                              "uv"              : uv_vector[0].cpu().numpy(), 
                              "center"          : center_,
                              "scale"           : scale_,
                              "smpl"            : pred_smpl_params,
                              "camera"          : pred_cam_.cpu().numpy(),
                              "3d_joints"       : pred_joints[0].cpu().numpy(),
                              "2d_joints"       : pred_joints_2d_[0].cpu().numpy(),
                              "size"            : [img_height, img_width],
                              "img_path"        : frame_name,
                              "img_name"        : frame_name.split('/')[-1],
                              "class_name"      : cls_id,
                              "ground_truth"    : gt,
                              "time"            : t_,
                         }
        
        return detection_data

    def forward_for_tracking(self, vectors, attibute="A", time=1):
        
        if(attibute=="P"):

            vectors_pose         = vectors[0]
            vectors_data         = vectors[1]
            vectors_time         = vectors[2]
        
            en_pose              = torch.from_numpy(vectors_pose)
            en_data              = torch.from_numpy(vectors_data)
            en_time              = torch.from_numpy(vectors_time)
            
            if(len(en_pose.shape)!=3):
                en_pose          = en_pose.unsqueeze(0) # (BS, 7, 4096)
                en_time          = en_time.unsqueeze(0) # (BS, 7)
                en_data          = en_data.unsqueeze(0) # (BS, 7, 6)
            
            BS                   = en_pose.size(0)
            history              = en_pose.size(1)
            attn                 = torch.ones(BS, history, history)

            xf_trans             = self.HMAR.pose_transformer.relational(en_pose[:, :, 2048:].float().cuda(), en_data.float().cuda(), attn.float().cuda())  #bs, 13, 2048
            xf_trans             = xf_trans.view(-1, 2048)
            movie_strip_t        = self.HMAR.pose_transformer.smpl_head_prediction(en_pose[:, :, 2048:].float().view(-1, 2048).cuda(), xf_trans)  #bs*13, 2048 -> bs*13, 12, 2048
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
                loc_                 = loc_.shape[0] - torch.sum(loc_)+1

                M = t_[:, np.newaxis]**[0, 1]
                time_ = 48 if time[bs]>48 else time[bs]

                clf = Ridge(alpha=5.0)
                clf.fit(M, n_)
                n_p = clf.predict(np.array([1, time_+1+t_[-1]]).reshape(1, -1))
                n_p = n_p[0]
                n_hat = clf.predict(np.hstack((np.ones((t_.size, 1)), t_.reshape((-1, 1)))))
                n_pi  = get_prediction_interval(n_, n_hat, t_, time_+1+t_[-1])

                clf  = Ridge(alpha=1.2)
                clf.fit(M, x0_)
                x_p  = clf.predict(np.array([1, time_+1+t_[-1]]).reshape(1, -1))
                x_p  = x_p[0]
                x_p_ = (x_p-0.5)*np.exp(n_p)/5000.0*256.0
                x_hat = clf.predict(np.hstack((np.ones((t_.size, 1)), t_.reshape((-1, 1)))))
                x_pi  = get_prediction_interval(x0_, x_hat, t_, time_+1+t_[-1])

                clf  = Ridge(alpha=2.0)
                clf.fit(M, y0_)
                y_p  = clf.predict(np.array([1, time_+1+t_[-1]]).reshape(1, -1))
                y_p  = y_p[0]
                y_p_ = (y_p-0.5)*np.exp(n_p)/5000.0*256.0
                y_hat = clf.predict(np.hstack((np.ones((t_.size, 1)), t_.reshape((-1, 1)))))
                y_pi  = get_prediction_interval(y0_, y_hat, t_, time_+1+t_[-1])
                
                new_en_loca_n.append([x_p_, y_p_, np.exp(n_p), x_pi/loc_, y_pi/loc_, np.exp(n_pi)/loc_, 1, 1, 0])
                en_loca_xy[bs, -1, 44, 0] = x_p
                en_loca_xy[bs, -1, 44, 1] = y_p
                
            new_en_loca_n        = torch.from_numpy(np.array(new_en_loca_n))
            xt                   = torch.cat((en_loca_xy[:, -1, :, :].view(BS, 90), (new_en_loca_n.float()).view(BS, 9)), 1)

        return xt

    def get_uv_distance(self, t_uv, d_uv):
        t_uv         = torch.from_numpy(t_uv).cuda().float()
        d_uv         = torch.from_numpy(d_uv).cuda().float()
        d_mask       = d_uv[3:, :, :]>0.5
        t_mask       = t_uv[3:, :, :]>0.5
        
        mask_dt      = torch.logical_and(d_mask, t_mask)
        mask_dt      = mask_dt.repeat(4, 1, 1)
        mask_        = torch.logical_not(mask_dt)
        
        t_uv[mask_]  = 0.0
        d_uv[mask_]  = 0.0

        with torch.no_grad():
            t_emb    = self.HMAR.autoencoder_hmar(t_uv.unsqueeze(0), en=True)
            d_emb    = self.HMAR.autoencoder_hmar(d_uv.unsqueeze(0), en=True)
        t_emb        = t_emb.view(-1)/10**3
        d_emb        = d_emb.view(-1)/10**3
        return t_emb.cpu().numpy(), d_emb.cpu().numpy(), torch.sum(mask_dt).cpu().numpy()/4/256/256/2

    def get_list_of_shots(self, list_of_frames):
        list_of_shots    = []
        if(self.cfg.detect_shots):
            video_tmp_name   = self.cfg.video.output_dir + "/_TMP/" + str(self.cfg.video_seq) + ".mp4"
            for ft_, fname_ in enumerate(list_of_frames):
                im_ = cv2.imread(fname_)
                if(ft_==0): video_file = cv2.VideoWriter(video_tmp_name, cv2.VideoWriter_fourcc(*'mp4v'), 24, frameSize=(im_.shape[1], im_.shape[0]))
                video_file.write(im_)
            video_file.release()
            try:    scene_list = detect(video_tmp_name, AdaptiveDetector())
            except: pass
            os.system("rm " + video_tmp_name)
            for scene in scene_list:
                print(scene)
                list_of_shots.append(scene[0].get_frames())
                list_of_shots.append(scene[1].get_frames())
            list_of_shots = np.unique(list_of_shots)
            list_of_shots = list_of_shots[1:-1]
        return list_of_shots

    def cached_download_from_drive(self):
        """Download a file from Google Drive if it doesn't exist yet.
        :param url: the URL of the file to download
        :param path: the path to save the file to
        """
        os.makedirs("_DATA/", exist_ok=True)

        if not os.path.exists("_DATA/models/smpl/SMPL_NEUTRAL.pkl"):
            # We are downloading the SMPL model here for convenience. Please accept the license
            # agreement on the SMPL website: https://smpl.is.tue.mpg.
            os.system('mkdir -p _DATA/models')
            os.system('mkdir -p _DATA/models/smpl')
            # os.system('wget https://github.com/classner/up/raw/master/models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pickle')
            os.system('wget https://github.com/classner/up/raw/master/models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')

            convert_pkl('basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
            os.system('rm basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
            os.system('mv basicModel_neutral_lbs_10_207_0_v1.0.0_p3.pkl _DATA/models/smpl/SMPL_NEUTRAL.pkl')

        
        download_files = {
            "posetrack_gt_data.pickle" : "https://drive.google.com/file/d/1pmtc3l6W8AXScRnhV_KIYqTrzM0-Qb-D/view?usp=sharing",
            "posetrack-val_videos.npy" : "https://drive.google.com/file/d/1ln5M1Lro7mKH-IQJ4kA0Uj0Xks9apQOH/view?usp=sharing",
            "texture.npz"              : "https://drive.google.com/file/d/1T37ym8d6tDxLpdOaCJyQ9bZ1ejIvJAoH/view?usp=sharing",
            "SMPL_to_J19.pkl"          : "https://drive.google.com/file/d/1UWsrBc5XH1ZkB_cfIR9aJVGtwE_0NOPP/view?usp=sharing",
            "smpl_mean_params.npz"     : "https://drive.google.com/file/d/11mMhMmPJqtDNoOQWA_B4neVpOW_3unCE/view?usp=sharing",
            "J_regressor_h36m.npy"     : "https://drive.google.com/file/d/1I0QZqGJpyP7Hv5BypmxqX60gwjX2nPNn/view?usp=sharing",
            "hmar_v2_weights.pth"      : "https://drive.google.com/file/d/1_wZcPv8MxPoZyEGA9rI5ayXiB7Fhhj4b/view?usp=sharing",
            "hmmr_v2_weights.pt"       : "https://drive.google.com/file/d/1hMjFoyVkoHIiYJBndvCoy2fs9T8j-ULU/view?usp=sharing",
        }
        
        for file_name, url in download_files.items():
            if not os.path.exists("_DATA/" + file_name):
                print("Downloading file: " + file_name)
                output = gdown.cached_download(url, "_DATA/" + file_name, fuzzy=True)

                assert os.path.exists("_DATA/" + file_name), f"{output} does not exist"