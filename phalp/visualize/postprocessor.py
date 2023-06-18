import copy

import os
import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from phalp.utils.utils import progress_bar
from phalp.utils.utils_tracks import create_fast_tracklets, get_tracks
from phalp.utils.utils import pose_camera_vector_to_smpl
from phalp.utils.lart_utils import to_ava_labels


class Postprocessor(nn.Module):
    
    def __init__(self, cfg, phalp_tracker):
        super(Postprocessor, self).__init__()
        
        self.cfg = cfg
        self.device = 'cuda'
        self.phalp_tracker = phalp_tracker

    def post_process(self, final_visuals_dic):

        if(self.cfg.post_process.apply_smoothing):
            final_visuals_dic_ = copy.deepcopy(final_visuals_dic)
            track_dict = get_tracks(final_visuals_dic_)

            for tid_ in track_dict.keys():
                fast_track_ = create_fast_tracklets(track_dict[tid_])
            
                with torch.no_grad():
                    smoothed_fast_track_ = self.phalp_tracker.pose_predictor.smooth_tracks(fast_track_, moving_window=True, step=32, window=32)

                for i_ in range(smoothed_fast_track_['pose_shape'].shape[0]):
                    f_key = smoothed_fast_track_['frame_name'][i_]
                    tids_ = np.array(final_visuals_dic_[f_key]['tid'])
                    idx_  = np.where(tids_==tid_)[0]
                    
                    if(len(idx_)>0):

                        pose_shape_ = smoothed_fast_track_['pose_shape'][i_]
                        smpl_camera = pose_camera_vector_to_smpl(pose_shape_[0].cpu().numpy())
                        smpl_ = smpl_camera[0]
                        camera = smpl_camera[1]
                        camera_ = smoothed_fast_track_['cam_smoothed'][i_][0].cpu().numpy()

                        dict_ = {}
                        for k, v in smpl_.items():
                            dict_[k] = v

                        if(final_visuals_dic[f_key]['tracked_time'][idx_[0]]>0):
                            final_visuals_dic[f_key]['camera'][idx_[0]] = np.array([camera_[0], camera_[1], 200*camera_[2]])
                            final_visuals_dic[f_key]['smpl'][idx_[0]] = copy.deepcopy(dict_)
                            final_visuals_dic[f_key]['tracked_time'][idx_[0]] = -1
                        
                        # attach ava labels
                        ava_ = smoothed_fast_track_['ava_action'][i_]
                        ava_ = ava_.cpu()
                        ava_labels, _ = to_ava_labels(ava_, self.cfg)
                        final_visuals_dic[f_key].setdefault('label', {})[tid_] = ava_labels
                        final_visuals_dic[f_key].setdefault('ava_action', {})[tid_] = ava_
                        
        
        return final_visuals_dic

    def run_lart(self, phalp_pkl_path):
        
        # lart_output = {}
        video_pkl_name = phalp_pkl_path.split("/")[-1].split(".")[0]
        final_visuals_dic = joblib.load(phalp_pkl_path)

        os.makedirs(self.cfg.video.output_dir + "/results_temporal/", exist_ok=True)
        os.makedirs(self.cfg.video.output_dir + "/results_temporal_videos/", exist_ok=True)
        save_pkl_path = os.path.join(self.cfg.video.output_dir, "results_temporal/", video_pkl_name + ".pkl")
        save_video_path = os.path.join(self.cfg.video.output_dir, "results_temporal_videos/", video_pkl_name + "_.mp4")

        if(os.path.exists(save_pkl_path) and not(self.cfg.overwrite)):
            return 0
        
        # aplly smoothing/action recognition etc.
        final_visuals_dic  = self.post_process(final_visuals_dic)
        
        # render the video
        if(self.cfg.render.enable):
            self.offline_render(final_visuals_dic, save_pkl_path, save_video_path)
        
        joblib.dump(final_visuals_dic, save_pkl_path)

    def run_renderer(self, phalp_pkl_path):
        
        video_pkl_name = phalp_pkl_path.split("/")[-1].split(".")[0]
        final_visuals_dic = joblib.load(phalp_pkl_path)

        os.makedirs(self.cfg.video.output_dir + "/videos/", exist_ok=True)
        os.makedirs(self.cfg.video.output_dir + "/videos_tmp/", exist_ok=True)
        save_pkl_path = os.path.join(self.cfg.video.output_dir, "videos_tmp/", video_pkl_name + ".pkl")
        save_video_path = os.path.join(self.cfg.video.output_dir, "videos/", video_pkl_name + ".mp4")

        if(os.path.exists(save_pkl_path) and not(self.cfg.overwrite)):
            return 0
        
        # render the video
        self.offline_render(final_visuals_dic, save_pkl_path, save_video_path)


    def offline_render(self, final_visuals_dic, save_pkl_path, save_video_path):
        
        video_pkl_name = save_pkl_path.split("/")[-1].split(".")[0]
        list_of_frames = list(final_visuals_dic.keys())
        
        for t_, frame_path in progress_bar(enumerate(list_of_frames), description="Rendering : " + video_pkl_name, total=len(list_of_frames), disable=False):
            
            image = self.phalp_tracker.io_manager.read_frame(frame_path)

            ################### Front view #########################
            self.cfg.render.up_scale = int(self.cfg.render.output_resolution / self.cfg.render.res)
            self.phalp_tracker.visualizer.reset_render(self.cfg.render.res*self.cfg.render.up_scale)
            final_visuals_dic[frame_path]['frame'] = image
            panel_render, f_size = self.phalp_tracker.visualizer.render_video(final_visuals_dic[frame_path])      
            del final_visuals_dic[frame_path]['frame']

            # resize the image back to render resolution
            panel_rgb = cv2.resize(image, (f_size[0], f_size[1]), interpolation=cv2.INTER_AREA)

            # save the predicted actions labels
            if('label' in final_visuals_dic[frame_path]):
                labels_to_save = []
                for tid_ in final_visuals_dic[frame_path]['label']:
                    ava_labels = final_visuals_dic[frame_path]['label'][tid_]
                    labels_to_save.append(ava_labels)
                labels_to_save = np.array(labels_to_save)

            panel_1 = np.concatenate((panel_rgb, panel_render), axis=1)
            final_panel = panel_1

            self.phalp_tracker.io_manager.save_video(save_video_path, final_panel, (final_panel.shape[1], final_panel.shape[0]), t=t_)
            t_ += 1

        self.phalp_tracker.io_manager.close_video()


