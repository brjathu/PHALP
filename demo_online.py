import os
import cv2
import time
import joblib
import argparse
import warnings
import traceback
import numpy as np

from PHALP import PHALP_tracker
from deep_sort_ import nn_matching
from deep_sort_.detection import Detection
from deep_sort_.tracker import Tracker

from utils.make_video import render_frame_main_online
from utils.utils import FrameExtractor, str2bool
from pytube import YouTube

warnings.filterwarnings('ignore')
  
        
def test_tracker(opt, phalp_tracker: PHALP_tracker):
    
    eval_keys       = ['tracked_ids', 'tracked_bbox', 'tid', 'bbox', 'tracked_time']
    history_keys    = ['appe', 'loca', 'pose', 'uv'] if opt.render else []
    prediction_keys = ['prediction_uv', 'prediction_pose', 'prediction_loca'] if opt.render else []
    extra_keys_1    = ['center', 'scale', 'size', 'img_path', 'img_name', 'mask_name', 'conf']
    extra_keys_2    = ['smpl', '3d_joints', 'camera', 'embedding']
    history_keys    = history_keys + extra_keys_1 + extra_keys_2
    visual_store_   = eval_keys + history_keys + prediction_keys
    tmp_keys_       = ['uv', 'prediction_uv', 'prediction_pose', 'prediction_loca']
    
                                                
    if(not(opt.overwrite) and os.path.isfile('out/' + opt.storage_folder + '/results/' + str(opt.video_seq) + '.pkl')): return 0
    print(opt.storage_folder + '/results/' + str(opt.video_seq))
    
    try:
        os.makedirs('out/' + opt.storage_folder, exist_ok=True)  
        os.makedirs('out/' + opt.storage_folder + '/results', exist_ok=True)  
        os.makedirs('out/' + opt.storage_folder + '/_TMP', exist_ok=True)  
    except: pass
    
    phalp_tracker.eval()
    phalp_tracker.HMAR.reset_nmr(opt.res)    
    
    metric  = nn_matching.NearestNeighborDistanceMetric(opt, opt.hungarian_th, opt.past_lookback)
    tracker = Tracker(opt, metric, max_age=opt.max_age_track, n_init=opt.n_init, phalp_tracker=phalp_tracker, dims=[4096, 4096, 99])  
        
    try: 
        
        main_path_to_frames = opt.base_path + '/' + opt.video_seq + opt.sample
        list_of_frames      = np.sort([i for i in os.listdir(main_path_to_frames) if '.jpg' in i])
        list_of_frames      = list_of_frames if opt.start_frame==-1 else list_of_frames[opt.start_frame:opt.end_frame]
            
        tracked_frames          = []
        final_visuals_dic       = {}

        for t_, frame_name in enumerate(list_of_frames):
            if(opt.verbose): 
                print('\n\n\nTime: ', opt.video_seq, frame_name, t_, time.time()-time_ if t_>0 else 0 )
                time_ = time.time()
            
            image_frame               = cv2.imread(main_path_to_frames + '/' + frame_name)
            img_height, img_width, _  = image_frame.shape
            new_image_size            = max(img_height, img_width)
            top, left                 = (new_image_size - img_height)//2, (new_image_size - img_width)//2,
            measurments               = [img_height, img_width, new_image_size, left, top]

            ############ detection ##############
            pred_bbox, pred_masks, pred_scores, mask_names, gt = phalp_tracker.get_detections(image_frame, frame_name, t_)
            
            ############ HMAR ##############
            detections = []
            for bbox, mask, score, mask_name, gt_id in zip(pred_bbox, pred_masks, pred_scores, mask_names, gt):
                if bbox[2]-bbox[0]<50 or bbox[3]-bbox[1]<100: continue
                detection_data = phalp_tracker.get_human_apl(image_frame, mask, bbox, score, [main_path_to_frames, frame_name], mask_name, t_, measurments, gt_id)
                detections.append(Detection(detection_data))

            ############ tracking ##############
            tracker.predict()
            tracker.update(detections, t_, frame_name, 0)

            ############ record the results ##############
            final_visuals_dic.setdefault(frame_name, {'time': t_})
            if(opt.render): final_visuals_dic[frame_name]['frame'] = image_frame
            for key_ in visual_store_: final_visuals_dic[frame_name][key_] = []
            
            for tracks_ in tracker.tracks:
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
                    
                    if(tracks_.hits==opt.n_init):
                        for pt in range(opt.n_init-1):
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
            if(opt.render and t_>=opt.n_init):
                d_ = opt.n_init+1 if(t_+1==len(list_of_frames)) else 1
                for t__ in range(t_, t_+d_):
                    frame_key          = list_of_frames[t__-opt.n_init]
                    rendered_, f_size  = render_frame_main_online(opt, phalp_tracker, frame_key, final_visuals_dic[frame_key], opt.track_dataset, track_id=-100)      
                    if(t__-opt.n_init==0):
                        file_name      = 'out/' + opt.storage_folder + '/PHALP_' + str(opt.video_seq) + '_'+ str(opt.detection_type) + '.mp4'
                        video_file     = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, frameSize=f_size)
                    video_file.write(rendered_)
                    del final_visuals_dic[frame_key]['frame']
                    for tkey_ in tmp_keys_:  del final_visuals_dic[frame_key][tkey_] 

        joblib.dump(final_visuals_dic, 'out/' + opt.storage_folder + '/results/' + opt.track_dataset + "_" + str(opt.video_seq) + opt.post_fix  + '.pkl')
        if(opt.use_gt): joblib.dump(tracker.tracked_cost, 'out/' + opt.storage_folder + '/results/' + str(opt.video_seq) + '_' + str(opt.start_frame) + '_distance.pkl')
        if(opt.render): video_file.release()
        
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
    parser.add_argument('--overwrite', type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--low_th_c', type=float, default=0.95)
    parser.add_argument('--hungarian_th', type=float, default=100.0)
    parser.add_argument('--track_history', type=int, default=7)
    parser.add_argument('--max_age_track', type=int, default=20)
    parser.add_argument('--n_init',  type=int, default=1)
    parser.add_argument('--max_ids', type=int, default=50)
    parser.add_argument('--verbose', type=str2bool, nargs='?', const=True, default=False)
    
    parser.add_argument('--base_path', type=str)
    parser.add_argument('--video_seq', type=str, default='_DATA/posetrack/list_videos_val.npy')
    parser.add_argument('--all_videos', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--store_mask', type=str2bool, nargs='?', const=True, default=True)

    parser.add_argument('--render', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--render_type', type=str, default='HUMAN_HEAD_FAST')
    parser.add_argument('--render_up_scale', type=int, default=2)
    parser.add_argument('--res', type=int, default=256)
    parser.add_argument('--downsample',  type=int, default=1)
    
    parser.add_argument('--encode_type', type=str, default='3c')
    parser.add_argument('--cva_type', type=str, default='least_square')
    parser.add_argument('--past_lookback', type=int, default=1)
    parser.add_argument('--mask_type', type=str, default='feat')
    parser.add_argument('--detection_type', type=str, default='mask2')
    parser.add_argument('--start_frame', type=int, default='1000')
    parser.add_argument('--end_frame', type=int, default='1100')
    parser.add_argument('--store_extra_info', type=str2bool, nargs='?', const=True, default=False)
    
    opt                   = parser.parse_args()
    opt.sample            = ''
    opt.post_fix          = ''
    
    phalp_tracker         = PHALP_tracker(opt)
    phalp_tracker.cuda()
    phalp_tracker.eval()

    if(opt.track_dataset=='youtube'):   
        
        video_id = "xEH_5T9jMVU"
        video    = "youtube_" + video_id

        os.system("rm -rf " + "_DEMO/" + video)
        os.makedirs("_DEMO/" + video, exist_ok=True)    
        os.makedirs("_DEMO/" + video + "/img", exist_ok=True)    
        youtube_video = YouTube('https://www.youtube.com/watch?v=' + video_id)
        print(f'Title: {youtube_video.title}')
        print(f'Duration: {youtube_video.length / 60:.2f} minutes')
        youtube_video.streams.get_by_itag(136).download(output_path = "_DEMO/" + video, filename="youtube.mp4")
        fe = FrameExtractor("_DEMO/" + video + "/youtube.mp4")
        print('Number of frames: ', fe.n_frames)
        fe.extract_frames(every_x_frame=1, img_name='', dest_path= "_DEMO/" + video + "/img/", start_frame=1100, end_frame=1300)

        opt.base_path       = '_DEMO/'
        opt.video_seq       = video
        opt.sample          =  '/img/'
        test_tracker(opt, phalp_tracker)


            