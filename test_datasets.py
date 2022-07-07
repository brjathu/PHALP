import os
import argparse
import warnings
import numpy as np

from PHALP import PHALP_tracker
from utils.utils import str2bool
from demo_online import test_tracker

warnings.filterwarnings('ignore')
  
    
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
    parser.add_argument('--detect_shots', type=str2bool, nargs='?', const=True, default=False)
    
    parser.add_argument('--base_path', type=str)
    parser.add_argument('--video_seq', type=str, default='_DATA/posetrack/list_videos_val.npy')
    parser.add_argument('--youtube_id', type=str, default="xEH_5T9jMVU")
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

    if(opt.track_dataset=='posetrack-val'):
        
        videos = np.load('_DATA/posetrack-val_videos.npy')
        batch_length = len(videos)//100
        start_       = opt.batch_id*(batch_length+1)
        end_         = (opt.batch_id+1)*(batch_length+1)
        if(start_>len(videos)): exit()
        if(end_  >len(videos)): end_ = len(videos)
        videos    = videos[start_:end_] if opt.batch_id>=0 else videos
        
        if(not(opt.all_videos)): videos = ['000583_mpii_test']

        for video in videos:
            opt.base_path = '_DATA/posetrack_2018/images/val/'
            opt.video_seq = video
            test_tracker(opt, phalp_tracker)