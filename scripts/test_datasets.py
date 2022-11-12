import argparse
import os
import warnings

import numpy as np
from demo_online import options, test_tracker
from utils.utils import str2bool

from PHALP import PHALP_tracker

warnings.filterwarnings('ignore')
  
    
if __name__ == '__main__':
    
    opt                   = options().parse()
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