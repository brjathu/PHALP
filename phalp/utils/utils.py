from __future__ import absolute_import, division, print_function

import argparse
import datetime
import math
import os
import pickle

import cv2
import dill
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim
from colordict import *


def numpy_to_torch_image(ndarray):
    torch_image              = torch.from_numpy(ndarray)
    torch_image              = torch_image.unsqueeze(0)
    torch_image              = torch_image.permute(0, 3, 1, 2)
    torch_image              = torch_image[:, [2,1,0], :, :]
    return torch_image
  
def get_colors():  
    try:  
        RGB_tuples                   = np.vstack([np.loadtxt("assets/colors.txt", skiprows=1) , np.random.uniform(0, 255, size=(10000, 3)), [[0,0,0]]])
        b                            = np.where(RGB_tuples==0)
        RGB_tuples[b]                = 1    
    except:
        colormap = np.array(list(ColorDict(norm=255, mode='rgb', palettes_path="", is_grayscale=False, palettes='all').values()))
        RGB_tuples = np.vstack([colormap[1:, :3], np.random.uniform(0, 255, size=(10000, 3)), [[0,0,0]]])
        
    return RGB_tuples

def get_prediction_interval(y, y_hat, x, x_hat):
    n     = y.size
    resid = y - y_hat
    s_err = np.sqrt(np.sum(resid**2) / (n-2))                    # standard deviation of the error
    t     = stats.t.ppf(0.975, n - 2)                            # used for CI and PI bands
    pi    = t * s_err * np.sqrt( 1 + 1/n + (x_hat - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    return pi

def convert_pkl(old_pkl):
    # Code adapted from https://github.com/nkolot/ProHMR
    # Convert SMPL pkl file to be compatible with Python 3
    # Script is from https://rebeccabilbro.github.io/convert-py2-pickles-to-py3/

    # Make a name for the new pickle
    new_pkl = os.path.splitext(os.path.basename(old_pkl))[0]+"_p3.pkl"

    # Convert Python 2 "ObjectType" to Python 3 object
    dill._dill._reverse_typemap["ObjectType"] = object

    # Open the pickle using latin1 encoding
    with open(old_pkl, "rb") as f:
        loaded = pickle.load(f, encoding="latin1")

    # Re-save as Python 3 pickle
    with open(new_pkl, "wb") as outfile:
        pickle.dump(loaded, outfile)

class FrameExtractor():
    '''
    Class used for extracting frames from a video file.
    '''
    def __init__(self, video_path):
        self.video_path = video_path
        self.vid_cap    = cv2.VideoCapture(video_path)
        self.n_frames   = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps        = int(self.vid_cap.get(cv2.CAP_PROP_FPS))

    def get_video_duration(self):
        duration = self.n_frames/self.fps
        print(f'Duration: {datetime.timedelta(seconds=duration)}')

    def get_n_images(self, every_x_frame):
        n_images = math.floor(self.n_frames / every_x_frame) + 1
        print(f'Extracting every {every_x_frame} (nd/rd/th) frame would result in {n_images} images.')

    def extract_frames(self, every_x_frame, img_name, dest_path=None, img_ext = '.jpg', start_frame=1000, end_frame=2000):
        if not self.vid_cap.isOpened():
            self.vid_cap = cv2.VideoCapture(self.video_path)

        if dest_path is None:
            dest_path = os.getcwd()
        else:
            if not os.path.isdir(dest_path):
                os.mkdir(dest_path); print(f'Created the following directory: {dest_path}')

        frame_cnt = 0; img_cnt = 0
        while self.vid_cap.isOpened():
            success,image = self.vid_cap.read()
            if not success: break
            if frame_cnt % every_x_frame == 0 and frame_cnt > start_frame and frame_cnt < end_frame:
                img_path = os.path.join(dest_path, ''.join([img_name,  '%06d' % (img_cnt+1), img_ext]))
                cv2.imwrite(img_path, image)
                img_cnt += 1
            frame_cnt += 1
        self.vid_cap.release()
        cv2.destroyAllWindows()
        

def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device, dtype=points.dtype)
    K[:,0,0] = focal_length[:,0]
    K[:,1,1] = focal_length[:,1]
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points    
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / (points[:,:,-1].unsqueeze(-1))
    
    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]
        
def compute_uvsampler(vt, ft, tex_size=6):
    """
    For this mesh, pre-computes the UV coordinates for
    F x T x T points.
    Returns F x T x T x 2
    """
    uv = obj2nmr_uvmap(ft, vt, tex_size=tex_size)
    uv = uv.reshape(-1, tex_size, tex_size, 2)
    return uv

def obj2nmr_uvmap(ft, vt, tex_size=6):
    """
    Converts obj uv_map to NMR uv_map (F x T x T x 2),
    where tex_size (T) is the sample rate on each face.
    """
    # This is F x 3 x 2
    uv_map_for_verts = vt[ft]

    # obj's y coordinate is [1-0], but image is [0-1]
    uv_map_for_verts[:, :, 1] = 1 - uv_map_for_verts[:, :, 1]

    # range [0, 1] -> [-1, 1]
    uv_map_for_verts = (2 * uv_map_for_verts) - 1

    alpha = np.arange(tex_size, dtype=np.float64) / (tex_size - 1)
    beta = np.arange(tex_size, dtype=np.float64) / (tex_size - 1)
    import itertools

    # Barycentric coordinate values
    coords = np.stack([p for p in itertools.product(*[alpha, beta])])

    # Compute alpha, beta (this is the same order as NMR)
    v2 = uv_map_for_verts[:, 2]
    v0v2 = uv_map_for_verts[:, 0] - uv_map_for_verts[:, 2]
    v1v2 = uv_map_for_verts[:, 1] - uv_map_for_verts[:, 2]
    # Interpolate the vertex uv values: F x 2 x T*2
    uv_map = np.dstack([v0v2, v1v2]).dot(coords.T) + v2.reshape(-1, 2, 1)

    # F x T*2 x 2  -> F x T x T x 2
    uv_map = np.transpose(uv_map, (0, 2, 1)).reshape(-1, tex_size, tex_size, 2)

    return uv_map
