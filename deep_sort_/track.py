"""
Modified code from https://github.com/nwojke/deep_sort
"""

import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.signal as signal
from scipy.ndimage.filters import gaussian_filter1d

    
class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted   = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, opt, track_id, n_init, max_age, feature=None, uv_map=None, bbox=None, detection_data=None, confidence=None, detection_id=None, dims=None, time=None):
        self.opt               = opt
        self.track_id          = track_id
        self.hits              = 1
        self.age               = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative            

        if(dims is not None):
            self.A_dim = dims[0]
            self.P_dim = dims[1]
            self.L_dim = dims[2]
        
        self.phalp_uv_map        = uv_map
        self.phalp_uv_map_       = [uv_map]
        self.phalp_uv_predicted  = copy.deepcopy(self.phalp_uv_map)
        self.phalp_uv_predicted_ = [copy.deepcopy(self.phalp_uv_map)]
        
        self.phalp_appe_features = []
        self.phalp_pose_features = []
        self.phalp_loca_features = []
        self.phalp_time_features = []
        self.phalp_bbox          = []
        self.phalp_detection_id  = []
        self.detection_data      = []
        self.confidence_c        = []
        if feature is not None:
            
            for i_ in range(self.opt.track_history):
                self.phalp_appe_features.append(feature[:self.A_dim])
                self.phalp_pose_features.append(feature[self.A_dim:self.A_dim+self.P_dim])
                self.phalp_loca_features.append(feature[self.A_dim+self.P_dim:])
                self.phalp_time_features.append(time)
                self.phalp_bbox.append(bbox)
                self.phalp_detection_id.append(detection_id)
                self.detection_data.append(detection_data)
                self.confidence_c.append(confidence[0])
                
        self._n_init    = n_init
        self._max_age   = max_age

        self.track_data = {
                            "xy"   : self.detection_data[-1]['xy'],
                            "bbox" : np.asarray(self.detection_data[-1]['bbox'], dtype=np.float),
                          }
        

        self.phalp_pose_predicted_ = []
        self.phalp_loca_predicted_ = []
        self.phalp_features_       = []
    
        
    def predict(self, phalp_tracker, increase_age=True):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        if(increase_age):
            self.age += 1
            self.time_since_update += 1
            
    def add_predicted(self, appe=None, pose=None, loca=None, uv=None):
        self.phalp_appe_predicted = copy.deepcopy(appe.numpy()) if(appe is not None) else copy.deepcopy(self.phalp_appe_features[-1])
        self.phalp_pose_predicted = copy.deepcopy(pose.numpy()) if(pose is not None) else copy.deepcopy(self.phalp_pose_features[-1])
        self.phalp_loca_predicted = copy.deepcopy(loca.numpy()) if(loca is not None) else copy.deepcopy(self.phalp_loca_features[-1])
        self.phalp_features       = np.concatenate((self.phalp_appe_predicted, self.phalp_pose_predicted, self.phalp_loca_predicted), axis=0)
        
        self.phalp_pose_predicted_.append(self.phalp_pose_predicted)
        if(len(self.phalp_pose_predicted_)>self.opt.n_init+1): self.phalp_pose_predicted_ = self.phalp_pose_predicted_[1:]
            
        self.phalp_loca_predicted_.append(self.phalp_loca_predicted)
        if(len(self.phalp_loca_predicted_)>self.opt.n_init+1): self.phalp_loca_predicted_ = self.phalp_loca_predicted_[1:]
        
        self.phalp_features_.append(self.phalp_features)
        if(len(self.phalp_features_)>self.opt.n_init+1): self.phalp_features_ = self.phalp_features_[1:]
        
    def update(self, detection, detection_id, shot):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        h = detection.tlwh[3]
        w = detection.tlwh[2]                 

        self.phalp_appe_features.append(detection.feature[:self.A_dim])
        self.phalp_appe_features = copy.deepcopy(self.phalp_appe_features[1:])
        self.phalp_pose_features.append(detection.feature[self.A_dim:self.A_dim+self.P_dim])
        self.phalp_pose_features = copy.deepcopy(self.phalp_pose_features[1:])
        self.phalp_loca_features.append(detection.feature[self.A_dim+self.P_dim:])
        self.phalp_loca_features = copy.deepcopy(self.phalp_loca_features[1:])
        if(shot==1): self.phalp_loca_features = [detection.feature[self.A_dim+self.P_dim:] for i in range(self.opt.track_history)]
        self.phalp_time_features.append(detection.time)
        self.phalp_time_features = copy.deepcopy(self.phalp_time_features[1:])
        self.phalp_bbox.append(detection.tlwh)
        self.phalp_bbox          = self.phalp_bbox[1:]
        
        self.confidence_c.append(detection.confidence_c)
        self.confidence_c        = self.confidence_c[1:]
        
        
        self.detection_data.append(detection.detection_data)
        self.detection_data      = self.detection_data[1:]
        self.phalp_detection_id.append(detection_id)
        


        self.phalp_uv_map                  = copy.deepcopy(detection.uv_map)
        self.phalp_uv_map_.append(copy.deepcopy(detection.uv_map))
        if(self.opt.render or "T" in self.opt.predict):
            mixing_alpha_                      = self.opt.alpha*(detection.confidence_c**2)
            ones_old                           = self.phalp_uv_predicted[3:, :, :]==1
            ones_new                           = self.phalp_uv_map[3:, :, :]==1
            ones_old                           = np.repeat(ones_old, 3, 0)
            ones_new                           = np.repeat(ones_new, 3, 0)
            ones_intersect                     = np.logical_and(ones_old, ones_new)
            ones_union                         = np.logical_or(ones_old, ones_new)
            good_old_ones                      = np.logical_and(np.logical_not(ones_intersect), ones_old)
            good_new_ones                      = np.logical_and(np.logical_not(ones_intersect), ones_new)
            new_rgb_map                        = np.zeros((3, 256, 256))
            new_mask_map                       = np.zeros((1, 256, 256))-1
            new_mask_map[ones_union[:1, :, :]] = 1.0
            new_rgb_map[ones_intersect]        = (1-mixing_alpha_)*self.phalp_uv_predicted[:3, :, :][ones_intersect] + mixing_alpha_*self.phalp_uv_map[:3, :, :][ones_intersect]
            new_rgb_map[good_old_ones]         = self.phalp_uv_predicted[:3, :, :][good_old_ones] 
            new_rgb_map[good_new_ones]         = self.phalp_uv_map[:3, :, :][good_new_ones] 
            self.phalp_uv_predicted            = np.concatenate((new_rgb_map, new_mask_map), 0)
            self.phalp_uv_predicted_.append(self.phalp_uv_predicted)
            if(len(self.phalp_uv_predicted_)>self.opt.n_init+1): self.phalp_uv_predicted_ = self.phalp_uv_predicted_[1:]
        else:
            self.phalp_uv_predicted            = self.phalp_uv_map
            
            
            
        self.track_data = {
                            "xy"   : detection.detection_data['xy'],
                            "bbox" : np.asarray(detection.detection_data['bbox'], dtype=np.float64)
                          }
        
        
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

        
    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    def smooth_bbox(self, bbox):
        kernel_size = 5
        sigma       = 3
        bbox        = np.array(bbox)
        smoothed    = np.array([signal.medfilt(param, kernel_size) for param in bbox.T]).T
        out         = np.array([gaussian_filter1d(traj, sigma) for traj in smoothed.T]).T
        return list(out)