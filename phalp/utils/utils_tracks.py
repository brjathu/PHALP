import warnings
import numpy as np

warnings.filterwarnings("ignore")

def get_tracks(phalp_tracks):

    tracks_ids     = []
    tracks_dict    = {}
    for frame_ in phalp_tracks: tracks_ids += phalp_tracks[frame_]['tracked_ids']

    tracks_ids = list(set(tracks_ids))
    
    for track_id in tracks_ids:
        tracks_dict[track_id] = {}
        list_valid_frames     = []
        list_frame_names      = []
        for fid, frame_name in enumerate(phalp_tracks.keys()):
            list_tracks  = np.array(phalp_tracks[frame_name]['tid'])
            list_time   = np.array(phalp_tracks[frame_name]['tracked_time'])
            idx_        = np.where(list_tracks==track_id)[0]
            if(len(idx_)>=1 and list_time[idx_[0]]==0):
                time_       = list_time[idx_[0]]
                track_results = {
                    'track_id'   : track_id,
                    'frame_name' : frame_name,
                    'time'       : time_,
                    'bbox'       : phalp_tracks[frame_name]['bbox'][idx_[0]],
                    'center'     : phalp_tracks[frame_name]['center'][idx_[0]],
                    'scale'      : phalp_tracks[frame_name]['scale'][idx_[0]],
                    'conf'       : phalp_tracks[frame_name]['conf'][idx_[0]],
                    'size'       : phalp_tracks[frame_name]['size'][idx_[0]],
                    'smpl'       : phalp_tracks[frame_name]['smpl'][idx_[0]],
                    'camera'     : phalp_tracks[frame_name]['camera'][idx_[0]],
                    'camera_bbox': phalp_tracks[frame_name]['camera_bbox'][idx_[0]],
                    'img_path'   : phalp_tracks[frame_name]['img_path'][idx_[0]],
                    'img_name'   : phalp_tracks[frame_name]['img_name'][idx_[0]],
                    '3d_joints'  : phalp_tracks[frame_name]['3d_joints'][idx_[0]],
                    
                    'has_detection' : True,
                    'fid'           : fid,
                    'frame_path'    : phalp_tracks[frame_name]['frame_path'],
                }
                
            
                tracks_dict[track_id][frame_name] = track_results
                list_valid_frames.append(1)
            else:
                track_results = {
                    'track_id'      : -1,
                    'frame_name'    : frame_name,
                    'has_detection' : False,
                    'fid'           : fid,
                    'time'          : -1,
                    'bbox'          : None,
                    'size'          : None,
                    'frame_path'    : phalp_tracks[frame_name]['frame_path'],
                }
                tracks_dict[track_id][frame_name] = track_results
                list_valid_frames.append(0)
            
            list_frame_names.append(frame_name)
        
        list_valid_frames2 = np.array(list_valid_frames)
        loc_ = np.where(list_valid_frames2==1)
        s_ = np.min(loc_)
        e_ = np.max(loc_)

        for i, fname in enumerate(list_frame_names):
            if(i<s_ or i>e_):
                del tracks_dict[track_id][fname]

    return tracks_dict



def create_fast_tracklets(data):
    list_of_frames = list(data.keys())
    frame_length   = len(list_of_frames)
    
    array_fid              = np.zeros((frame_length, 1, 1))-1
    
    array_pose_shape       = np.zeros((frame_length, 1, 229))
    array_smpl             = []
    array_3d_joints        = np.zeros((frame_length, 1, 45, 3))
    array_camera           = np.zeros((frame_length, 1, 3))
    array_camera_bbox      = np.zeros((frame_length, 1, 3))
    
    array_has_detection    = np.zeros((frame_length, 1, 1))
    
    array_frame_name = []
    array_frame_bbox = []
    array_frame_size = []
    array_frame_conf = []

    for fid, frame_name in enumerate(list_of_frames):
        
        frame_data = data[frame_name]
        array_fid[fid, 0, 0] = frame_data["fid"]
        
        if(frame_data['has_detection']):
            has_gt          = 0
            gloabl_ori_     = frame_data['smpl']['global_orient'].reshape(1, -1)
            body_pose_      = frame_data['smpl']['body_pose'].reshape(1, -1)
            betas_          = frame_data['smpl']['betas'].reshape(1, -1)
            location_       = frame_data['camera'].reshape(1, -1)
            location_[0, 2] = location_[0, 2]/200.0
            array_smpl.append(frame_data['smpl'])
            pose_shape_     = np.hstack([gloabl_ori_, body_pose_, betas_, location_]) 
            
            joints_3d = frame_data['3d_joints']
            camera    = frame_data['camera']
            camera_bbox   = frame_data['camera_bbox']
            
            array_pose_shape[fid, 0, :] = pose_shape_
            array_3d_joints[fid, 0, :] = joints_3d
            array_camera[fid, 0, :] = camera
            array_camera_bbox[fid, 0, :] = camera_bbox
            array_has_detection[fid, 0, 0] = 1

            h, w = frame_data['size']
            nis  = max(h, w)
            top, left = (nis - h)//2, (nis - w)//2,
        else:
            array_smpl.append(-1)        
        array_frame_name.append(frame_data['frame_name'])
        array_frame_size.append(np.array(frame_data['size']) if frame_data['has_detection'] else np.array([0, 0]))
        array_frame_bbox.append(frame_data['bbox'] if frame_data['has_detection'] else np.array([0.0, 0.0, 0.0, 0.0]))
        array_frame_conf.append(frame_data['conf'] if frame_data['has_detection'] else 0)
        
    
    return  {
                'fid'                : array_fid,
                'pose_shape'         : array_pose_shape, 
                '3d_joints'          : array_3d_joints, 
                'camera'             : array_camera, 
                'camera_bbox'        : array_camera_bbox, 
                'has_detection'      : array_has_detection, 
                'frame_name'         : array_frame_name,
                'frame_size'         : array_frame_size,
                'frame_bbox'         : array_frame_bbox,
                'frame_conf'         : array_frame_conf,
                'smpl'               : array_smpl,
        }

