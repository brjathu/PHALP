import math

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from phalp.models.predictor.smpl_head import SMPLHead
from omegaconf import OmegaConf
from torch import nn

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe          = torch.zeros(length, d_model)
    position    = torch.arange(0, length).unsqueeze(1)
    div_term    = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn   = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class ProjectInOut(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn

        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim    = dim_head *  heads
        project_out  = not (heads == 1 and dim_head == dim)

        self.heads   = heads
        self.scale   = dim_head ** -0.5
        self.attend  = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv  = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out  = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask_all):
        qkv          = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v      = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots         = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        masks_np     = mask_all[0]
        masks_bert   = mask_all[1]
        BS           = masks_np.shape[0]
        masks_np     = masks_np.view(BS, -1)
        masks_bert   = masks_bert.view(BS, -1)
        
        masks_np_    = rearrange(masks_np, 'b i -> b () i ()') * rearrange(masks_np, 'b j -> b () () j')
        masks_np_    = masks_np_.repeat(1, self.heads, 1, 1)
        
        masks_bert_  = rearrange(masks_bert, 'b i -> b () () i')
        masks_bert_  = masks_bert_.repeat(1, self.heads, masks_bert_.shape[-1], 1)
                
        dots[masks_np_==0]   = -1e3
        dots[masks_bert_==1] = -1e3
        
        del masks_np, masks_np_, masks_bert, masks_bert_
        
        attn    = self.attend(dots)
        attn    = self.dropout(attn)

        out     = torch.matmul(attn, v)
        out     = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'
    
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., drop_path = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, mask_np):
        for attn, ff in self.layers:
            x_          = attn(x, mask_all=mask_np) 
            x           = x + self.drop_path(x_)
            x           = x + self.drop_path(ff(x))
            
        return x

class lart_transformer(nn.Module):
    def __init__(self, opt, opt2, dim, depth, heads, mlp_dim, dim_head = 64, dropout = 0., emb_dropout = 0., droppath = 0., device=None):
        super().__init__()
        self.cfg  = opt
        self.cfg2 = opt2
        self.dim  = dim
        self.device = device
        self.mask_token = nn.Parameter(torch.randn(self.dim,))
        self.class_token = nn.Parameter(torch.randn(1, 1, self.dim))
        
        self.pos_embedding = nn.Parameter(positionalencoding1d(self.dim, 10000))#.to(self.device)
        self.pos_embedding_learned1 = nn.Parameter(torch.randn(1, self.cfg.frame_length, self.dim))
        self.pos_embedding_learned2 = nn.Parameter(torch.randn(1, self.cfg.frame_length, self.dim))
        self.register_buffer('pe', self.pos_embedding)
        
        self.transformer    = Transformer(self.dim, depth, heads, dim_head, mlp_dim, dropout, drop_path = droppath)
        self.transformer1       = Transformer(self.dim, depth, heads, dim_head, mlp_dim, dropout, drop_path = droppath)
        self.transformer2       = Transformer(self.dim, 1, heads, dim_head, mlp_dim, dropout, drop_path = droppath)

        pad                 = self.cfg.transformer.conv.pad
        stride              = self.cfg.transformer.conv.stride
        kernel              = stride + 2 * pad
        self.conv_en        = nn.Conv1d(self.dim, self.dim, kernel_size=kernel, stride=stride, padding=pad)
        self.conv_de        = nn.ConvTranspose1d(self.dim, self.dim, kernel_size=kernel, stride=stride, padding=pad)

        # Pose shape encoder for encoding pose shape features, used by default
        self.pose_shape_encoder     = nn.Sequential(
                                            nn.Linear(self.cfg.extra_feat.pose_shape.dim, self.cfg.extra_feat.pose_shape.mid_dim), 
                                            nn.ReLU(), 
                                            nn.Linear(self.cfg.extra_feat.pose_shape.mid_dim, self.cfg.extra_feat.pose_shape.en_dim),
                                        )
        
        # SMPL head for predicting SMPL parameters
        self.smpl_head              = nn.ModuleList([SMPLHead(self.cfg, self.cfg2) for _ in range(self.cfg.num_smpl_heads)])
        
        # Location head for predicting 3D location of the person
        self.loca_head              = nn.ModuleList([nn.Sequential(
                                            nn.Linear(self.cfg.in_feat, self.cfg.in_feat), 
                                            nn.ReLU(), 
                                            nn.Linear(self.cfg.in_feat, self.cfg.in_feat),
                                            nn.ReLU(),         
                                            nn.Linear(self.cfg.in_feat, 3)
                                        ) for _ in range(self.cfg.num_smpl_heads)])
        
        # Action head for predicting action class in AVA dataset labels  
        ava_action_classes          = self.cfg.ava.num_action_classes if not(self.cfg.ava.predict_valid) else self.cfg.ava.num_valid_action_classes
        self.action_head_ava        = nn.ModuleList([nn.Sequential(    
                                            nn.Linear(self.cfg.in_feat, ava_action_classes),
                                        ) for _ in range(self.cfg.num_smpl_heads)])

    def bert_mask(self, data, mask_type):
        if(mask_type=="random"):
            has_detection  = data['has_detection']==1
            mask_detection = data['mask_detection']
            for i in range(data['has_detection'].shape[0]):
                indexes        = has_detection[i].nonzero()
                indexes_mask   = indexes[torch.randperm(indexes.shape[0])[:int(indexes.shape[0]*self.cfg.mask_ratio)]]
                mask_detection[i, indexes_mask[:, 0], indexes_mask[:, 1], indexes_mask[:, 2]] = 1.0

        elif(mask_type=="zero"):
            has_detection  = data['has_detection']==0
            mask_detection = data['mask_detection']
            indexes_mask   = has_detection.nonzero()
            mask_detection[indexes_mask[:, 0], indexes_mask[:, 1], indexes_mask[:, 2], :] = 1.0
            has_detection = has_detection*0 + 1.0
        
        else:
            raise NotImplementedError
            
        return data, has_detection, mask_detection

    def forward(self, data, mask_type="random"):
        
        # prepare the input data and masking
        data, has_detection, mask_detection = self.bert_mask(data, mask_type)

        # encode the input pose tokens
        pose_   = data['pose_shape'].float()
        pose_en = self.pose_shape_encoder(pose_)
        x       = pose_en
        
        # mask the input tokens
        x[mask_detection[:, :, :, 0]==1] = self.mask_token

        BS, T, P, dim = x.size()
        x = x.view(BS, T*P, dim)

        # adding 2D posistion embedding
        # x = x + self.pos_embedding[None, :, :self.cfg.frame_length, :self.cfg.max_people].reshape(1, dim, self.cfg.frame_length*self.cfg.max_people).permute(0, 2, 1)
        
        x                = x + self.pos_embedding_learned1
        x    = self.transformer1(x, [has_detection, mask_detection])

        x = x.transpose(1, 2)
        x = self.conv_en(x)
        x = self.conv_de(x)
        x = x.transpose(1, 2)
        x = x.contiguous()

        x                = x + self.pos_embedding_learned2
        has_detection    = has_detection*0 + 1
        mask_detection   = mask_detection*0
        x    = self.transformer2(x, [has_detection, mask_detection])
        x = torch.concat([self.class_token.repeat(BS, self.cfg.max_people, 1), x], dim=1)
        

        return x, 0


class Pose_transformer_v2(nn.Module):
    
    def __init__(self, cfg, phalp_tracker):
        super(Pose_transformer_v2, self).__init__()
        
        self.phalp_cfg = cfg

        # load a config file
        self.cfg = OmegaConf.load(self.phalp_cfg.pose_predictor.config_path).configs
        self.cfg.max_people = 1
        self.encoder      = lart_transformer(   
                                opt         = self.cfg, 
                                opt2        = self.phalp_cfg,
                                dim         = self.cfg.in_feat,
                                depth       = self.cfg.transformer.depth,
                                heads       = self.cfg.transformer.heads,
                                mlp_dim     = self.cfg.transformer.mlp_dim,
                                dim_head    = self.cfg.transformer.dim_head,
                                dropout     = self.cfg.transformer.dropout,
                                emb_dropout = self.cfg.transformer.emb_dropout,
                                droppath    = self.cfg.transformer.droppath,
                                )
        
        self.mean_, self.std_ = np.load(self.phalp_cfg.pose_predictor.mean_std, allow_pickle=True)
        self.mean_            = np.concatenate((self.mean_, np.zeros((1, 229-self.mean_.shape[1]))), axis=1)
        self.std_             = np.concatenate((self.std_, np.ones((1, 229-self.std_.shape[1]))), axis=1)
        self.mean_, self.std_ = torch.tensor(self.mean_), torch.tensor(self.std_)
        self.mean_, self.std_ = self.mean_.float(), self.std_.float()
        self.mean_, self.std_ = self.mean_.unsqueeze(0), self.std_.unsqueeze(0)   
        self.register_buffer('mean', self.mean_)
        self.register_buffer('std', self.std_)
        
        self.smpl = phalp_tracker.HMAR.smpl
            
    def load_weights(self, path):
        # import ipdb; ipdb.set_trace()
        checkpoint_file = torch.load(path)
        checkpoint_file_filtered = {k[8:]: v for k, v in checkpoint_file['state_dict'].items()} # remove "encoder." from keys
        self.encoder.load_state_dict(checkpoint_file_filtered, strict=False)
    
    def readout_pose(self, output):
        
        # return predicted gt pose, betas and location
        BS = output.shape[0]
        FL = output.shape[1]
        pose_tokens      = output.contiguous()
        pose_tokens_     = rearrange(pose_tokens, 'b tp dim -> (b tp) dim')
        
        pred_smpl_params = [self.encoder.smpl_head[i](pose_tokens_)[0] for i in range(self.cfg.num_smpl_heads)]
        pred_cam         = [self.encoder.loca_head[i](pose_tokens) for i in range(self.cfg.num_smpl_heads)]
        pred_ava         = [self.encoder.action_head_ava[i](pose_tokens) for i in range(self.cfg.num_smpl_heads)]
        
        pred_cam         = torch.stack(pred_cam, dim=0)[0]
        pred_cam         = rearrange(pred_cam, 'b (t p) dim -> b t p dim', b=BS, t=FL ,p=self.cfg.max_people) # (BS, T, P, 3)        
        

        global_orient    = rearrange(pred_smpl_params[0]['global_orient'], '(b t p) x y z -> b t p (x y z)', b=BS, t=FL ,p=self.cfg.max_people, x=1, y=3, z=3) # (BS, T, P, 9)
        body_pose        = rearrange(pred_smpl_params[0]['body_pose'], '(b t p) x y z -> b t p (x y z)', b=BS, t=FL ,p=self.cfg.max_people, x=23, y=3, z=3) # (BS, T, P, 207)
        betas            = rearrange(pred_smpl_params[0]['betas'], '(b t p) z -> b t p z', b=BS, t=FL ,p=self.cfg.max_people, z=10) # (BS, T, P, 10)
        pose_vector      = torch.cat((global_orient, body_pose, betas, pred_cam), dim=-1) # (BS, T, P, 229)
        
        pred_ava         = torch.stack(pred_ava, dim=0)[0]
        pred_ava         = rearrange(pred_ava, 'b (t p) dim -> b t p dim', b=BS, t=FL ,p=self.cfg.max_people) # (BS, T, P, 60)        

        # TODO: apply moving average for pridictions

        smpl_outputs = {
            'pose_camera'      : pose_vector,
            'camera'           : pred_cam,
            'ava_action'       : pred_ava,
        }
            
        return smpl_outputs
            
    def predict_next(self, en_pose, en_data, en_time, time_to_predict):
        
        """encoder takes keys : 
                    pose_shape (bs, self.cfg.frame_length, 229)
                    has_detection (bs, self.cfg.frame_length, 1), 1 if there is a detection, 0 otherwise
                    mask_detection (bs, self.cfg.frame_length, 1)*0       
        """
        
        # set number of people to one 
        n_p = 1
        pose_shape_ = torch.zeros(en_pose.shape[0], self.cfg.frame_length, n_p, 229)
        has_detection_ = torch.zeros(en_pose.shape[0], self.cfg.frame_length, n_p, 1)
        mask_detection_ = torch.zeros(en_pose.shape[0], self.cfg.frame_length, n_p, 1)
        
        # loop thorugh each person and construct the input data
        t_end = []
        for p_ in range(en_time.shape[0]):
            t_min = en_time[p_, 0].min()
            # loop through time 
            for t_ in range(en_time.shape[1]):
                # get the time from start.
                t = min(en_time[p_, t_] - t_min, self.cfg.frame_length - 1)
                
                # get the pose
                pose_shape_[p_, t, 0, :] = en_pose[p_, t_, :]
                
                # get the mask
                has_detection_[p_, t, 0, :] = 1
            t_end.append(t.item())
            
        input_data = {
            "pose_shape" : (pose_shape_ - self.mean_[:, :, None, :]) / (self.std_[:, :, None, :] + 1e-10),
            "has_detection" : has_detection_,
            "mask_detection" : mask_detection_
        }
        
        # place all the data in cuda
        input_data = {k: v.cuda() for k, v in input_data.items()}

        # single forward pass
        output, _ = self.encoder(input_data, self.cfg.mask_type_test)
        decoded_output = self.readout_pose(output[:, self.cfg.max_people:, :])
        
        assert len(t_end) == len(time_to_predict)
        t_end += time_to_predict + 1
        
        predicted_pose_camera_at_t = []
        for i in range(en_time.shape[0]): 
            t_x = min(t_end[i], self.cfg.frame_length-1)
            predicted_pose_camera_at_t.append(decoded_output['pose_camera'][:, t_x, 0, :])
        predicted_pose_camera_at_t = torch.stack(predicted_pose_camera_at_t, dim=0)[0]
        
        return predicted_pose_camera_at_t
    
    def add_slowfast_features(self, fast_track):
        # add slowfast features to the fast track
        from slowfast.utils.parser import load_config, parse_args
        from slowfast.config.defaults import assert_and_infer_cfg
        from slowfast.visualization.predictor import ActionPredictor, Predictor
        from phalp.models.predictor.wrapper_pyslowfast import SlowFastWrapper

        device = 'cuda'
        path_to_config = "/private/home/jathushan/3D/slowfast/configs/AVA/MViT-L-312_masked.yaml"
        center_crop = False
        if("MViT" in path_to_config): 
            center_crop = True

        self.cfg.opts = None
        cfg = load_config(self.cfg, path_to_config)
        cfg = assert_and_infer_cfg(cfg)

        video_model    = Predictor(cfg=cfg, gpu_id=None)
        seq_length     = cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE

        list_of_frames = fast_track['frame_name']
        list_of_bbox   = fast_track['frame_bbox']
        list_of_fids   = fast_track['fid']
        fast_track['mvit_emb'] = []
        fast_track['action_emb'] = []

        NUM_STEPS        = 6 # 5Hz
        NUM_FRAMES       = seq_length
        list_iter        = list(range(len(list_of_frames)//NUM_STEPS + 1))

        for t_, time_stamp in enumerate(list_iter):    

            start_      = time_stamp * NUM_STEPS
            end_        = (time_stamp + 1) * NUM_STEPS if (time_stamp + 1) * NUM_STEPS < len(list_of_frames) else len(list_of_frames)
            time_stamp_ = list_of_frames[start_:end_]
            if(len(time_stamp_)==0): continue

            mid_        = (start_ + end_)//2
            mid_frame   = list_of_frames[mid_]
            mid_bbox    = list_of_bbox[mid_]
            mid_fid     = list_of_fids[mid_]

            list_of_all_frames = []
            for i in range(-NUM_FRAMES//2,NUM_FRAMES//2 + 1):
                if(mid_ + i < 0):
                    frame_id = 0
                elif(mid_ + i >= len(list_of_frames)):
                    frame_id = len(list_of_frames) - 1
                else:
                    frame_id = mid_ + i
                list_of_all_frames.append(list_of_frames[frame_id])


            mid_bbox_   = mid_bbox.reshape(1, 4).astype(np.int32)
            mid_bbox_   = np.concatenate([mid_bbox_[:, :2], mid_bbox_[:, :2] + mid_bbox_[:, 2:4]], 1)
            # img1 = cv2.imread(mid_frame)
            # img1 = cv2.rectangle(img1, (mid_bbox_[0, 0], mid_bbox_[0, 1]), (mid_bbox_[0, 2], mid_bbox_[0, 3]), (0, 255, 0), 2)
            # cv2.imwrite("test.png", img1)
            with torch.no_grad():
                task_      = SlowFastWrapper(t_, cfg, list_of_all_frames, mid_bbox_, video_model, center_crop=center_crop)
                preds      = task_.action_preds[0]
                feats      = task_.action_preds[1]
                preds      = preds.cpu().numpy()
                feats      = feats.cpu().numpy()

            for frame_ in time_stamp_:
                fast_track['mvit_emb'].append(feats)
                fast_track['action_emb'].append(preds)
        
        assert len(fast_track['mvit_emb']) == len(fast_track['frame_name'])
        assert len(fast_track['action_emb']) == len(fast_track['frame_name'])
        fast_track['mvit_emb'] = np.array(fast_track['mvit_emb'])
        fast_track['action_emb'] = np.array(fast_track['action_emb'])
        
        return fast_track

    def smooth_tracks(self, fast_track, moving_window=False, step=1, window=20):
        
        if("mvit" in self.cfg.extra_feat.enable):
            fast_track = self.add_slowfast_features(fast_track)

        # set number of people to one 
        n_p = 1
        fl  = fast_track['pose_shape'].shape[0]

        pose_shape_all = torch.zeros(1, fl, n_p, 229)
        has_detection_all = torch.zeros(1, fl, n_p, 1)
        mask_detection_all = torch.zeros(1, fl, n_p, 1)

        if("mvit" in self.cfg.extra_feat.enable):
            mvit_feat_all = fast_track['mvit_emb'][None, :, :,]
        
        if("joints_3D" in self.cfg.extra_feat.enable):
            joints_ = fast_track['3d_joints'][:, :, :, :]
            camera_ = fast_track['camera'][:, None, :, :]
            joints_3d_all = joints_ + camera_
            joints_3d_all = joints_3d_all.reshape(1, fl, n_p, 135)

        for t_ in range(fast_track['pose_shape'].shape[0]):
            pose_shape_all[0, t_, 0, :] = torch.tensor(fast_track['pose_shape'][t_])
            has_detection_all[0, t_, 0, :] = 1
            mask_detection_all[0, t_, 0, :] = 1.0 - torch.tensor(fast_track['has_detection'][t_, 0])

        S_ = 0
        STEP_ = step
        WINDOW_ = window
        w_steps = range(S_, S_+fl, STEP_)
        assert 2*WINDOW_ + STEP_ < self.cfg.frame_length
        STORE_OUTPUT_ = torch.zeros(1, fl, self.cfg.in_feat)

        for w_ in w_steps:

            pose_shape_ = torch.zeros(1, self.cfg.frame_length, n_p, 229)
            has_detection_ = torch.zeros(1, self.cfg.frame_length, n_p, 1)
            mask_detection_ = torch.zeros(1, self.cfg.frame_length, n_p, 1)

            start_ = w_ - WINDOW_ if (w_ - WINDOW_>0) else 0
            end_ = w_ + STEP_ + WINDOW_ if (w_ + STEP_ + WINDOW_<=fl) else fl

            pose_shape_[:, :end_-start_, :, :] = pose_shape_all[:, start_:end_, :, :]
            has_detection_[:, :end_-start_, :, :] = has_detection_all[:, start_:end_, :, :]
            mask_detection_[:, :end_-start_, :, :] = mask_detection_all[:, start_:end_, :, :]

            input_data = {
                "pose_shape" : (pose_shape_ - self.mean_[0, :, None, :]) / (self.std_[0, :, None, :] + 1e-10),
                "has_detection" : has_detection_,
                "mask_detection" : mask_detection_
            }
            
            # add other features if enables:
            if("joints_3D" in self.cfg.extra_feat.enable):
                joints_ = torch.zeros(1, self.cfg.frame_length, n_p, 135)
                joints_[:, :end_-start_, :, :] = torch.tensor(joints_3d_all[:, start_:end_, :, :])
                input_data["joints_3D"] = joints_

            if("mvit" in self.cfg.extra_feat.enable):
                mvit_ = torch.zeros(1, self.cfg.frame_length, n_p, 1152)
                mvit_[:, :end_-start_, :, :] = torch.tensor(mvit_feat_all[:, start_:end_, :, :])
                input_data["mvit_emb"] = mvit_

            input_data = {k: v.cuda() for k, v in input_data.items()}

            output, _ = self.encoder(input_data, self.cfg.mask_type_test)
            output = output[:, self.cfg.max_people:, :]

            
            if(w_+STEP_<fl):
                if(w_<=WINDOW_):
                    STORE_OUTPUT_[:,  w_:w_+STEP_, :] = output[:,  w_:w_+STEP_, :]
                else:
                    STORE_OUTPUT_[:,  w_:w_+STEP_, :] = output[:,  WINDOW_:WINDOW_+STEP_, :]
            else:
                if(w_<=WINDOW_):
                    STORE_OUTPUT_[:,  w_:fl, :] = output[:,  w_:fl, :]
                else:
                    STORE_OUTPUT_[:,  w_:fl, :] = output[:,  WINDOW_:WINDOW_+(fl-w_), :]

        decoded_output = self.readout_pose(STORE_OUTPUT_.cuda())

        fast_track['pose_shape'] = decoded_output['pose_camera'][0, :fast_track['pose_shape'].shape[0], :, :]
        fast_track['cam_smoothed'] = decoded_output['camera'][0, :fast_track['pose_shape'].shape[0], :, :]
        fast_track['ava_action'] = decoded_output['ava_action'][0, :fast_track['pose_shape'].shape[0], :, :]
        
        return fast_track