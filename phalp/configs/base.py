from dataclasses import dataclass, field
import os
from typing import Dict, Optional

import hydra
from omegaconf import MISSING

CACHE_DIR = os.path.join(os.environ.get("HOME"), ".cache")  # None if the variable does not exist

@dataclass
class VideoConfig:
    source: str = MISSING
    output_dir: str = 'outputs/'
    extract_video: bool = True
    base_path: Optional[str] = None
    start_frame: int = -1
    end_frame: int = 1300
    useffmpeg: bool = True

    # this will be used if extract_video=False
    start_time: str = '0s'
    end_time: str = '10s'


@dataclass
class PHALPConfig:
    predict: str = 'TPL'
    pose_distance: str = 'smpl'
    distance_type: str = 'EQ_019'
    alpha: float = 0.1
    low_th_c: float = 0.8
    hungarian_th: float = 100.0
    track_history: int = 7
    max_age_track: int = 50
    n_init: int = 5
    encode_type: str = '4c'
    past_lookback: int = 1
    detector: str = 'vitdet'
    shot: int = 0
    start_frame: int = -1
    end_frame: int = 10

    small_w: int = 50
    small_h: int = 100


@dataclass
class PosePredictorConfig:
    config_path: str = f"{CACHE_DIR}/phalp/weights/pose_predictor.yaml"
    weights_path: str = f"{CACHE_DIR}/phalp/weights/pose_predictor.pth"
    mean_std: str = f"{CACHE_DIR}/phalp/3D/mean_std.npy"

@dataclass
class HMRConfig:
    hmar_path: str = f"{CACHE_DIR}/phalp/weights/hmar_v2_weights.pth"

@dataclass
class RenderConfig:
    enable: bool = True
    type: str = 'HUMAN_MESH' # options: HUMAN_MESH, HUMAN_MASK, HUMAN_BBOX
    up_scale: int = 2
    res: int = 256
    side_view_each: bool = False
    metallicfactor: float = 0.0
    roughnessfactor: float = 0.7
    colors: str = "phalp"
    head_mask: bool = False
    head_mask_path: str = f"{CACHE_DIR}/phalp/3D/head_faces.npy"
    output_resolution: int = 1440
    fps: int = 30
    blur_faces: bool = False
    show_keypoints: bool = False

@dataclass
class PostProcessConfig:
    apply_smoothing: bool = True
    phalp_pkl_path: str = '_OUT/videos_v0'

@dataclass
class SMPLConfig:
    MODEL_PATH: str = f"{CACHE_DIR}/phalp/3D/models/smpl/"
    GENDER: str = 'neutral'
    MODEL_TYPE: str = 'smpl'
    NUM_BODY_JOINTS: int = 23
    JOINT_REGRESSOR_EXTRA: str = f"{CACHE_DIR}/phalp/3D/SMPL_to_J19.pkl"
    TEXTURE: str = f"{CACHE_DIR}/phalp/3D/texture.npz"

# Config for HMAR
@dataclass
class SMPLHeadConfig:
    TYPE: str = 'basic'
    POOL: str = 'max'
    SMPL_MEAN_PARAMS: str = f"{CACHE_DIR}/phalp/3D/smpl_mean_params.npz"
    IN_CHANNELS: int = 2048

@dataclass
class BackboneConfig:
    TYPE: str = 'resnet'
    NUM_LAYERS: int = 50
    MASK_TYPE: str = 'feat'

@dataclass
class TransformerConfig:
    HEADS: int = 1
    LAYERS: int = 1
    BOX_FEATS: int = 6

@dataclass
class ModelConfig:
    IMAGE_SIZE: int = 256
    SMPL_HEAD: SMPLHeadConfig = field(default_factory=SMPLHeadConfig)
    BACKBONE: BackboneConfig = field(default_factory=BackboneConfig)
    TRANSFORMER: TransformerConfig = field(default_factory=TransformerConfig)
    pose_transformer_size: int = 2048

@dataclass
class ExtraConfig:
    FOCAL_LENGTH: int = 5000

@dataclass
class FullConfig:
    seed: int = 42
    track_dataset: str = "demo"
    device: str = "cuda"
    base_tracker: str = "PHALP"
    train: bool = False
    debug: bool = False
    use_gt: bool = False
    overwrite: bool = True
    task_id: int = -1
    num_tasks: int = 100
    verbose: bool = False
    detect_shots: bool = False
    video_seq: Optional[str] = None

    # Fields
    video: VideoConfig = field(default_factory=VideoConfig)
    phalp: PHALPConfig = field(default_factory=PHALPConfig)
    pose_predictor: PosePredictorConfig = field(default_factory=PosePredictorConfig)
    hmr: HMRConfig = field(default_factory=HMRConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    post_process: PostProcessConfig = field(default_factory=PostProcessConfig)
    SMPL: SMPLConfig = field(default_factory=SMPLConfig)
    MODEL: ModelConfig = field(default_factory=ModelConfig)
    EXTRA: ExtraConfig = field(default_factory=ExtraConfig)

    # tmp configs
    hmr_type: str = "hmr2018"

    # hydra configs
    hydra: Dict = field(default_factory = lambda: dict(
                            mode=hydra.types.RunMode.RUN,
                            run=dict(dir="${video.output_dir}"),
                        )
                    )
