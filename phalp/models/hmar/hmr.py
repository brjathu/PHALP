import warnings

from torch import nn

from phalp.utils import get_pylogger
from phalp.models.hmar.hmar import HMAR

warnings.filterwarnings('ignore')
log = get_pylogger(__name__)

class HMR2018Predictor(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        # Old HMAR stuff
        self.hmar_old = HMAR(cfg)
        self.smpl = self.hmar_old.smpl
        self.load_weights(cfg.hmr.hmar_path)

    def forward(self, x):
        return self.hmar_old(x)

    def load_weights(self, path):
        self.hmar_old.load_weights(path)

    # Other stuff from hmar
    def autoencoder_hmar(self, *args, **kwargs):
        return self.hmar_old.autoencoder_hmar(*args, **kwargs)

    def get_3d_parameters(self, *args, **kwargs):
        return self.hmar_old.get_3d_parameters(*args, **kwargs)