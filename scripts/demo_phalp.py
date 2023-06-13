import warnings
from dataclasses import dataclass
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from phalp.configs.base import FullConfig
from phalp.trackers.PHALP import PHALP
from phalp.utils import get_pylogger

warnings.filterwarnings('ignore')
log = get_pylogger(__name__)

@dataclass
class DemoConfig(FullConfig):
    # override default config if needed
    pass
cs = ConfigStore.instance()
cs.store(name="config", node=DemoConfig)

@hydra.main(version_base="1.2", config_name="config")
def main(cfg: DictConfig) -> Optional[float]:
    """Main function for running the PHALP tracker."""

    phalp_tracker = PHALP(cfg)

    phalp_tracker.track()

if __name__ == "__main__":
    main()
