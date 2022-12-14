import os
import warnings
from typing import List, Optional, Tuple

import hydra
import pyrootutils
import submitit
from omegaconf import DictConfig, OmegaConf
from phalp import PHALP

warnings.filterwarnings('ignore')

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

@hydra.main(version_base="1.2", config_path=root / "configs", config_name="base.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main function for running the PHALP tracker."""
    
    phalp_tracker = PHALP(cfg)
    
    phalp_tracker.cuda()
    
    phalp_tracker.eval()
    
    phalp_tracker.track()

if __name__ == "__main__":
    main()