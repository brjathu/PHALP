
def get_tracker(cfg):
    from phalp.trackers.PHALP import PHALP
    """Get the tracker object from the config."""
    if cfg.base_tracker == "PHALP":
        return PHALP(cfg)
    elif cfg.base_tracker == "PHALP_v2":
        return PHALP_v2(cfg)
    elif cfg.base_tracker == "PHALP_v2_pymafx":
        return PHALP_v2_pymafx(cfg)
    else:
        raise ValueError("Unknown tracker")
