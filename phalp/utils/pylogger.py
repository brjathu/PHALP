import logging

from rich.logging import RichHandler

logging.getLogger('fvcore.common.checkpoint').setLevel(logging.WARNING)
logging.getLogger('iopath.common.file_io').setLevel(logging.WARNING)
logging.getLogger('detectron2.data.dataset_mapper').setLevel(logging.WARNING)

def get_pylogger(name=__name__):
    """Get a logger with a custom format."""
    
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%m/%d %H:%M:%S]",
        handlers=[RichHandler()]
    )
    root_logger = logging.getLogger(name)
    
    return root_logger
