import logging

logging.getLogger('fvcore.common.checkpoint').setLevel(logging.WARNING)
logging.getLogger('iopath.common.file_io').setLevel(logging.WARNING)
logging.getLogger('detectron2.data.dataset_mapper').setLevel(logging.WARNING)

def get_pylogger(name=__name__):

    FORMAT = "%(message)s"
    logging.basicConfig(
        level="INFO", format=FORMAT, datefmt="[%X]")
    log = logging.getLogger("rich")
    
    return log
