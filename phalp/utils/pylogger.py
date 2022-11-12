import logging


def get_pylogger(name=__name__):

    FORMAT = "%(message)s"
    logging.basicConfig(
        level="INFO", format=FORMAT, datefmt="[%X]")
    log = logging.getLogger("rich")
    
    return log
