import logging
logger = logging.getLogger('base')


def create_model(opt):
    from .model import DDPM as M
    m = M(opt)  # initialize a DDPM model with opts
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m