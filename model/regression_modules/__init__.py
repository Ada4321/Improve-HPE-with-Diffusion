import sys
sys.path.append('/root/Improve-HPE-with-Diffusion')
sys.path.append('/root/Improve-HPE-with-Diffusion/model/regression_modules')
#from core.registry import Registry
from backbone import BACKBONE_REGISTRY
from neck import NECK_REGISTRY
from regressor import MODEL_REGISTRY


def build_regressor(opt):
    # # build backbone
    # backbone = BACKBONE_REGISTRY.get(opt["backbone"])(opt)
    # # build neck
    # neck = NECK_REGISTRY.get(opt["neck"])(opt)
    # # build regressor
    # regressor = MODEL_REGISTRY.get(opt['name'])(backbone, neck)

    regressor = MODEL_REGISTRY.get(opt['name'])(**opt)
    return regressor