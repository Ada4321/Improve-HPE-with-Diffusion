import sys
sys.path.append('/root/Improve-HPE-with-Diffusion')
sys.path.append('/root/Improve-HPE-with-Diffusion/model/regression_modules')
#from core.registry import Registry
from regressor import MODEL_REGISTRY

def build_regressor(opt):
    regressor = MODEL_REGISTRY.get(opt['name'])(opt)

    return regressor