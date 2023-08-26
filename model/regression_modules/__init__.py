import sys
sys.path.append('/root/Improve-HPE-with-Diffusion')
from core.registry import MODEL_REGISTRY

def build_regressor(opt):
    regressor = MODEL_REGISTRY.get(opt['name'])(opt)

    return regressor