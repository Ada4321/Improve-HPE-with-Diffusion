import torch
import sys
sys.path.append('/root/Improve-HPE-with-Diffusion')
sys.path.append('/root/Improve-HPE-with-Diffusion/model/regression_modules')
from regressor import MODEL_REGISTRY


def build_regressor(opt):
    regressor = MODEL_REGISTRY.get(opt['name'])(**opt)
    # load ckpt
    checkpoint = torch.load(opt["ckpt_path"], map_location=lambda storage, loc: storage)
    #print(model.state_dict().keys())
    kvs = {".".join(k.split(".")[1:]):v for k,v in checkpoint["model_pos"].items()}
    regressor.load_state_dict(kvs, strict=True)
    return regressor