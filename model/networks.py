import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import modules
logger = logging.getLogger('base')
import sys
sys.path.append('/home/zhuhe/Improve-HPE-with-Diffusion-7.22/Improve-HPE-with-Diffusion/model')

from diffusion_modules.diffusion import GaussianDiffusion
from diffusion_modules.denoise_transformer import DenoiseTransformer
from regression_modules.regressor import Regressor

from torch.nn.parallel import DistributedDataParallel as DDP

####################
# initialize
####################

def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))

####################
# define network
####################

# Generator
def define_G(opt):
    model_opt = opt['model']
    model_opt['regressor']['preset'] = opt['data_preset']
    # if model_opt['which_model_G'] == 'ddpm':
    #     from .ddpm_modules import diffusion, unet
    # elif model_opt['which_model_G'] == 'sr3':
    #     from .sr3_modules import diffusion, unet
    
    regressor = Regressor(model_opt['regressor'])
    denoiser = DenoiseTransformer(model_opt['denoise_transformer'])
    
    netG = GaussianDiffusion(
        regressor,
        denoiser,
        #loss_type=opt['loss']['type'],
        loss_opt=opt['loss'],
        condition_on_preds=model_opt['diffusion']['condition_on_preds'],
    )
    # if opt['phase'] == 'train':
        # init_weights(netG, init_type='kaiming', scale=0.1)
        # init_weights(netG, init_type='orthogonal')
    if opt['gpu_ids']:
        assert torch.cuda.is_available()
        if opt['distributed']:
            # netG = nn.DataParallel(netG)
            netG = netG.to(opt['current_id'])
            netG = DDP(netG, device_ids=[opt['current_id']])
        else:
            netG = netG.to('cuda')

    return netG  # a nn.Module or DataParallel