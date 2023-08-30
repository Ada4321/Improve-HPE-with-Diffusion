import torch.nn as nn
import sys
sys.path.append('/root/Improve-HPE-with-Diffusion')
from core.registry import Registry

LOSS_REGISTRY = Registry('loss')


# basic loss functions for regression and diffusion
@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    def __init__(self, dev) -> None:
        super().__init__()
        self.loss_fn = nn.L1Loss(reduction='sum').to(dev)
    def forward(self, pred, tgt):
        return self.loss_fn(pred, tgt)

@LOSS_REGISTRY.register()
class L1LossBatchAvg(nn.Module):
    def __init__(self, dev) -> None:
        super().__init__()
        self.loss_fn = nn.L1Loss(reduction='sum').to(dev)
    def forward(self, pred, tgt):
        return self.loss_fn(pred, tgt) / len(pred)

@LOSS_REGISTRY.register()
class L2Loss(nn.Module):
    def __init__(self, dev) -> None:
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction='sum').to(dev)
    def forward(self, pred, tgt):
        return self.loss_fn(pred, tgt)

@LOSS_REGISTRY.register()
class L2LossBatchAvg(nn.Module):
    def __init__(self, dev) -> None:
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction='sum').to(dev)
    def forward(self, pred, tgt):
        return self.loss_fn(pred, tgt) / len(pred)


# wrapped loss functions
@LOSS_REGISTRY.register()
class SanityCheck(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.loss_fn = kwargs['reg']
    def forward(self, **kwargs):
        return self.loss_fn(kwargs['preds'], kwargs['gt'])

@LOSS_REGISTRY.register()
class FixedResDiff(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.reg_loss_fn = kwargs['reg']
        self.diff_loss_fn = kwargs['diff']
    def forward(self, **kwargs):
        # res loss
        reg_loss = self.reg_loss_fn(kwargs['res_recon'], kwargs['gt_res'])
        # diff loss
        if kwargs['predict_x_start']:
            diff_loss = self.diff_loss_fn(kwargs['res_recon'], kwargs['gt_res'])
        else:
            diff_loss = self.loss_fn(kwargs['pred_noise'], kwargs['gt_noise'])

        return {'reg_loss': reg_loss, 'diff_loss': diff_loss}

@LOSS_REGISTRY.register()
class ResDiff(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.reg_loss_fn = kwargs['reg']
        self.diff_loss_fn = kwargs['diff']
    def forward(self, **kwargs):
        # res loss
        reg_loss = self.res_loss_fn(kwargs['preds'], kwargs['gt'])
        # diff loss
        if kwargs['predict_x_start']:
            diff_loss = self.diff_loss_fn(kwargs['res_recon'], kwargs['gt_res'])
        else:
            diff_loss = self.diff_loss_fn(kwargs['pred_noise'], kwargs['gt_noise'])

        losses = {
            'reg_loss': reg_loss,
            'diff_loss': diff_loss,
            # 'loss': reg_loss + diff_loss,
        }
        return  losses

@LOSS_REGISTRY.register()
class ResDiffPlus(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.reg_loss_fn = kwargs['reg']
        self.diff_loss_fn = kwargs['diff']
    def forward(self, **kwargs):
        # res loss
        reg_loss = self.res_loss_fn(kwargs['preds']+kwargs['res'].detach(), kwargs['gt'])
        # diff loss
        if kwargs['predict_x_start']:
            diff_loss = self.diff_loss_fn(kwargs['res_recon'], kwargs['gt_res'])
        else:
            diff_loss = self.diff_loss_fn(kwargs['pred_noise'], kwargs['gt_noise'])

        losses = {
            'reg_loss': reg_loss,
            'diff_loss': diff_loss,
            # 'loss': reg_loss + diff_loss,
        }
        return  losses


# loss_dic = {
#     'sanity_check': SanityCheck,
#     'fixed_res_and_diff': FixedResDiff,
#     'res_and_diff': ResDiff,
#     'res_and_diff_plus': ResDiffPlus,
# }

def build_criterion(loss_opt, device):
    #loss_fn = loss_dic[loss_opt['type']](loss_opt, device)
    losses = {}
    if not loss_opt['regress'] is None:
        reg_loss_fn = LOSS_REGISTRY.get(loss_opt['regress'])(dev=device)
        losses['reg'] = reg_loss_fn
    if not loss_opt['diffusion'] is None:
        diff_loss_fn = LOSS_REGISTRY.get(loss_opt['diffusion'])(dev=device)
        losses['diff'] = diff_loss_fn
    loss_fn = LOSS_REGISTRY.get(loss_opt['type'])(**losses)

    return loss_fn