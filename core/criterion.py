import math
import torch
import torch.nn as nn

import sys
sys.path.append('/root/Improve-HPE-with-Diffusion')
from core.registry import Registry
from core.metrics import mpjpe

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
class L1LossBatchAvgVaraibleStd(nn.Module):
    def __init__(self, dev) -> None:
        super().__init__()
        self.amp = 1 / math.sqrt(2 * math.pi)
    def forward(self, pred, tgt, sigma):
        b = len(pred)
        loss = torch.log(2 * sigma) + torch.abs(tgt - pred) / (sigma + 1e-9)
        loss = loss.sum() / b
        return loss

@LOSS_REGISTRY.register()
class L2LossBatchAvgVaraibleStd(nn.Module):
    def __init__(self, dev) -> None:
        super().__init__()
        self.amp = 1 / math.sqrt(2 * math.pi)
    def forward(self, preds, sigmas, tgt):
        # b = len(pred)
        #loss = torch.log(sigmas / self.amp) + torch.square(tgt - preds) / (2 * torch.square(sigmas) + 1e-9)
        loss = 0.5*torch.log(sigmas/self.amp + 1e-9) + torch.abs(tgt - preds) / (math.sqrt(2) * sigmas + 1e-9)
        assert loss.ndim == 4
        loss = torch.mean(torch.sum(loss, dim=-1))
        return loss
    
@LOSS_REGISTRY.register()
class MpjpeVaraibleStd(nn.Module):
    def __init__(self, dev) -> None:
        super().__init__()
        self.amp = 1 / math.sqrt(2 * math.pi)
    def forward(self, preds, sigmas, tgt):
        # b = len(pred)
        loss = torch.log(sigmas / self.amp) + torch.square(tgt - preds) / (2 * torch.square(sigmas) + 1e-9)
        loss = torch.sqrt(loss)
        assert loss.ndim == 4
        loss = torch.mean(torch.sum(loss, dim=-1))
        return loss

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
        assert pred.ndim == 4
        # pred = pred.reshape(pred.shape[0], -1, 3)
        # tgt = tgt.reshape(tgt.shape[0], -1, 3)
        #return torch.mean(torch.norm(pred - tgt, dim=-1))
        #return self.loss_fn(pred, tgt) / (pred.shape[0] * pred.shape[1])
        return self.loss_fn(pred, tgt) / (pred.shape[0] * pred.shape[1] * pred.shape[2])
    
@LOSS_REGISTRY.register()
class L2Norm(nn.Module):
    def __init__(self, dev) -> None:
        super().__init__()
    def forward(self, pred, tgt):
        pred = pred.reshape(pred.shape[0], -1, 3)
        tgt = tgt.reshape(tgt.shape[0], -1, 3)
        return torch.mean(torch.norm(pred - tgt, dim=-1))



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
            if not kwargs['rle_loss']:
                diff_loss = self.diff_loss_fn(kwargs['res_recon'], kwargs['gt_res'])
            else:
                diff_loss = self.diff_loss_fn(kwargs['res_recon'], kwargs['gt_res'], kwargs['sigma'])
        else:
            if not kwargs['rle_loss']:
                diff_loss = self.diff_loss_fn(kwargs['pred_noise'], kwargs['gt_noise'])
            else:
                diff_loss = self.diff_loss_fn(kwargs['pred_noise'], kwargs['gt_noise'], kwargs['sigma'])

        return {'reg_loss': reg_loss, 'diff_loss': diff_loss}

@LOSS_REGISTRY.register()
class ResDiff(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.reg_loss_fn = kwargs['reg_fn']
        self.diff_loss_fn = kwargs['diff_fn']
        self.reg_weight = kwargs['reg_weight']
        self.diff_weight = kwargs['diff_weight']
    def forward(self, **kwargs):
        # res loss
        reg_loss = self.reg_loss_fn(kwargs['preds'], kwargs['gt'])
        # diff loss
        if kwargs['predict_x_start']:
            diff_loss = self.diff_loss_fn(kwargs['res_recon'], kwargs['gt_res'])
        else:
            diff_loss = self.diff_loss_fn(kwargs['pred_noise'], kwargs['gt_noise'])

        losses = {
            'reg_loss': reg_loss,
            'diff_loss': diff_loss,
            'loss': reg_loss * self.reg_weight + diff_loss * self.diff_weight,
        }
        return  losses

@LOSS_REGISTRY.register()
class MixSTELoss(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.w = torch.tensor(kwargs["weights_joints"]).cuda() if "weights_joints" in kwargs else None
        self.w_dif = kwargs["w_dif"]
        self.w_vel = kwargs["w_vel"]

    def weighted_mpjpe(self, pred, tgt):  # weighted mpjpe
        return torch.mean(self.w * torch.norm(pred - tgt, dim=-1))
    
    def dif_seq(self, pred):  # temporal consistency loss
        # pred - (b,f,n,3)
        dif_seq = pred[:,1:,:,:] - pred[:,:-1,:,:]

        weights_joints = torch.ones_like(dif_seq).cuda()
        assert self.w.shape[0] == weights_joints.shape[-2]
        weights_joints = torch.mul(weights_joints.permute(0,1,3,2), self.w).permute(0,1,3,2)
        dif_seq = torch.mean(torch.multiply(weights_joints, torch.square(dif_seq)))

        return dif_seq
    
    def mean_velocity_error(self, pred, tgt):
        """
        Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
        """
        assert pred.shape == tgt.shape
        
        velocity_predicted = torch.diff(pred, dim=1)  # compute difference along the frame axis
        velocity_target = torch.diff(tgt, dim=1)

        return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=-1))
    
    def forward(self, **kwargs):
        loss1 = self.weighted_mpjpe(kwargs["preds"], kwargs["gt"])
        loss2 = self.dif_seq(kwargs["preds"])
        loss3 = self.mean_velocity_error(kwargs["preds"], kwargs["gt"])
        loss = loss1 + self.w_dif*loss2 + self.w_vel*loss3
        train_mpjpe = mpjpe(kwargs["preds"], kwargs["gt"])
        return {
            "weighted_mpjpe": loss1,
            "dif_seq": loss2,
            "mean_velocity_error": loss3,
            "loss": loss,
            "train_mpjpe": train_mpjpe
            }
    
@LOSS_REGISTRY.register()
class MixSTELossWithL2(MixSTELoss):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def forward(self, **kwargs):
        device = torch.device("cuda") if torch.cuda.is_available else None
        loss1 = L2LossBatchAvg(dev=device)(kwargs["preds"], kwargs["gt"])
        loss2 = self.dif_seq(kwargs["preds"])
        loss3 = self.mean_velocity_error(kwargs["preds"], kwargs["gt"])
        loss = loss1 + self.w_dif*loss2 + self.w_vel*loss3
        train_mpjpe = mpjpe(kwargs["preds"], kwargs["gt"])
        return {
            "averaged_l2_loss": loss1,
            "dif_seq": loss2,
            "mean_velocity_error": loss3,
            "loss": loss,
            "train_mpjpe": train_mpjpe
            }

@LOSS_REGISTRY.register()
class MixSTELossWithSigmaL2(MixSTELoss):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def forward(self, **kwargs):
        device = torch.device("cuda") if torch.cuda.is_available else None
        loss1 = L2LossBatchAvgVaraibleStd(dev=device)(kwargs["preds"], kwargs["sigmas"], kwargs["gt"])
        loss2 = self.dif_seq(kwargs["preds"])
        loss3 = self.mean_velocity_error(kwargs["preds"], kwargs["gt"])
        loss = loss1 + self.w_dif*loss2 + self.w_vel*loss3
        train_mpjpe = mpjpe(kwargs["preds"], kwargs["gt"])
        return {
            "averaged_l2_loss": loss1,
            "dif_seq": loss2,
            "mean_velocity_error": loss3,
            "loss": loss,
            "train_mpjpe": train_mpjpe
            }
    
@LOSS_REGISTRY.register()
class MixSTELossWithSigmaNorm(MixSTELoss):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def forward(self, **kwargs):
        device = torch.device("cuda") if torch.cuda.is_available else None
        loss1 = MpjpeVaraibleStd(dev=device)(kwargs["preds"], kwargs["sigmas"], kwargs["gt"])
        loss2 = self.dif_seq(kwargs["preds"])
        loss3 = self.mean_velocity_error(kwargs["preds"], kwargs["gt"])
        loss = loss1 + self.w_dif*loss2 + self.w_vel*loss3
        train_mpjpe = mpjpe(kwargs["preds"], kwargs["gt"])
        return {
            "averaged_l2_loss": loss1,
            "dif_seq": loss2,
            "mean_velocity_error": loss3,
            "loss": loss,
            "train_mpjpe": train_mpjpe
            }


def build_criterion(loss_opt, device):
    # losses = {}
    if not loss_opt['regress'] is None:
        reg_loss_fn = LOSS_REGISTRY.get(loss_opt['regress'])(dev=device)
        loss_opt['reg_fn'] = reg_loss_fn
    if not loss_opt['diffusion'] is None:
        diff_loss_fn = LOSS_REGISTRY.get(loss_opt['diffusion'])(dev=device)
        loss_opt['diff_fn'] = diff_loss_fn
    loss_fn = LOSS_REGISTRY.get(loss_opt['type'])(**loss_opt)
    return loss_fn