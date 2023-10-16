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
    def __init__(self, dev=torch.device("cuda"), **kwargs) -> None:
        super().__init__()
        self.loss_fn = nn.L1Loss(reduction='sum').to(dev)
    def forward(self, preds, sigmas, gt):
        return self.loss_fn(preds, gt)

@LOSS_REGISTRY.register()
class L1LossBatchAvg(nn.Module):
    def __init__(self, dev=torch.device("cuda"), **kwargs) -> None:
        super().__init__()
        self.loss_fn = nn.L1Loss(reduction='sum').to(dev)
    def forward(self, preds, sigmas, gt):
        assert preds.ndim == 4
        return self.loss_fn(preds, gt) / (preds.shape[0] * preds.shape[1] * preds.shape[2])

@LOSS_REGISTRY.register()
class L1LossBatchAvgMLE(nn.Module):
    def __init__(self, dev=torch.device("cuda"), **kwargs) -> None:
        super().__init__()
        self.amp = 1 / math.sqrt(2 * math.pi)
    def forward(self, preds, sigmas, gt):
        assert preds.ndim == 4
        loss = torch.log(sigmas/self.amp + 1e-9) + torch.abs(gt - preds) / (math.sqrt(2) * sigmas + 1e-9)
        loss = torch.mean(torch.sum(loss, dim=-1))
        return loss

@LOSS_REGISTRY.register()
class L2Loss(nn.Module):
    def __init__(self, dev=torch.device("cuda"), **kwargs) -> None:
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction='sum').to(dev)
    def forward(self, preds, sigmas, gt):
        return self.loss_fn(preds, gt)

@LOSS_REGISTRY.register()
class L2LossBatchAvg(nn.Module):
    def __init__(self, dev=torch.device("cuda"), **kwargs) -> None:
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction='sum').to(dev)
    def forward(self, preds, sigmas, gt):
        assert preds.ndim == 4
        return self.loss_fn(preds, gt) / (preds.shape[0] * preds.shape[1] * preds.shape[2])
    
@LOSS_REGISTRY.register()
class L2Norm(nn.Module):
    def __init__(self, dev=torch.device("cuda"), **kwargs) -> None:
        super().__init__()
    def forward(self, preds, sigmas, gt):
        assert preds.ndim == 4
        return torch.mean(torch.norm(preds - gt, dim=-1))

@LOSS_REGISTRY.register()
class L2NormBatchAvgMLE(nn.Module):
    def __init__(self, dev=torch.device("cuda"), **kwargs) -> None:
        super().__init__()
        self.amp = 1 / math.sqrt(2 * math.pi)
    def forward(self, preds, sigmas, gt):
        assert preds.ndim == 4
        loss1 = torch.norm(torch.abs(gt - preds) / (math.sqrt(2) * sigmas + 1e-9), dim=-1)
        loss2 = torch.sum(torch.log(sigmas/self.amp + 1e-9), dim=-1)
        loss = torch.mean(loss1) + torch.mean(loss2)
        return loss

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
        loss = loss1 + self.w_dif * loss2 + self.w_vel * loss3
        train_mpjpe = mpjpe(kwargs["preds"], kwargs["gt"])
        return {
            "weighted_mpjpe": loss1,
            "dif_seq": loss2,
            "mean_velocity_error": loss3,
            "loss": loss,
            "train_mpjpe_backbone": train_mpjpe
            }

@LOSS_REGISTRY.register()
class MixSTELossMLE(MixSTELoss):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        device = torch.device("cuda") if torch.cuda.is_available else None
        self.mle_loss = LOSS_REGISTRY.get(kwargs["mle_loss"])(dev=device)

    def forward(self, **kwargs):
        loss1 = self.mle_loss(kwargs["preds"], kwargs["sigmas"], kwargs["gt"])
        loss2 = self.dif_seq(kwargs["preds"])
        loss3 = self.mean_velocity_error(kwargs["preds"], kwargs["gt"])
        loss = loss1 + self.w_dif * loss2 + self.w_vel * loss3
        train_mpjpe = mpjpe(kwargs["preds"], kwargs["gt"])
        return {
            "averaged_l2_loss": loss1,
            "dif_seq": loss2,
            "mean_velocity_error": loss3,
            "loss": loss,
            "train_mpjpe_backbone": train_mpjpe
            }

@LOSS_REGISTRY.register()
class ResDiff(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.reg_loss_fn = kwargs['reg_fn']
        self.diff_loss_fn = kwargs['diff_fn']
        self.reg_weight = kwargs['reg_weight']
        self.diff_weight = kwargs['diff_weight']
        self.warm_up = kwargs["warm_up"]
        self.warm_up_phase1_epochs = kwargs["warm_up_phase1_epochs"]
        self.warm_up_phase2_epochs = kwargs["warm_up_phase2_epochs"]
    def forward(self, **kwargs):
        # res loss
        # preds, gt -- (b,f,n,d)
        reg_losses = self.reg_loss_fn(preds=kwargs['preds'], sigmas=kwargs["sigmas"], gt=kwargs['gt'])
        reg_loss = reg_losses.pop("loss")
        # diff loss
        if kwargs['predict_x_start']:
            diff_loss = self.diff_loss_fn(kwargs['res_recon'], None, kwargs['gt_res'])
        else:
            diff_loss = self.diff_loss_fn(kwargs['pred_noise'], None, kwargs['gt_noise'])
        if kwargs["norm_res"]:
            train_mpjpe = mpjpe(kwargs["preds"]+kwargs["res_recon"]*kwargs["sigmas"], kwargs["gt"])
        else:
            train_mpjpe = mpjpe(kwargs["preds"]+kwargs["res_recon"], kwargs["gt"])

        epoch = kwargs["epoch"]
        if self.warm_up:
            if epoch < self.warm_up_phase1_epochs:
                self.diff_weight = 0.
            elif epoch < self.warm_up_phase2_epochs:
                self.diff_weight = \
                    self.diff_weight * (epoch - self.warm_up_phase1_epochs) / (self.warm_up_phase2_epochs - self.warm_up_phase1_epochs)
        
        losses = {
            'reg_loss': reg_loss,
            'diff_loss': diff_loss,
            "diff_weight": self.diff_weight,
            'loss': reg_loss * self.reg_weight + diff_loss * self.diff_weight,
            "train_mpjpe": train_mpjpe
        }
        losses.update(reg_losses)

        return  losses



def build_criterion(loss_opt, device):
    # losses = {}
    if not loss_opt['regress'] is None:
        reg_loss_fn = LOSS_REGISTRY.get(loss_opt['regress'])(dev=device, **loss_opt)
        loss_opt['reg_fn'] = reg_loss_fn
    if not loss_opt['diffusion'] is None:
        diff_loss_fn = LOSS_REGISTRY.get(loss_opt['diffusion'])(dev=device, **loss_opt)
        loss_opt['diff_fn'] = diff_loss_fn
    loss_fn = LOSS_REGISTRY.get(loss_opt['type'])(**loss_opt)
    return loss_fn