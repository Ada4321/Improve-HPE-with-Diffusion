import torch.nn as nn


class SanityCheck(nn.Module):
    def __init__(self, opt, device) -> None:
        super().__init__()
        self.opt = opt
        if opt['regress'] == 'l1':
            self.loss_fn = nn.L1Loss(reduction='sum').to(device)
        elif opt['regress'] == 'l2':
            self.loss_fn = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError
    
    def forward(self, **kwargs):
        if self.opt['avg_batch']['reg']:
            loss = self.loss_fn(kwargs['preds'], kwargs['gt']) / len(kwargs['preds'])
        else:
            loss = self.loss_fn(kwargs['preds'], kwargs['gt'])
        return {'reg_loss': loss}
    
class FixedResDiff(nn.Module):
    def __init__(self, opt, device) -> None:
        super().__init__()
        self.opt = opt
        #print(opt['diffusion'])
        if opt['diffusion'] == 'l1':
            self.loss_fn = nn.L1Loss(reduction='sum').to(device)
        elif opt['diffusion'] == 'l2':
            self.loss_fn = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError 
    
    def forward(self, **kwargs):
        if kwargs['predict_x_start']:
            if self.opt['avg_batch']['diff']:
                loss = self.loss_fn(kwargs['res_recon'], kwargs['gt_res']) / len(kwargs['res_recon'])
            else:
                loss = self.loss_fn(kwargs['res_recon'], kwargs['gt_res'])
        else:
            if self.opt['avg_batch']['diff']:
                loss = self.loss_fn(kwargs['pred_noise'], kwargs['gt_noise']) / len(kwargs['pred_noise'])
            else:
                loss = self.loss_fn(kwargs['pred_noise'], kwargs['gt_noise'])

        return {'diff_loss': loss}
    
class ResDiff(nn.Module):
    def __init__(self, opt, device) -> None:
        super().__init__()
        self.opt = opt
        self.res_loss_fn = self.set_loss(opt['regress'], device)
        self.diff_loss_fn = self.set_loss(opt['diffusion'], device)
    
    def set_loss(self, type, device):
        if type == 'l1':
            loss_fn = nn.L1Loss(reduction='sum').to(device)
        elif type == 'l2':
            loss_fn = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError
        return loss_fn

    def forward(self, **kwargs):
        # res loss
        if self.opt['avg_batch']['reg']:
            reg_loss = self.res_loss_fn(kwargs['preds'], kwargs['gt']) / len(kwargs['preds'])
        else:
            reg_loss = self.res_loss_fn(kwargs['preds'], kwargs['gt'])
        
        # diff loss
        if kwargs['predict_x_start']:
            if self.opt['avg_batch']['diff']:
                diff_loss = self.diff_loss_fn(kwargs['res_recon'], kwargs['gt_res']) / len(kwargs['res_recon'])
            else:
                diff_loss = self.diff_loss_fn(kwargs['res_recon'], kwargs['gt_res'])
        else:
            if self.opt['avg_batch']['diff']:
                diff_loss = self.diff_loss_fn(kwargs['pred_noise'], kwargs['gt_noise']) / len(kwargs['pred_noise'])
            else:
                diff_loss = self.diff_loss_fn(kwargs['pred_noise'], kwargs['gt_noise'])

        losses = {
            'reg_loss': reg_loss,
            'diff_loss': diff_loss,
            # 'loss': reg_loss + diff_loss,
        }
        return  losses
    
class ResDiffPlus(nn.Module):
    def __init__(self, opt, device) -> None:
        super().__init__()
        self.opt = opt
        self.res_loss_fn = self.set_loss(opt['regress'], device)
        self.diff_loss_fn = self.set_loss(opt['diffusion'], device)

    def set_loss(self, type, device):
        if type == 'l1':
            loss_fn = nn.L1Loss(reduction='sum').to(device)
        elif type == 'l2':
            loss_fn = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError
        return loss_fn
    
    def forward(self, **kwargs):
        # res loss
        if self.opt['avg_batch']['reg']:
            reg_loss = self.res_loss_fn(kwargs['preds']+kwargs['res'].detach(), kwargs['gt']) / len(kwargs['preds'])
        else:
            reg_loss = self.res_loss_fn(kwargs['preds']+kwargs['res'].detach(), kwargs['gt'])

        # diff loss
        if kwargs['predict_x_start']:
            if self.opt['avg_batch']['diff']:
                diff_loss = self.diff_loss_fn(kwargs['res_recon'], kwargs['gt_res']) / len(kwargs['res_recon'])
            else:
                diff_loss = self.diff_loss_fn(kwargs['res_recon'], kwargs['gt_res'])
        else:
            if self.opt['avg_batch']['diff']:
                diff_loss = self.diff_loss_fn(kwargs['pred_noise'], kwargs['gt_noise']) / len(kwargs['pred_noise'])
            else:
                diff_loss = self.diff_loss_fn(kwargs['pred_noise'], kwargs['gt_noise'])

        losses = {
            'reg_loss': reg_loss,
            'diff_loss': diff_loss,
            # 'loss': reg_loss + diff_loss,
        }
        return  losses


loss_dic = {
    'sanity_check': SanityCheck,
    'fixed_res_and_diff': FixedResDiff,
    'res_and_diff': ResDiff,
    'res_and_diff_plus': ResDiffPlus,
}

def build_criterion(loss_opt, device):
    loss_fn = loss_dic[loss_opt['type']](loss_opt, device)

    return loss_fn