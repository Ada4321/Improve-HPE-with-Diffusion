import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler


class MyExponentialLR(_LRScheduler):
    """The version of ExponentialLR that reduces lr by gamma every 'decay_epochs'
    """
    def __init__(self, optimizer, decay_epochs, gamma=0.1, last_epoch=-1):
        self.decay_epochs = decay_epochs
        self.gamma = gamma
        super(MyExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch / self.decay_epochs)
                for base_lr in self.base_lrs]
    
    
scheduler_dict = {
    'exponential': MyExponentialLR,
    'step': lr_scheduler.StepLR,
    'multistep': lr_scheduler.MultiStepLR,
    'plateau': lr_scheduler.ReduceLROnPlateau,
}

def build_scheduler(sche_type, optimizer, **kwargs):
    return scheduler_dict[sche_type](optimizer=optimizer, **kwargs)