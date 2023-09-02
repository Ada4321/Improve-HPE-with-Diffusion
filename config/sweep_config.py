import math


def build_sweep_config():
    # searching method
    sweep_config = {
        'method': 'bayes'
    }

    # parameter range
    parameters_dict = {
        # learning rate
        'lr_diff': {
            'distribution': 'uniform',
            'min': 1e-6,
            'max': 1e-4
        },
        # batch size
        # 'batch_size': {
        #     'distribution': 'q_log_uniform',
        #     'q': 1,
        #     'min': math.log(64),
        #     'max': math.log(512),
        # },
        # denoiser
        'dim': {
            'values': [256, 512, 768, 1024]
            },
        'num_image_embeds': {
            'values': [1, 2, 4, 8]
            },
        'depth': {
            'values': [8, 12]
            },
    }

    # goal of sweeping
    metric = {
        'name': 'val_diff_loss',
        'goal': 'minimize'
    }

    sweep_config['parameters'] = parameters_dict
    sweep_config['metric'] = metric
    return sweep_config

def merge_opt(opt, sweep_config):
    # searching for appropriate lr, bz, denoiser depth and width
    if 'lr_reg' in sweep_config:
        opt['train']['lr']['reg'] = sweep_config['lr_reg']
    if 'lr_diff' in sweep_config:
        opt['train']['lr']['diff'] = sweep_config['lr_diff']
    if 'batch_size' in sweep_config:
        opt['train']['batch_size'] = sweep_config['batch_size']
        opt['test']['batch_size'] = sweep_config['batch_size']
    if 'dim' in sweep_config:
        opt['model']['denoise_transformer']['dim'] = sweep_config['dim']
        assert sweep_config['dim'] % opt['model']['denoise_transformer']['transformer']['dim_head'] == 0
        opt['model']['denoise_transformer']['transformer']['heads'] = int(sweep_config['dim'] / opt['model']['denoise_transformer']['transformer']['dim_head'])
    if 'depth' in sweep_config:
        opt['model']['denoise_transformer']['transformer']['depth'] = sweep_config['depth']
    if 'num_image_embeds' in sweep_config:
        opt['model']['denoise_transformer']['num_image_embeds'] = sweep_config['num_image_embeds']

    return opt