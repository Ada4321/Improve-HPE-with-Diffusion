import torch
# import data as Data
from datasets import build_datasets
import model as Model
import argparse
import logging
import core.logger as Logger
from core.transforms import get_coord
from core.wandb_logger import WandbLogger
# from tensorboardX import SummaryWriter
import wandb
# import os
# import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sanitycheck.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                         help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    # parser.add_argument('-debug', '-d', action='store_true')
    # parser.add_argument('-enable_wandb', action='store_true')
    # parser.add_argument('-log_wandb_ckpt', action='store_true')
    # parser.add_argument('-log_eval', action='store_true')
    # parser.add_argument('-project_name', type=str, default='DiffHPE')
    # parser.add_argument('-run_name', type=str, default='')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # init wandb
    project_name = opt['wandb']['project_name']
    run_name = opt['wandb']['run_name']
    wandb.init(project=project_name, config=args, name=run_name)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    # tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    # wandb_logger = WandbLogger(opt)
    # wandb.define_metric('validation/val_step')
    # wandb.define_metric('epoch')
    # wandb.define_metric("validation/*", step_metric="val_step")

    # dataset
    train_dataset, val_dataset, test_dataset = build_datasets(opt['datasets'], opt['data_preset'])
    logger.info('Initial Dataset Finished')

    # dataloder
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt['train']['batch_size'], shuffle=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt['test']['batch_size'], shuffle=False)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt['test']['batch_size'], shuffle=False)

    # model
    diffusion = Model.create_model(opt)    # diffusion - DDPM
    logger.info('Initial Model Finished')

    # eval tools
    output_3d = opt['data_preset'].get('out_3d', False)
    heatmap_to_coord = get_coord(opt, opt['data_preset']['heatmap_size'], output_3d)
    json_path = opt['path']['json_dt']
    json_path_gt = opt['path']['json_gt']

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    
    if opt['phase'] == 'train':
        # Train
        num_of_epochs = opt['train']['end_epoch']
        current_step = diffusion.begin_step
        current_epoch = diffusion.begin_epoch
        # n_iter = opt['train']['n_iter']

        if opt['path']['resume_state']:
            logger.info('Resuming training from epoch: {}, iter: {}.'.format(
                current_epoch, current_step))
        else:
            logger.info('Starting training from epoch: 0, iter: 0.')

        while current_epoch < num_of_epochs:
            for _, (inps, labels, _, _) in enumerate(train_loader):
                train_data = (inps, labels)             
                
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()

                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}>\n '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        #tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)
                    wandb.log(logs, step=current_step, commit=False)

                # validation
                if current_step != 0 and current_step % opt['train']['val_freq'] == 0:
                    # result_path = '{}/{}'.format(opt['path']['results'], current_epoch)
                    # os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    diffusion.validate(json_path, heatmap_to_coord, test_loader)
                    diffusion.validate_gt(json_path_gt, heatmap_to_coord, val_loader)
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    
                    # log
                    eval_logs = diffusion.get_current_metrics()
                    message = 'Evaluation at <epoch:{:3d}, iter:{:8,d}\n> '.format(
                        current_epoch, current_step)
                    for k, v in eval_logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                    logger.info(message)
                    wandb.log(eval_logs, step=current_step)

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                current_step += 1

            current_epoch += 1

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        # result_path = '{}'.format(opt['path']['results'])
        # os.makedirs(result_path, exist_ok=True)

        diffusion.set_new_noise_schedule(
            opt['model']['beta_schedule']['val'], schedule_phase='val')
        diffusion.validate(json_path, heatmap_to_coord, test_loader)
        diffusion.validate_gt(json_path_gt, heatmap_to_coord, val_loader)

        # log
        eval_logs = diffusion.get_current_metrics()
        message = 'Final eval:\n'
        for k, v in eval_logs.items():
            message += '{:s}: {:.4e} '.format(k, v)
        logger.info(message)