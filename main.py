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
import random
import os
import numpy as np

# distributed training
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import sys
sys.path.append('/root/Improve-HPE-with-Diffusion')
from core.dist import *

# sweep search
from config.sweep_config import *

# num_gpu = torch.cuda.device_count()


def _init_fn(worker_id):
    np.random.seed(int(os.environ['PYTHONHASHSEED']))
    random.seed(int(os.environ['PYTHONHASHSEED']))

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    # rank 0 process
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # nccl：NVIDIA Collective Communication Library 
    # 分布式情况下的，gpus 间通信
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def main_worker(gpu, opt, args):
    if opt['distributed']:
        ddp_setup(gpu, opt['world_size'])

    # # set up seed
    if opt['seed'] is not None:
        setup_seed(opt['seed'])

    if is_primary():
        # init wandb
        if not args.sweep:
            project_name = opt['wandb']['project_name']
            run_name = opt['wandb']['run_name']
            wandb.init(project=project_name, config=args, name=run_name)
        else:
            wandb.init(project="sweep-coco-subset")
            sweep_cfg = wandb.config
            print(sweep_cfg)
            opt = merge_opt(opt=opt, sweep_config=sweep_cfg)

        # logging
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        Logger.setup_logger(None, opt['path']['log'],
                            'train', level=logging.INFO, screen=True)
        Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
        logger = logging.getLogger('base')
        logger.info(Logger.dict2str(opt))

    # dataset
    train_dataset, val_dataset = build_datasets(opt['datasets'])
    if is_primary():
        logger.info('Initial Dataset Finished')
    train_generator = train_dataset.get_generator()
    val_generator = val_dataset.get_generator()
    
    # model
    if opt['distributed']:
        opt['current_id'] = gpu
    diffusion = Model.create_model(opt)    # diffusion - DDPM
    if is_primary():
        logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    
    if opt['phase'] == 'train':
        #Train
        num_of_epochs = opt['train']['end_epoch']
        current_step = diffusion.begin_step
        current_epoch = diffusion.begin_epoch

        if is_primary():
            if opt['path']['resume_state']:
                logger.info('Resuming training from epoch: {}, iter: {}.'.format(
                    current_epoch, current_step))
                if diffusion.random_state is not None:
                    train_generator.set_random_state(diffusion.random_state)
            else:
                logger.info('Starting training from epoch: 0, iter: 0.')

        all_actions = train_dataset.get_actions()
        all_actions_val = val_dataset.get_actions()
        best_val = 1e9

        while current_epoch < num_of_epochs:
            # if opt['distributed']:
            #     train_sampler.set_epoch(current_epoch)
            # for train_data in train_loader:             
            for cameras_train, _, batch_3d, batch_2d in train_generator.next_epoch():
                # forward pass
                # diffusion.feed_data(train_data)
                if cameras_train is not None:
                    cameras_train = torch.from_numpy(cameras_train.astype('float32'))
                inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
                inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                if torch.cuda.is_available():
                    inputs_3d = inputs_3d.cuda()
                    inputs_2d = inputs_2d.cuda()
                    if cameras_train is not None:
                        cameras_train = cameras_train.cuda()
                inputs_3d[:, :, 0] = 0

                diffusion.optimize_parameters(inputs_2d, inputs_3d, epoch=current_epoch)

                # lr stepping
                # if opt['train']['scheduler_type'] != 'plateau':
                #     diffusion.lr_scheduler.step()

                # log
                if current_step % opt['train']['print_freq'] == 0 and is_primary():
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}>\n '.format(current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        #tb_logger.add_scalar(k, v, current_step)
                    # logger.info(message)
                    wandb.log(logs, step=current_step, commit=False)
                    wandb.log({'lr': diffusion.optimizer.param_groups[0]['lr']}, step=current_step, commit=False)

                current_step += 1
                #break
            # end of epoch ===================================================================

            # end of epoch evaluation
            #if current_epoch % 5 == 0 and current_epoch != 0:
            if current_epoch % 1 == 0:
                diffusion.set_new_noise_schedule(
                    opt['model']['beta_schedule']['val'], schedule_phase='val')
                diffusion.validate(val_dataset.kps_left, val_dataset.kps_right, val_dataset.joints_left, val_dataset.joints_right, opt["datasets"]["num_frames"], val_generator, all_actions_val)
                # log
                if is_primary():
                    eval_logs = diffusion.get_current_metrics()
                    message = 'Evaluation at <epoch:{:3d}, iter:{:8,d}\n> '.format(current_epoch, current_step)
                    for k, v in eval_logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                    logger.info(message)
                    wandb.log(eval_logs, step=current_step)

                diffusion.set_new_noise_schedule(
                    opt['model']['beta_schedule']['train'], schedule_phase='train')

                # save best model
                during_warm_up = opt["loss"]["warm_up"] and current_epoch <= opt["loss"]["warm_up_phase1_epochs"]
                if not opt["model"]["diffusion"]["diff_on"] or during_warm_up:
                    current_val = eval_logs["Average_p1"]
                else:
                    current_val = eval_logs["Average_diff_p1"]
                if current_val < best_val and is_primary():
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step, train_generator.random_state())
                    best_val = current_val
            
            if opt['distributed']:
                dist.barrier()  # Sync

            diffusion.lr_scheduler.step()
            current_epoch += 1

        # end of training
        if is_primary():
            logger.info('End of training.')
    else:
        if is_primary():
            logger.info('Begin Model Evaluation.')
        all_actions = val_dataset.get_actions()
        diffusion.set_new_noise_schedule(
            opt['model']['beta_schedule']['val'], schedule_phase='val')
        diffusion.validate(val_dataset.kps_left, val_dataset.kps_right, val_dataset.joints_left, val_dataset.joints_right, opt["datasets"]["num_frames"], val_generator, all_actions)

        # log
        if is_primary():
            eval_logs = diffusion.get_current_metrics()
            message = 'Final eval:\n'
            for k, v in eval_logs.items():
                message += '{:s}: {:.4e} '.format(k, v)
            logger.info(message)
            wandb.log(eval_logs)

def main(opt, args):
    # launch main_worker
    if opt['world_size'] == 1:
        if opt['seed'] is not None:
            setup_seed(opt['seed'])
        main_worker(None, opt, args)
    else:
        mp.spawn(main_worker, nprocs=opt['world_size'], args=(opt, args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/root/Improve-HPE-with-Diffusion/config/mixste/two_stage_mixste_h36m_train_diff_u.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                         help='Run either train(training) or val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default="0")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--sweep', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # search hyperparams with wandb sweep
    if not args.sweep:
        main(opt, args)
    else:
        sweep_config = build_sweep_config()
        #opt = merge_opt(opt, sweep_config)
        sweep_fn = lambda opt=opt, args=args: main(opt, args)
        sweep_id = wandb.sweep(sweep_config)
        #wandb.init()
        wandb.agent(sweep_id=sweep_id, function=sweep_fn, count=64)