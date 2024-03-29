import torch
# import data as Data
from datasets import build_datasets_debug
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

    # set up seed
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
    train_dataset, val_dataset, test_dataset = build_datasets_debug(opt['datasets'], opt['data_preset'])
    if is_primary():
        logger.info('Initial Dataset Finished')

    # dataloder
    if not opt['distributed']:
        train_loader = DataLoader(
            train_dataset, batch_size=opt['test']['batch_size'], shuffle=True)
        val_loader = DataLoader(
            val_dataset, batch_size=opt['test']['batch_size'], shuffle=False)
        test_loader = DataLoader(
            test_dataset, batch_size=opt['test']['batch_size'], shuffle=False)
    else:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=opt['world_size'], rank=get_rank())
        train_loader = DataLoader(
            train_dataset, batch_size=opt['train']['batch_size'], shuffle=(train_sampler is None), num_workers=0, sampler=train_sampler, worker_init_fn=_init_fn)
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=opt['world_size'], rank=get_rank())
        val_loader = DataLoader(
            val_dataset, batch_size=opt['test']['batch_size'], shuffle=False, num_workers=0, sampler=val_sampler, drop_last=False)
        test_sampler = DistributedSampler(
            test_dataset, num_replicas=opt['world_size'], rank=get_rank())
        test_loader = DataLoader(
            test_dataset, batch_size=opt['test']['batch_size'], shuffle=False, num_workers=0, sampler=test_sampler, drop_last=False)
    
    # model
    if opt['distributed']:
        opt['current_id'] = gpu
    diffusion = Model.create_model_debug(opt)    # diffusion - DDPM
    if is_primary():
        logger.info('Initial Model Finished')

    # eval tools
    output_3d = opt['data_preset'].get('out_3d', False)
    heatmap_to_coord = get_coord(opt, opt['data_preset']['heatmap_size'], output_3d)
    json_path = opt['path']['json_dt']
    json_path_gt = opt['path']['json_gt']

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
            else:
                logger.info('Starting training from epoch: 0, iter: 0.')
            

        while current_epoch < num_of_epochs:
            if opt['distributed']:
                train_sampler.set_epoch(current_epoch)
            for _, (inps, labels, img_ids, bboxes) in enumerate(train_loader):
                train_data = (inps, labels)
                a = labels['target_uv'][4]
                # for index, inp in enumerate(inps):
                #     from imageio.v2 import imwrite
                #     imwrite('/root/Improve-HPE-with-Diffusion/vis/vis_input_train/{}.png'.format(index), ((inp+0.5)*255).permute(1,2,0).cpu().numpy().astype(np.uint8))
                # val_data = (inps, labels, img_ids, bboxes)
                # test_data = (inps, labels, img_ids, bboxes)             
                
                # forward pass
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # lr stepping
                if opt['train']['scheduler_type']['reg'] != 'plateau':
                    diffusion.lr_scheduler_reg.step()
                if opt['train']['scheduler_type']['diff'] != 'plateau':
                    diffusion.lr_scheduler_diff.step()

                if args.debug:
                    print('lr', diffusion.opt_diff.param_groups[0]['lr'])
                    diffusion.opt_diff.param_groups[0]['lr'] = diffusion.opt_diff.param_groups[0]['lr'] * 2

                # log
                if current_step % opt['train']['print_freq'] == 0 and is_primary():
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}>\n '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        #tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)
                    wandb.log(logs, step=current_step, commit=False)
                    wandb.log({'lr_reg': diffusion.opt_reg.param_groups[0]['lr'],
                               'lr_diff': diffusion.opt_diff.param_groups[0]['lr']}, step=current_step, commit=False)

                # validation
                #if current_step != 0 and current_step % opt['train']['val_freq'] == 0:
                if current_step % opt['train']['val_freq'] == 0 and current_step != 0:

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    #diffusion.validate(json_path, heatmap_to_coord, test_loader, opt)
                    #diffusion.validate_gt(json_path_gt, heatmap_to_coord, val_data, opt)
                    diffusion.validate_gt(json_path_gt, heatmap_to_coord, val_loader, opt)

                    # log
                    if is_primary():
                        eval_logs = diffusion.get_current_metrics()
                        message = 'Evaluation at <epoch:{:3d}, iter:{:8,d}\n> '.format(
                            current_epoch, current_step)
                        for k, v in eval_logs.items():
                            message += '{:s}: {:.4e} '.format(k, v)
                        logger.info(message)
                        wandb.log(eval_logs, step=current_step)
                        if opt['train']['scheduler_type']['reg'] == 'plateau':
                            diffusion.lr_scheduler_reg.step(eval_logs['val_reg_loss'])
                        if opt['train']['scheduler_type']['diff'] == 'plateau':
                            diffusion.lr_scheduler_diff.step(eval_logs['val_diff_loss'])

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')

                #if current_step % opt['train']['save_checkpoint_freq'] == 0 and is_primary():
                if current_step % opt['train']['save_checkpoint_freq'] == 0 and is_primary() and current_step != 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                current_step += 1
                #break
            # end of epoch ===================================================================

            if opt['distributed']:
                dist.barrier()  # Sync

            current_epoch += 1

        # save model
        if is_primary():
            logger.info('End of training.')
    else:
        if is_primary():
            logger.info('Begin Model Evaluation.')

        diffusion.set_new_noise_schedule(
            opt['model']['beta_schedule']['val'], schedule_phase='val')
        diffusion.validate(json_path, heatmap_to_coord, test_loader, opt)
        # for inps, labels, img_ids, bboxes in train_loader:
        #     val_data = (inps, labels, img_ids, bboxes)
        #     diffusion.validate_gt(json_path_gt, heatmap_to_coord, val_data, opt)
        #     break
        diffusion.validate_gt(json_path, heatmap_to_coord, val_loader, opt)

        # log
        if is_primary():
            eval_logs = diffusion.get_current_metrics()
            message = 'Final eval:\n'
            for k, v in eval_logs.items():
                message += '{:s}: {:.4e} '.format(k, v)
            logger.info(message)


def main(opt, args):
    # launch main_worker
    if opt['world_size'] == 1:
        if opt['seed'] is not None:
            setup_seed(opt['seed'])
        main_worker(None, opt, args)
    else:
        mp.spawn(main_worker, nprocs=opt['world_size'], args=(opt, args))
    

# def sweep_main(opt, args):
#     # get sweep config
#     sweep_config = build_sweep_config()
#     opt = merge_opt(opt, sweep_config)
#     # init sweep_id
#     sweep_id = wandb.sweep(opt, project)
#     wandb.agent(function=main, count=64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/root/Improve-HPE-with-Diffusion/config/fixed_res_and_diff_debug.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                         help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
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