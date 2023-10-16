import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import os
import json
import pickle as pk
from tqdm import tqdm

import sys
sys.path.append('/root/Improve-HPE-with-Diffusion')

import model.networks as networks
from .base_model import BaseModel
from core.metrics import *
from core.nms import oks_pose_nms
logger = logging.getLogger('base')

from core.dist import is_primary
from core.optimize import build_scheduler
from core.optimize import MyExponentialLR
from data.utils import eval_data_prepare


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.netG = networks.define_G(opt)
        self.schedule_phase = None

        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        
        # set optimizer
        if self.opt['phase'] == 'train':
            self.netG.train()
            # optimizer
            if not isinstance(self.netG, DDP):
                self.optimizer = self.set_optimizer(
                    self.netG.parameters(), 
                    lr=opt['train']['lr'], 
                    opt_type=opt['train']['optimizer'])
            else:
                self.optimizer = self.set_optimizer(
                    self.netG.module.parameters(), 
                    lr=opt['train']['lr'], 
                    opt_type=opt['train']['optimizer'])
            # log dict
            self.log_dict = OrderedDict()
        self.eval_dict = OrderedDict()
        self.load_network()
        if self.opt['phase'] == 'train':
            # scheduler
            opt['train']['scheduler']['last_epoch'] = self.begin_epoch - 1  # num of iters
            self.lr_scheduler = self.set_scheduler(
                sche_type=opt['train']['scheduler_type'],
                optimizer=self.optimizer, 
                **opt['train']['scheduler']
            )
        self.print_network()

    def feed_data(self, data):
        """data {
            'cam_params': cam, 'cam_id': cam_id,
            'gt_3d': gt_3D, 'gt_2d_pixel': None,
            'action': action, 'cano_action': cano_action,
            'subject': subject, 
            'image': image,
            'scale': scale, 'bb_box': bb_box
        }
        """
        self.data = self.set_device(data)

    def optimize_parameters(self, inputs_2d, inputs_3d, epoch):
        self.optimizer.zero_grad()

        losses = self.netG(inputs_2d, inputs_3d, epoch)
        if isinstance(losses, dict):
            assert "loss" in losses
        else:
            losses = {"loss": losses}
        losses["loss"].backward()

        self.optimizer.step()

        # set log
        for k,v in losses.items():
            losses[k] = v if isinstance(v, float) else v.item()
        self.log_dict = losses

    def sample(self, inputs_2d):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, DDP):
                ret = self.netG.module.sample(inputs_2d)
            else:
                ret = self.netG.sample(inputs_2d)
            results = {'preds': ret['preds']}
            if 'residual' in ret.keys():
                results['residual'] = ret['residual']
                if self.opt['model']['diffusion']['norm_res']:
                    assert 'sigmas' in ret
                    results['final_preds'] = results['preds'] + results['residual'] * ret['sigmas']
                else:
                    results['final_preds'] = results['preds'] + results['residual']
            else:
                results['final_preds'] = results['preds']
        self.netG.train()
        return results

    def set_loss(self):
        if isinstance(self.netG, DDP):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_optimizer(self, optim_params, lr, opt_type='adam'):
        if opt_type == 'adam':
            return torch.optim.Adam(list(optim_params), lr=lr)
        elif opt_type == "adamw":
            return torch.optim.AdamW(list(optim_params), lr=lr, weight_decay=0.1)
        elif opt_type == 'sgd':
            return torch.optim.SGD(list(optim_params), lr=lr, momentum=0.9, weight_decay=0.0001)
        else:
            raise NotImplementedError
    
    def set_scheduler(self, sche_type, optimizer, **kwargs):
        return build_scheduler(sche_type=sche_type, optimizer=optimizer, **kwargs)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, DDP):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict
    
    def get_current_metrics(self):
        return self.eval_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, DDP):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step, random_state):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'best_gen.pth')
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'best_opt.pth')
        # gen
        network = self.netG
        if isinstance(self.netG, DDP):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None, "random_state": random_state}
        opt_state['optimizer'] = self.optimizer.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        self.random_state = None
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = os.path.join(load_path, 'best_gen.pth')
            opt_path = os.path.join(load_path, 'best_opt.pth')
            # gen
            network = self.netG
            if isinstance(self.netG, DDP):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=True)
            if self.opt['phase'] == 'train':
                # optimizer
                optim = torch.load(opt_path)
                self.optimizer.load_state_dict(optim['optimizer'])
                self.begin_step = optim['iter']
                self.begin_epoch = optim['epoch']
                if "random_state" in optim:
                    self.random_state = optim["random_state"]
    
    def validate(self, kps_left, kps_right, joints_left, joints_right, receptive_field, val_generator, all_actions):
        action_error_sum = define_error_list(all_actions)
        action_error_sum_diff = define_error_list(all_actions)
        for _, cano_action, batch, batch_2d in tqdm(val_generator.next_epoch()):
            inputs_3d = torch.from_numpy(batch.astype('float32'))
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            ##### apply test-time-augmentation (following Videopose3d)
            inputs_2d_flip = inputs_2d.clone()
            inputs_2d_flip[:, :, :, 0] *= -1
            inputs_2d_flip[:, :, kps_left + kps_right, :] = inputs_2d_flip[:, :, kps_right + kps_left, :]
            ##### convert size
            inputs_3d_p = inputs_3d
            inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d_p)
            inputs_2d_flip, _ = eval_data_prepare(receptive_field, inputs_2d_flip, inputs_3d_p)

            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()
                inputs_2d_flip = inputs_2d_flip.cuda()
            inputs_3d[:, :, 0] = 0

            results = self.sample(inputs_2d)
            results_flip = self.sample(inputs_2d_flip)

            # ensemble raw predictions
            results_flip['preds'][:, :, :, 0] *= -1
            results_flip['preds'][:, :, joints_left + joints_right, :] = results_flip['preds'][:, :, joints_right + joints_left, :]
            for i in range(results["preds"].shape[0]):
                results["preds"][i] = (results["preds"][i] + results_flip["preds"][i]) / 2
            # ensemble final predictions
            results_flip['final_preds'][:, :, :, 0] *= -1
            results_flip['final_preds'][:, :, joints_left + joints_right, :] = results_flip['final_preds'][:, :, joints_right + joints_left, :]
            for i in range(results["final_preds"].shape[0]):
                results["final_preds"][i] = (results["final_preds"][i] + results_flip["final_preds"][i]) / 2

            # compute metrics
            action_error_sum = mpjpe_by_action(results['preds'], 
                                               inputs_3d, 
                                               cano_action, 
                                               action_error_sum)
            action_error_sum_diff = mpjpe_by_action(results['final_preds'], 
                                               inputs_3d, 
                                               cano_action, 
                                               action_error_sum_diff)
        # average across actions
        action_error_sum.update({
            "Average": {
                "p1": AccumLoss(),
                "p2": AccumLoss()
            }
        })
        action_error_sum_diff.update({
            "Average": {
                "p1": AccumLoss(),
                "p2": AccumLoss()
            }
        })

        for k, v in action_error_sum.items():
            if k == "Average":
                continue
            else:
                action_error_sum["Average"]["p1"].update(v["p1"].avg, 1)
                action_error_sum["Average"]["p2"].update(v["p2"].avg, 1)
        for k, v in action_error_sum_diff.items():
            if k == "Average":
                continue
            else:
                action_error_sum_diff["Average"]["p1"].update(v["p1"].avg, 1)
                action_error_sum_diff["Average"]["p2"].update(v["p2"].avg, 1)        

        # log eval metrics
        for k in action_error_sum.keys():
            self.eval_dict[k+"_p1"] = action_error_sum[k]["p1"].avg * 1000
            self.eval_dict[k+"_p2"] = action_error_sum[k]["p2"].avg * 1000
            self.eval_dict[k+"_diff_p1"] = action_error_sum_diff[k]["p1"].avg * 1000
            self.eval_dict[k+"_diff_p2"] = action_error_sum_diff[k]["p2"].avg * 1000

        return