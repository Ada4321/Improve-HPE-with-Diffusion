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

import sys
sys.path.append('/root/Improve-HPE-with-Diffusion')

import model.networks as networks
from .base_model import BaseModel
from core.metrics import evaluate_mAP
from core.nms import oks_pose_nms
logger = logging.getLogger('base')

from core.dist import is_primary
from core.optimize import build_scheduler
from core.optimize import MyExponentialLR


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.netG = networks.define_G(opt)
        # self.netG = self.set_device(self.netG, gpu)
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
                self.opt_reg = self.set_optimizer(
                    self.netG.regressor.parameters(), 
                    lr=opt['train']['lr']['reg'], 
                    opt_type=opt['train']['optimizer']['reg'])
                self.opt_diff = self.set_optimizer(
                    self.netG.denoise_fn.parameters(), 
                    lr=opt['train']['lr']['diff'], 
                    opt_type=opt['train']['optimizer']['diff'])
            else:
                self.opt_reg = self.set_optimizer(
                    self.netG.module.regressor.parameters(), 
                    lr=opt['train']['lr']['reg'], 
                    opt_type=opt['train']['optimizer']['reg'])
                self.opt_diff = self.set_optimizer(
                    self.netG.module.denoise_fn.parameters(), 
                    lr=opt['train']['lr']['diff'], 
                    opt_type=opt['train']['optimizer']['diff'])
            # log dict
            self.log_dict = OrderedDict()
        self.eval_dict = OrderedDict()
        self.load_network()
        if self.opt['phase'] == 'train':
            # scheduler
            opt['train']['scheduler']['reg']['last_epoch'] = self.begin_epoch * 140 - 1
            opt['train']['scheduler']['diff']['last_epoch']= self.begin_epoch * 140 - 1  # num of iters
            self.lr_scheduler_reg = self.set_scheduler(
                sche_type=opt['train']['scheduler_type']['reg'],
                optimizer=self.opt_reg, 
                **opt['train']['scheduler']['reg']
            )
            self.lr_scheduler_diff = self.set_scheduler(
                sche_type=opt['train']['scheduler_type']['diff'],
                optimizer=self.opt_diff, 
                **opt['train']['scheduler']['diff']
            ) 
            # self.lr_scheduler_reg = MyExponentialLR(
            #     self.opt_reg, decay_epochs=opt['train']['decay_epochs'], gamma=opt['train']['lr_factor'], last_epoch=self.begin_epoch-1
            # )
            # self.lr_scheduler_diff = MyExponentialLR(
            #     self.opt_diff, decay_epochs=opt['train']['decay_epochs'], gamma=opt['train']['lr_factor'], last_epoch=self.begin_epoch-1
            # )
        self.print_network()

    def feed_data(self, data):
        """
        data -- a dict
        {
            'images': images to detect kps
            'gt_kps': ground truth key points
        }
        """
        # self.data = self.set_device(data)
        self.data = {}
        if isinstance(data, tuple):
            inps, labels = data
            inps = self.set_device(inps)
            for k, _ in labels.items():
                if k == 'type':
                    continue
                labels[k] = self.set_device(labels[k])

            self.data['images'] = inps
            self.data['gt_kps'] = labels['target_uv']
        else:
            data = self.set_device(data)
            self.data['images'] = data


    def optimize_parameters(self):
        self.opt_reg.zero_grad()
        self.opt_diff.zero_grad()

        losses = self.netG(self.data['images'], self.data['gt_kps'])
        
        if 'reg_loss' in losses:
            losses['reg_loss'].backward()
            self.opt_reg.step()  
        if 'diff_loss' in losses:
            losses['diff_loss'].backward()
            self.opt_diff.step()
        #dist.barrier()

        # set log
        for k,v in losses.items():
            losses[k] = v.item()
        self.log_dict = losses

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, DDP):
                ret = self.netG.module.sample(self.data['images'])
            else:
                ret = self.netG.sample(self.data['images'])
            self.results = {'preds': ret['preds']}
            self.results['preds']['pred_jts'] = ret['preds']['raw_pred_jts']
            if 'res' in ret.keys():
                self.results['res'] = ret['res']
                self.results['preds']['pred_jts'] += ret['res']
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, DDP):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_optimizer(self, optim_params, lr, opt_type='adam'):
        if opt_type == 'adam':
            return torch.optim.Adam(list(optim_params), lr=lr)
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

    def save_network(self, epoch, iter_step):
        # gen_path = os.path.join(
        #     self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        # opt_path = os.path.join(
        #     self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'last_gen.pth')
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'last_opt.pth')
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
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer_reg'] = self.opt_reg.state_dict()
        opt_state['optimizer_diff'] = self.opt_diff.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = os.path.join(load_path, 'last_gen.pth')
            opt_path = os.path.join(load_path, 'last_opt.pth')
            # gen
            network = self.netG
            if isinstance(self.netG, DDP):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=(not self.opt['model']['finetune_norm']))
            # network.load_state_dict(torch.load(
            #     gen_path), strict=False)
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.opt_reg.load_state_dict(opt['optimizer_reg'])
                self.opt_diff.load_state_dict(opt['optimizer_diff'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']

    def validate(self, json_path, heatmap_to_coord, val_loader, opt):
        kpt_json = []

        for inps, crop_bboxes, bboxes, img_ids, scores, _, _ in val_loader:
            self.feed_data(inps)
            self.test()
            
            # compute metrics
            for i in range(inps.shape[0]):
                bbox = crop_bboxes[i].tolist()
                raw_preds = {k:v.reshape(inps.shape[0], opt['data_preset']['num_joints'], -1) for k,v in self.results['preds'].items()}
                pose_coords, pose_scores = heatmap_to_coord(raw_preds, bbox, idx=i)
                if pose_scores is None:
                    pose_scores = np.zeros((pose_coords.shape[0], pose_coords.shape[1], 1))    
                
                keypoints = np.concatenate((pose_coords[0], pose_scores[0]), axis=1)
                keypoints = keypoints.reshape(-1).tolist()

                data = dict()
                data['bbox'] = bboxes[i, 0].tolist()
                data['image_id'] = int(img_ids[i])
                data['score'] = float(scores[i] + np.mean(pose_scores) + np.max(pose_scores))
                #data['score'] = float(scores[i])
                data['category_id'] = 1
                data['keypoints'] = keypoints
                data['area'] = float((crop_bboxes[i][2] - crop_bboxes[i][0]) * (crop_bboxes[i][3] - crop_bboxes[i][1]))

                kpt_json.append(data)
            #break

        if isinstance(self.netG, DDP):
            with open(os.path.join(json_path, 'test_kpt_rank_{}.pkl'.format(opt['current_id'] if opt['current_id'] is not None else 0)), 'wb') as fid:
                pk.dump(kpt_json, fid, pk.HIGHEST_PROTOCOL)

            dist.barrier()  # Make sure all JSON files are saved

            if is_primary():
                kpt_json_all = []
                for r in range(opt['world_size']):
                    with open(os.path.join(json_path, f'test_kpt_rank_{r}.pkl'), 'rb') as fid:
                        kpt_pred = pk.load(fid)

                    os.remove(os.path.join(json_path, f'test_kpt_rank_{r}.pkl'))
                    kpt_json_all += kpt_pred

                kpt_json_all = oks_pose_nms(kpt_json_all)

                with open(os.path.join(json_path, 'result.json'), 'w') as fid:
                    json.dump(kpt_json_all, fid)
        else:
            with open(os.path.join(json_path, 'result.json'), 'w') as fid:
                json.dump(kpt_json, fid)

        res = evaluate_mAP(os.path.join(json_path, 'result.json'), ann_type='keypoints')
        self.eval_dict['det_AP'] = res['AP']
        self.eval_dict['det_AP50'] = res['Ap .5']
        self.eval_dict['det_AP75'] = res['AP .75']

    
    def validate_gt(self, json_path, heatmap_to_coord, val_loader, opt):
        kpt_json = []
        
        for inps, labels, img_ids, bboxes in val_loader:
            self.feed_data((inps, labels))
            self.test()
            with torch.no_grad():
                # val losses
                val_losses = {
                    'val_reg_loss': torch.nn.L1Loss(reduction='sum')(self.results['preds']['pred_jts'], self.data['gt_kps']).item(),
                    #'val_diff_loss': torch.nn.MSELoss(self.results['preds'][''], self.data['gt_kps']-self.raw_kpt).item()
                }
                if 'res' in self.results.keys():
                    val_losses['val_diff_loss'] = torch.nn.MSELoss(reduction='sum')(self.results['res'], self.data['gt_kps']-self.results['preds']['raw_pred_jts']).item() / inps.shape[0]
            # compute metrics
            for i in range(inps.shape[0]):
                bbox = bboxes[i].tolist()
                raw_preds = {k:v.reshape(inps.shape[0], opt['data_preset']['num_joints'], -1) for k,v in self.results['preds'].items()}
                pose_coords, pose_scores = heatmap_to_coord(raw_preds, bbox, idx=i)
                if pose_scores is None:
                    pose_scores = np.ones((pose_coords.shape[0], pose_coords.shape[1], 1))

                keypoints = np.concatenate((pose_coords[0], pose_scores[0]), axis=1)
                keypoints = keypoints.reshape(-1).tolist()

                data = dict()
                data['bbox'] = bboxes[i].tolist()
                data['image_id'] = int(img_ids[i])
                data['score'] = float(np.mean(pose_scores) + np.max(pose_scores))
                data['category_id'] = 1
                data['keypoints'] = keypoints

                kpt_json.append(data)
            #break

        if isinstance(self.netG, DDP):
            with open(os.path.join(json_path, 'test_gt_kpt_rank_{}.pkl'.format(opt['current_id'] if opt['current_id'] is not None else 0)), 'wb') as fid:
                pk.dump(kpt_json, fid, pk.HIGHEST_PROTOCOL)

            dist.barrier()  # Make sure all JSON files are saved

            if is_primary():
                kpt_json_all = []
                for r in range(opt['world_size']):
                    with open(os.path.join(json_path, f'test_gt_kpt_rank_{r}.pkl'), 'rb') as fid:
                        kpt_pred = pk.load(fid)

                    os.remove(os.path.join(json_path, f'test_gt_kpt_rank_{r}.pkl'))
                    kpt_json_all += kpt_pred
                
                with open(os.path.join(json_path, 'result.json'), 'w') as fid:
                    json.dump(kpt_json_all, fid)
        else:
            with open(os.path.join(json_path, 'result.json'), 'w') as fid:
                json.dump(kpt_json, fid)

        res = evaluate_mAP(os.path.join(json_path, 'result.json'), ann_type='keypoints')
        self.eval_dict['GT_AP'] = res['AP']
        self.eval_dict['GT_AP50'] = res['Ap .5']
        self.eval_dict['GT_AP75'] = res['AP .75']
        self.eval_dict.update(val_losses)