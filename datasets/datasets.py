import os
import numpy as np
from torch.utils.data import Dataset

import sys
sys.path.append('/root/Improve-HPE-with-Diffusion')
from data.camera import world_to_camera, normalize_screen_coordinates
from data.generator import ChunkedGenerator, UnchunkedGenerator
from data.h36m_dataset import Human36mDataset
# registry
from core.registry import Registry
DATASET_REGISTRY = Registry('dataset')


@DATASET_REGISTRY.register()
class LazyDataset(Dataset):
    """
    Directly loading batched backbone features, residuals and 2d keypoints,
    used for faster training.
    """
    def __init__(self, data_root, train) -> None:
        super().__init__()
        self.train = train
        self.all_data = np.load(data_root, allow_pickle=True)["pretrained_items"].item()
        
    def fetch(self):
        self._data = self.fetch()
        for i in range(len(self.all_data["feature"])):
            if self.train:
                b = self.all_data["feature"][i].shape[0]
                for j in range(b):  
                    self._data.append((self.all_data["feature"][i][j], 
                                    self.all_data["residual"][i][j], 
                                    self.all_data["keypoints_2d"][i][j]))
            else:
                self._data.append((self.all_data["feature"][i], 
                                self.all_data["residual"][i], 
                                self.all_data["keypoints_2d"][i],
                                self.all_data["action"][i]))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

@DATASET_REGISTRY.register()
class H36M(Dataset):
    def __init__(self, opt, train=True) -> None:
        super().__init__()
        
        self.train = train
        self.keypoints_type_2d = opt["keypoints_type_2d"]
        self.root_path = opt["root"]
        self.keypoints_3d_path = os.path.join(self.root_path, "data_3d_h36m.npz")
        self.keypoints_2d_path = os.path.join(self.root_path, 'data_2d_h36m' + '_' + self.keypoints_type_2d + '.npz')
        self.batch_size = opt["batch_size"]
        self.train_list = ["S1", "S5", "S6", "S7", "S8"]
        self.test_list = ["S9", "S11"]
        self.train_downsample = opt["train_downsample"]
        self.val_downsample = opt["val_downsample"]
        self.aug = opt["augmentation"]
        self.test_aug = opt["test_augmentation"]
        self.pad = opt["pad"]
        self.action_filter = opt["action_filter"]
        self.chunk_length = opt["num_frames"]
        self.num_joints = opt["num_joints"]
        self.kps_left = None
        self.kps_right = None
        self.joints_left = None
        self.joints_right = None

        self.dataset = Human36mDataset(path=self.keypoints_3d_path, remove_static_joints=opt["remove_static_joints"])

        if self.train:
            self.keypoints = self.prepare_data(self.train_list)
            self.cameras_train, self.poses_train, self.poses_train_2d, self.actions_train = self.fetch(self.train_list, self.train_downsample)
        else:
            self.keypoints = self.prepare_data(self.test_list)
            self.cameras_test, self.poses_test, self.poses_test_2d, self.actions_test = self.fetch(self.test_list, self.val_downsample)                              

    def prepare_data(self, folder_list):
        # prepare 3D kp annotations in camera frame
        for subject in folder_list:
            for action in self.dataset[subject].keys():
                anim = self.dataset[subject][action]

                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    pos_3d[:, 1:] = pos_3d[:, 1:] - pos_3d[:, :1] 
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d

        self.joints_left, self.joints_right = list(self.dataset.skeleton().joints_left()), list(self.dataset.skeleton().joints_right())
        # prepare 2D kp annotations according to 'keypoints_type_2d'(gt or CPN or ...)
        keypoints = np.load(self.keypoints_2d_path, allow_pickle=True)
        keypoints_symmetry = keypoints['metadata'].item()['keypoints_symmetry']

        self.kps_left, self.kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
        keypoints = keypoints['positions_2d'].item()

        for subject in folder_list:
            assert subject in keypoints, 'Subject {} is missing from the 2D detections self.dataset'.format(subject)
            for action in self.dataset[subject].keys():
                assert action in keypoints[
                    subject], 'Action {} of subject {} is missing from the 2D detections self.dataset'.format(action,
                                                                                                        subject)
                for cam_idx in range(len(keypoints[subject][action])):

                    mocap_length = self.dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                    assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

                    if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                        keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

        for subject in keypoints.keys():
            for action in keypoints[subject]:
                for cam_idx, kps in enumerate(keypoints[subject][action]):
                    cam = self.dataset.cameras()[subject][cam_idx]
                    kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                    keypoints[subject][action][cam_idx] = kps
        
        return keypoints

    def fetch(self, subjects, downsample):
        out_poses_3d = {}
        out_poses_2d = {}
        out_camera_params = {}
        out_actions = {}
        
        for subject in subjects:
            keys = self.dataset[subject].keys()
            for action in self.dataset[subject].keys():
                if self.action_filter:
                    found = False
                    # for a in self.action_filter:
                    for a in self.dataset.all_actions:
                        if action.startswith(a):
                            found = True
                            break
                    if not found:
                        continue
                
                poses_3d = self.dataset[subject][action]['positions_3d']
                for i in range(len(poses_3d)): 
                    out_poses_3d[(subject, action, i)] = poses_3d[i]
                    out_actions[(subject, action, i)] = action.split(" ")[0]

                if subject in self.dataset.cameras():
                    cams = self.dataset.cameras()[subject]
                    assert len(cams) == len(poses_3d), 'Camera count mismatch'       # len(cams) == len(poses_3d) == 4
                    for i, cam in enumerate(cams):
                        if 'intrinsic' in cam:
                            c = cam['intrinsic']
                            out_camera_params[(subject, action, i)] = cam['intrinsic']

                poses_2d = self.keypoints[subject][action]
                assert len(poses_2d) == len(poses_3d), 'Camera count mismatch'  # len(poses_3d) = len(poses_2d) = 4
                for i in range(len(poses_2d)): 
                    out_poses_2d[(subject, action, i)] = poses_2d[i]

        if len(out_camera_params) == 0:
            out_camera_params = None

        stride = downsample
        
        if stride > 1:
            for key in out_poses_3d.keys():
                out_poses_3d[key] = out_poses_3d[key][::stride]
                out_poses_2d[key] = out_poses_2d[key][::stride]

        return out_camera_params, out_poses_3d, out_poses_2d, out_actions
    
    def get_generator(self):
        if self.train:
            self.generator = \
            ChunkedGenerator(self.batch_size // self.chunk_length, self.chunk_length,
                            self.poses_train, self.poses_train_2d,
                            self.num_joints,
                            cameras=self.cameras_train, actions=self.actions_train,
                            pad=self.pad,
                            shuffle=True,
                            augment=self.aug,
                            kps_left=self.kps_left, kps_right=self.kps_right,
                            joints_left=self.joints_left, joints_right=self.joints_right)
            print('INFO: Training on {} frames'.format(self.generator.num_frames()))
            return self.generator
        else:
            self.generator = \
            UnchunkedGenerator(self.poses_test, self.poses_test_2d, 
                            cameras=self.cameras_test, 
                            actions=self.actions_test,
                            pad=self.pad, 
                            augment=self.test_aug, 
                            kps_left=self.kps_left, kps_right=self.kps_right, 
                            joints_left=self.joints_left, joints_right=self.joints_right)
            print('INFO: Testing on {} frames'.format(self.generator.num_frames()))
        return self.generator

    def get_actions(self):
        return self.dataset.all_actions