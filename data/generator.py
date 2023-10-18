import numpy as np
import cv2
from random import random
from itertools import zip_longest
import torch

import sys
sys.path.append("/root/Improve-HPE-with-Diffusion")
from data.utils import eval_data_prepare

PIXEL_MEAN = [0.406, 0.457, 0.480]
PIXEL_STD = [0.225, 0.224, 0.229]
BATCH_2D_DIM = 2
BATCH_3D_DIM = 3
CAMERA_DIM = 9

class ChunkedGenerator:
    """
    Batched data generator, used for training.
    The sequences are split into equal-length chunks and padded as necessary.
    
    Arguments:
    batch_size -- the batch size to use for training
    cameras -- dict of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- dict of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- dict of input 2D keypoints, one element for each video
    chunk_length -- number of output frames to predict for each training example (usually 1)
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    shuffle -- randomly shuffle the dataset before each epoch
    random_seed -- initial seed to use for the random generator
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    def __init__(self, 
                 batch_size, 
                 chunk_length, 
                 poses_3d, poses_2d,
                 num_joints, 
                 cameras=None,
                 actions=None, 
                 pad=0,
                 shuffle=False, random_seed=1234,
                 augment=False,
                 kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 ):
        assert cameras is None or len(cameras) == len(poses_3d), (len(cameras), len(poses_3d))
        assert actions is None or len(actions) == len(poses_3d), (len(actions), len(poses_3d))
        assert len(poses_2d) == len(poses_3d), (len(poses_2d), len(poses_3d))
        
        pairs = []  # (seq_idx, start_frame, end_frame, flip) tuples
        for key in poses_2d.keys():
            """
            poses_3d[key], (n_frames, n_kps, n_dims) 
            poses_2d[key], (n_frames, n_kps, n_dims)
            """
            assert poses_2d[key].shape[0] == poses_3d[key].shape[0]
            n_chunks = (poses_2d[key].shape[0] + chunk_length - 1) // chunk_length  # how many chunks can a video be devided into
            offset = (n_chunks * chunk_length - poses_2d[key].shape[0]) // 2
            bounds = np.arange(n_chunks + 1) * chunk_length - offset
            augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            keys = np.tile(np.array(key).reshape([1,3]), (len(bounds - 1), 1))  # (len(bounds-1),3) == (N,3)
            pairs += list(zip(keys, bounds[:-1], bounds[1:], augment_vector))
            if augment:
                pairs += zip(keys, bounds[:-1], bounds[1:], ~augment_vector)

        l = len(pairs)
        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.chunk_length = chunk_length
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.state = None
        self.num_joints = num_joints

        self.cameras = cameras
        self.actions = actions
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d
        self.batch_2d = np.empty((batch_size, chunk_length, num_joints, BATCH_2D_DIM))
        self.batch_3d = np.empty((batch_size, chunk_length, num_joints, BATCH_3D_DIM))
        if cameras is not None:
            self.batch_cam = np.empty((batch_size, CAMERA_DIM))
        if actions is not None:
            self.batch_action = []
        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right

    def num_frames(self):
        return self.num_batches * self.batch_size * self.chunk_length 
        # let num_frames be divsible by batch_size

    def random_state(self):
        return self.random

    def set_random_state(self, random):
        self.random = random

    def augment_enabled(self):
        return self.augment

    def next_pairs(self):
        if self.shuffle:
            pairs = self.random.permutation(self.pairs)
        else:
            pairs = self.pairs
        return 0, pairs

    def next_epoch(self):
        start_idx, pairs = self.next_pairs()
        for b_i in range(start_idx, self.num_batches):  # batch_id
            chunks = pairs[b_i*self.batch_size: (b_i+1)*self.batch_size]
            for i, (key, start_3d, end_3d, flip) in enumerate(chunks):
                subject, action, cam_index = key
                cam_index = int(cam_index)
                start_2d = start_3d
                end_2d = end_3d

                # 2D poses
                seq_2d = self.poses_2d[(subject, action, cam_index)]
                low_2d = max(0, start_2d)
                high_2d = min(end_2d, seq_2d.shape[0])
                pad_left_2d = low_2d - start_2d
                pad_right_2d = end_2d - high_2d
                if pad_left_2d != 0 or pad_right_2d != 0:
                    self.batch_2d[i] = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0,0), (0,0)), "edge")
                else:
                    self.batch_2d[i] = seq_2d[low_2d:high_2d]

                if flip:
                    # Flip 2D keypoints
                    self.batch_2d[i, :, :, 0] *= -1
                    self.batch_2d[i, :, self.kps_left + self.kps_right] = self.batch_2d[i, :, self.kps_right + self.kps_left]

                seq_3d = self.poses_3d[(subject, action, cam_index)]
                low_3d = max(0, start_3d)
                high_3d = min(end_3d, seq_3d.shape[0])
                pad_left_3d = low_3d - start_3d
                pad_right_3d = end_3d - high_3d
                if pad_left_3d != 0 or pad_right_3d != 0:
                    self.batch_3d[i] = np.pad(seq_3d[low_3d:high_3d], ((pad_left_3d, pad_right_3d), (0,0), (0,0)), "edge")
                else:
                    self.batch_3d[i] = seq_3d[low_3d:high_3d]
                if flip:
                    # Flip 3D joints
                    self.batch_3d[i, :, :, 0] *= -1
                    self.batch_3d[i, :, self.joints_left + self.joints_right] = \
                            self.batch_3d[i, :, self.joints_right + self.joints_left]
                    
                # Cameras
                if self.cameras is not None:
                    self.batch_cam[i] = self.cameras[(subject, action, cam_index)]
                    if flip:
                        # Flip horizontal distortion coefficients
                        self.batch_cam[i, 2] *= -1
                        self.batch_cam[i, 7] *= -1

                # Canonical Actions
                if self.actions is not None:
                    self.batch_action.append(self.actions[(subject, action, cam_index)])

            if self.cameras is None and self.actions is None:
                yield None, None, self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)]
            elif self.actions is None:
                yield self.batch_cam[:len(chunks)], None, self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)]
            else:
                yield self.batch_cam[:len(chunks)], self.batch_action, self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)]
    

class UnchunkedGenerator:
    """
    Non-batched data generator, used for testing.
    Sequences are returned one at a time (i.e. batch size = 1), without chunking.
    
    If data augmentation is enabled, the batches contain two sequences (i.e. batch size = 2),
    the second of which is a mirrored version of the first.
    
    Arguments:
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    
    def __init__(self, poses_3d, poses_2d, cameras, actions, pad=0,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None):
        assert poses_3d is None or len(poses_3d) == len(poses_2d)
        assert cameras is None or len(cameras) == len(poses_2d)

        self.augment = False
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        
        self.pad = pad
        self.cameras = cameras
        self.actions = actions
        self.poses_2d = poses_2d
        self.poses_3d = poses_3d
        
    def num_frames(self):
        count = 0
        for p in self.poses_2d.values():
            count += p.shape[0]
        return count
    
    def augment_enabled(self):
        return self.augment
    
    def set_augment(self, augment):
        self.augment = augment
    
    def next_epoch(self):
        for key in self.poses_2d:
            assert self.poses_2d[key].shape[0] == self.poses_3d[key].shape[0]
            batch_cam = None if self.cameras is None else np.expand_dims(self.cameras[key], axis=0)
            batch_action = None if self.actions is None else self.actions[key]
            batch_3d = np.expand_dims(self.poses_3d[key], axis=0)
            batch_2d = np.expand_dims(self.poses_2d[key], axis=0)
            
            if self.augment:
                # Append flipped version
                if batch_cam is not None:
                    batch_cam = np.concatenate((batch_cam, batch_cam), axis=0)
                    batch_cam[1, 2] *= -1
                    batch_cam[1, 7] *= -1
                
                if batch_3d is not None:
                    batch_3d = np.concatenate((batch_3d, batch_3d), axis=0)
                    batch_3d[1, :, :, 0] *= -1
                    batch_3d[1, :, self.joints_left + self.joints_right] = batch_3d[1, :, self.joints_right + self.joints_left]

                batch_2d = np.concatenate((batch_2d, batch_2d), axis=0)
                batch_2d[1, :, :, 0] *= -1
                batch_2d[1, :, self.kps_left + self.kps_right] = batch_2d[1, :, self.kps_right + self.kps_left]
            
            yield batch_cam, batch_action, batch_3d, batch_2d


class PreparedGenerator:
    def __init__(self, train, generator, model, batch_size, shuffle=False, random_seed=1234,
                 kps_left=None, kps_right=None, joints_left=None, joints_right=None, receptive_field=243) -> None:
        self.train = train
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        if self.train:
            self.shuffle = shuffle
            self.pairs = self.gen_train(generator, model)
        else:
            self.shuffle = False
            self.pairs = self.gen_val(generator, model, kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right, receptive_field=receptive_field)
        print("Pretrained features and residuals prepared.")
    
    def random_state(self):
        return self.random

    def set_random_state(self, random):
        self.random = random

    def num_seqs(self):
        return (len(self.pairs))

    def gen_train(self, generator, model):
        pairs = []
        assert self.generator.shuffle == False
        for _, _, batch_3d, batch_2d in enumerate(generator.next_epoch()):
            inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()
            inputs_3d[:, :, 0] = 0

            with torch.no_grad():
                predicted_3d, feats = model(inputs_2d)
            residual = inputs_3d - predicted_3d

            assert feats.ndim == 4 and residual.ndim == 4 and inputs_2d.ndim == 4
            assert feats.shape[:-1] == residual.shape[:-1] and feats.shape[:-1] == inputs_2d.shape[:-1]

            for i in feats.shape[0]:  # b
                self.pairs.append((residual[i].detach().cpu().numpy(),
                                feats[i].detach().cpu().numpy(),
                                inputs_2d[i].detach().cpu().numpy()))
        return pairs
    
    def gen_val(self, generator, model, kps_left, kps_right, joints_left, joints_right, receptive_field):
        for _, cano_action, batch, batch_2d in enumerate(generator.next_epoch()):
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

            with torch.no_grad():
                predicted_3d, feats = model(inputs_2d)
                predicted_3d_flip, feats_flip = model(inputs_2d_flip)
            residual = inputs_3d - predicted_3d
            predicted_3d_flip[:, :, :, 0] *= -1
            predicted_3d_flip[:, :, joints_left + joints_right, :] = predicted_3d_flip[:, :, joints_right + joints_left, :]
            residual_flip = inputs_3d - predicted_3d_flip

            output["residual"][2*i] = residual.detach().cpu().numpy()
            output["feature"][2*i] = feats.detach().cpu().numpy()
            output["keypoints_2d"][2*i] = inputs_2d.detach().cpu().numpy()
            output["action"][2*i] = cano_action
            output["residual"][2*i+1] = residual_flip.detach().cpu().numpy()
            output["feature"][2*i+1] = feats_flip.detach().cpu().numpy()
            output["keypoints_2d"][2*i+1] = inputs_2d_flip.detach().cpu().numpy()
            output["action"][2*i+1] = cano_action

    def next_epoch(self):
        if self.train:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            assert self.num_seqs() % self.batch_size == 0
            for b_i in range(int(self.num_seqs() // self.batch_size)):  # batch_id
                chunks = pairs[b_i*self.batch_size: (b_i+1)*self.batch_size]
                batch_res = np.stack([x[0] for x in chunks], axis=0)
                batch_feats = np.stack([x[1] for x in chunks], axis=0)
                batch_2d = np.stack([x[2] for x in chunks], axis=0)
                yield batch_res, batch_feats, batch_2d
        else:
            pass
        