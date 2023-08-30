"""MS COCO Human keypoint dataset."""
import json
import os
import copy
import pickle as pk
from abc import abstractmethod, abstractproperty
from imageio import imread
import cv2
import torch
import torch.utils.data as data
import numpy as np
import scipy.misc
import random
from easydict import EasyDict
from .custom import CustomDataset
import sys
sys.path.append('/root/Improve-HPE-with-Diffusion')

from rlepose.utils.presets import SimpleTransform
from rlepose.utils.presets import SimpleTransform3D
from rlepose.utils.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy
from rlepose.utils.pose_utils import (cam2pixel, pixel2cam,
                                      reconstruction_error, vis_keypoints,
                                      world2cam)
#from rlepose.models.builder import DATASET

from core.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy
from pycocotools.coco import COCO

# registry
from core.registry import Registry
DATASET_REGISTRY = Registry('dataset')

# h36m_mpii
s_mpii_2_hm36_jt = [6, 2, 1, 0, 3, 4, 5, -1, 8,
                    -1, 9, 13, 14, 15, 12, 11, 10, 7]
s_36_jt_num = 18


class CustomDataset(data.Dataset):
    """Custom dataset.
    Annotation file must be in `coco` format.

    Parameters
    ----------
    train: bool, default is True
        If true, will set as training mode.
    skip_empty: bool, default is False
        Whether skip entire image if no valid label is found.
    cfg: dict, dataset configuration.
    """

    CLASSES = None

    def __init__(self,
                 train=True,
                 skip_empty=True,
                 lazy_import=False,
                 **cfg):

        self._cfg = cfg
        self._preset_cfg = cfg['preset']
        self._root = cfg['root']
        self._img_prefix = cfg['img_prefix']
        self._ann_file = os.path.join(self._root, cfg['ann'])

        self._lazy_import = lazy_import
        self._skip_empty = skip_empty
        self._train = train

        if 'aug' in cfg.keys():
            self._scale_factor = cfg['aug']['scale_factor']
            self._rot = cfg['aug']['rot_factor']
            self.num_joints_half_body = cfg['aug']['num_joints_half_body']
            self.prob_half_body = cfg['aug']['prob_half_body']
        else:
            self._scale_factor = 0
            self._rot = 0
            self.num_joints_half_body = -1
            self.prob_half_body = -1

        self._input_size = self._preset_cfg['image_size']
        self._output_size = self._preset_cfg['heatmap_size']

        self._sigma = self._preset_cfg['sigma']

        self._check_centers = False

        self.num_class = len(self.CLASSES)

        self._loss_type = cfg['heatmap2coord']

        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.lower_body_ids = (11, 12, 13, 14, 15, 16)

        if self._preset_cfg['type'] == 'simple':
            self.transformation = SimpleTransform(
                self, scale_factor=self._scale_factor,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=self._rot, sigma=self._sigma,
                train=self._train, loss_type=self._loss_type)
        else:
            raise NotImplementedError

        self._items, self._labels = self._lazy_load_json()

    def __getitem__(self, idx):
        # get image id
        img_path = self._items[idx]
        img_id = int(os.path.splitext(os.path.basename(img_path))[0])

        # load ground truth, including bbox, keypoints, image size
        label = copy.deepcopy(self._labels[idx])

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # transform ground truth into training label and apply data augmentation
        target = self.transformation(img, label)

        img = target.pop('image')
        bbox = target.pop('bbox')
        return img, target, img_id, bbox

    def __len__(self):
        return len(self._items)

    def _lazy_load_ann_file(self):
        if os.path.exists(self._ann_file + '.pkl') and self._lazy_import:
            print('Lazy load json...')
            with open(self._ann_file + '.pkl', 'rb') as fid:
                return pk.load(fid)
        else:
            _database = COCO(self._ann_file)
            if os.access(self._ann_file + '.pkl', os.W_OK):
                with open(self._ann_file + '.pkl', 'wb') as fid:
                    pk.dump(_database, fid, pk.HIGHEST_PROTOCOL)
            return _database

    def _lazy_load_json(self):
        if os.path.exists(self._ann_file + '_annot_keypoint.pkl') and self._lazy_import:
            print('Lazy load annot...')
            with open(self._ann_file + '_annot_keypoint.pkl', 'rb') as fid:
                items, labels = pk.load(fid)
        else:
            items, labels = self._load_jsons()
            if os.access(self._ann_file + '_annot_keypoint.pkl', os.W_OK):
                with open(self._ann_file + '_annot_keypoint.pkl', 'wb') as fid:
                    pk.dump((items, labels), fid, pk.HIGHEST_PROTOCOL)

        return items, labels

    @abstractmethod
    def _load_jsons(self):
        pass

    @abstractproperty
    def CLASSES(self):
        return None

    @abstractproperty
    def num_joints(self):
        return None

    @abstractproperty
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return None


@DATASET_REGISTRY.register()
class Mscoco(CustomDataset):
    """ COCO Person dataset.

    Parameters
    ----------
    train: bool, default is True
        If true, will set as training mode.
    skip_empty: bool, default is False
        Whether skip entire image if no valid label is found. Use `False` if this dataset is
        for validation to avoid COCO metric error.
    """
    CLASSES = ['person']
    EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    num_joints = 17
    joint_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                   [9, 10], [11, 12], [13, 14], [15, 16]]
    joints_name = ('nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',    # 4
                   'left_shoulder', 'right_shoulder',                           # 6
                   'left_elbow', 'right_elbow',                                 # 8
                   'left_wrist', 'right_wrist',                                 # 10
                   'left_hip', 'right_hip',                                     # 12
                   'left_knee', 'right_knee',                                   # 14
                   'left_ankle', 'right_ankle')                                 # 16

    def _load_jsons(self):
        """Load all image paths and labels from JSON annotation files into buffer."""
        items = []
        labels = []

        _coco = self._lazy_load_ann_file()
        classes = [c['name'] for c in _coco.loadCats(_coco.getCatIds())]
        assert classes == self.CLASSES, "Incompatible category names with COCO. "

        self.json_id_to_contiguous = {
            v: k for k, v in enumerate(_coco.getCatIds())}

        # iterate through the annotations
        image_ids = sorted(_coco.getImgIds())
        for entry in _coco.loadImgs(image_ids):
            dirname, filename = entry['coco_url'].split('/')[-2:]
            # print(entry['coco_url'])
            # print(self._root)
            # print(dirname)
            # print(filename)
            abs_path = os.path.join(self._root, 'images', dirname, filename)
            label = self._check_load_keypoints(_coco, entry)
            if not label:
                continue

            # num of items are relative to person, not image
            for obj in label:
                items.append(abs_path)
                labels.append(obj)

        return items, labels

    def _check_load_keypoints(self, coco, entry):
        """Check and load ground-truth keypoints"""
        ann_ids = coco.getAnnIds(imgIds=entry['id'], iscrowd=False)
        objs = coco.loadAnns(ann_ids)
        # check valid bboxes
        valid_objs = []
        width = entry['width']
        height = entry['height']

        for obj in objs:
            contiguous_cid = self.json_id_to_contiguous[obj['category_id']]
            if contiguous_cid >= self.num_class:
                # not class of interest
                continue
            if max(obj['keypoints']) == 0:
                continue
            # convert from (x, y, w, h) to (xmin, ymin, xmax, ymax) and clip bound
            xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(obj['bbox']), width, height)
            # require non-zero box area
            if obj['area'] <= 0 or xmax <= xmin or ymax <= ymin:
                continue
            if obj['num_keypoints'] == 0:
                continue
            # joints 3d: (num_joints, 3, 2); 3 is for x, y, z; 2 is for position, visibility
            joints_3d = np.zeros((self.num_joints, 3, 2), dtype=np.float32)
            for i in range(self.num_joints):
                joints_3d[i, 0, 0] = obj['keypoints'][i * 3 + 0]
                joints_3d[i, 1, 0] = obj['keypoints'][i * 3 + 1]
                # joints_3d[i, 2, 0] = 0
                visible = min(1, obj['keypoints'][i * 3 + 2])
                joints_3d[i, :2, 1] = visible
                # joints_3d[i, 2, 1] = 0

            if np.sum(joints_3d[:, 0, 1]) < 1:
                # no visible keypoint
                continue

            if self._check_centers and self._train:
                bbox_center, bbox_area = self._get_box_center_area((xmin, ymin, xmax, ymax))
                kp_center, num_vis = self._get_keypoints_center_count(joints_3d)
                ks = np.exp(-2 * np.sum(np.square(bbox_center - kp_center)) / bbox_area)
                if (num_vis / 80.0 + 47 / 80.0) > ks:
                    continue

            valid_objs.append({
                'bbox': (xmin, ymin, xmax, ymax),
                'width': width,
                'height': height,
                'joints_3d': joints_3d
            })

        if not valid_objs:
            if not self._skip_empty:
                # dummy invalid labels if no valid objects are found
                valid_objs.append({
                    'bbox': np.array([-1, -1, 0, 0]),
                    'width': width,
                    'height': height,
                    'joints_3d': np.zeros((self.num_joints, 2, 2), dtype=np.float32)
                })
        return valid_objs

    def _get_box_center_area(self, bbox):
        """Get bbox center"""
        c = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
        area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        return c, area

    def _get_keypoints_center_count(self, keypoints):
        """Get geometric center of all keypoints"""
        keypoint_x = np.sum(keypoints[:, 0, 0] * (keypoints[:, 0, 1] > 0))
        keypoint_y = np.sum(keypoints[:, 1, 0] * (keypoints[:, 1, 1] > 0))
        num = float(np.sum(keypoints[:, 0, 1]))
        return np.array([keypoint_x / num, keypoint_y / num]), num


@DATASET_REGISTRY.register()
class Mscoco_det(data.Dataset):
    """ COCO human detection box dataset.

    """
    EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    def __init__(self,
                 det_file=None,
                 opt=None,
                 **cfg):

        self._cfg = cfg
        self._opt = opt
        self._preset_cfg = cfg['preset']
        self._root = cfg['root']
        self._img_prefix = cfg['img_prefix']
        if not det_file:
            det_file = cfg['det_file']
        self._ann_file = os.path.join(self._root, cfg['ann'])

        assert os.path.exists(det_file), "Error: no detection results found"
        with open(det_file, 'r') as fid:
            self._det_json = json.load(fid)

        self._input_size = self._preset_cfg['image_size']
        self._output_size = self._preset_cfg['heatmap_size']

        self._sigma = self._preset_cfg['sigma']

        if self._preset_cfg['type'] == 'simple':
            self.transformation = SimpleTransform(
                self, scale_factor=0,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=0, sigma=self._sigma,
                train=False)

    def __getitem__(self, index):
        det_res = self._det_json[index]
        if not isinstance(det_res['image_id'], int):
            img_id, _ = os.path.splitext(os.path.basename(det_res['image_id']))
            img_id = int(img_id)
        else:
            img_id = det_res['image_id']
        img_path = '/root/Improve-HPE-with-Diffusion/data/coco/images/val2017/%012d.jpg' % img_id

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)  # scipy.misc.imread(img_path, mode='RGB') is depreciated

        imght, imgwidth = image.shape[1], image.shape[2]
        x1, y1, w, h = det_res['bbox']
        bbox = [x1, y1, x1 + w, y1 + h]
        inp, bbox = self.transformation.test_transform(image, bbox)
        return inp, torch.Tensor(bbox), torch.Tensor([det_res['bbox']]), torch.Tensor([det_res['image_id']]), torch.Tensor([det_res['score']]), torch.Tensor([imght]), torch.Tensor([imgwidth])

    def __len__(self):
        return len(self._det_json)

    @property
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return [[1, 2], [3, 4], [5, 6], [7, 8],
                [9, 10], [11, 12], [13, 14], [15, 16]]
    

@DATASET_REGISTRY.register()
class H36m(data.Dataset):
    """ Human3.6M dataset.

    Parameters
    ----------
    ann_file: str,
        Path to the annotation json file.
    root: str, default './data/h36m'
        Path to the h36m dataset.
    train: bool, default is True
        If true, will set as training mode.
    skip_empty: bool, default is False
        Whether skip entire image if no valid label is found.
    """
    CLASSES = ['person']
    EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    num_joints = 18
    bbox_3d_shape = (2000, 2000, 2000)
    joints_name = ('Pelvis',  # 0
                   'R_Hip', 'R_Knee', 'R_Ankle',  # 3
                   'L_Hip', 'L_Knee', 'L_Ankle',  # 6
                   'Torso', 'Neck',  # 8
                   'Nose', 'Head',  # 10
                   'L_Shoulder', 'L_Elbow', 'L_Wrist',  # 13
                   'R_Shoulder', 'R_Elbow', 'R_Wrist',  # 16
                   'Thorax')  # 17
    action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases',
                   'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']
    skeleton = ((1, 0), (2, 1), (3, 2),  # 2
                (4, 0), (5, 4), (6, 5),  # 5
                (7, 0), (8, 7),  # 7
                (9, 8), (10, 9),  # 9
                (11, 7), (12, 11), (13, 12),  # 12
                (14, 7), (15, 14), (16, 15),  # 15
                (17, 7))  # 16
    block_list = ['s_09_act_05_subact_02_ca', 's_09_act_10_subact_02_ca', 's_09_act_13_subact_01_ca']

    def __init__(self,
                 cfg,
                 ann_file,
                 root='./data/h36m',
                 train=True,
                 skip_empty=True,
                 dpg=False,
                 lazy_import=False,
                 **kwargs):
        self._cfg = cfg
        self._preset_cfg = cfg['PRESET']
        self.protocol = self._preset_cfg.PROTOCOL

        self._ann_file = os.path.join(
            root, 'annotations', ann_file + f'_protocol_{self.protocol}.json')
        self._lazy_import = lazy_import
        self._root = root
        self._skip_empty = skip_empty
        self._train = train
        self._dpg = dpg

        self._det_bbox_file = getattr(cfg, 'DET_BOX', None)

        self._scale_factor = self._preset_cfg.SCALE_FACTOR
        self._color_factor = self._preset_cfg.COLOR_FACTOR
        self._rot = self._preset_cfg.ROT_FACTOR
        self._input_size = self._preset_cfg.IMAGE_SIZE
        self._output_size = self._preset_cfg.HEATMAP_SIZE

        self._occlusion = self._preset_cfg.OCCLUSION

        self._sigma = self._preset_cfg.SIGMA

        self._check_centers = False

        self.num_class = len(self.CLASSES)
        self.num_joints = self._preset_cfg.NUM_JOINTS

        self.num_joints_half_body = self._preset_cfg.NUM_JOINTS_HALF_BODY
        self.prob_half_body = self._preset_cfg.PROB_HALF_BODY

        self._loss_type = cfg['heatmap2coord']

        self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)
        self.lower_body_ids = (0, 1, 2, 3, 4, 5, 6)

        self.root_idx = self.joints_name.index('Pelvis')
        self.lshoulder_idx = self.joints_name.index('L_Shoulder')
        self.rshoulder_idx = self.joints_name.index('R_Shoulder')

        self._items, self._labels = self._lazy_load_json()

        if self._preset_cfg['TYPE'] == 'simple_3d':
            self.transformation = SimpleTransform3D(
                self, scale_factor=self._scale_factor,
                color_factor=self._color_factor,
                occlusion=self._occlusion,
                input_size=self._input_size,
                output_size=self._output_size,
                bbox_3d_shape=self.bbox_3d_shape,
                rot=self._rot, sigma=self._sigma,
                train=self._train, add_dpg=self._dpg,
                loss_type=self._loss_type, scale_mult=1)

    def __getitem__(self, idx):
        # get image id
        img_path = self._items[idx]
        img_id = int(self._labels[idx]['img_id'])

        # load ground truth, including bbox, keypoints, image size
        label = copy.deepcopy(self._labels[idx])

        img = scipy.misc.imread(img_path, mode='RGB')
        # img = load_image(img_path)
        # img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        # transform ground truth into training label and apply data augmentation
        target = self.transformation(img, label)
        img = target.pop('image')
        bbox = target.pop('bbox')
        return img, target, img_id, bbox

    def __len__(self):
        return len(self._items)

    def _lazy_load_json(self):
        if os.path.exists(self._ann_file + '_annot_keypoint.pkl') and self._lazy_import:
            print('Lazy load annot...')
            with open(self._ann_file + '_annot_keypoint.pkl', 'rb') as fid:
                items, labels = pk.load(fid)
        else:
            items, labels = self._load_jsons()
            try:
                with open(self._ann_file + '_annot_keypoint.pkl', 'wb') as fid:
                    pk.dump((items, labels), fid, pk.HIGHEST_PROTOCOL)
            except Exception as e:
                print(e)
                print('Skip writing to .pkl file.')

        return items, labels

    def _load_jsons(self):
        """Load all image paths and labels from JSON annotation files into buffer."""
        items = []
        labels = []

        with open(self._ann_file, 'r') as fid:
            database = json.load(fid)
        # iterate through the annotations
        bbox_scale_list = []
        det_bbox_set = {}
        if self._det_bbox_file is not None:
            bbox_list = json.load(open(os.path.join(
                self._root, 'annotations', self._det_bbox_file + f'_protocol_{self.protocol}.json'), 'r'))
            for item in bbox_list:
                image_id = item['image_id']
                det_bbox_set[image_id] = item['bbox']

        for ann_image, ann_annotations in zip(database['images'], database['annotations']):
            ann = dict()
            for k, v in ann_image.items():
                assert k not in ann.keys()
                ann[k] = v
            for k, v in ann_annotations.items():
                ann[k] = v
            skip = False
            for name in self.block_list:
                if name in ann['file_name']:
                    skip = True
            if skip:
                continue

            image_id = ann['image_id']

            width, height = ann['width'], ann['height']
            if self._det_bbox_file is not None:
                xmin, ymin, xmax, ymax = bbox_clip_xyxy(
                    bbox_xywh_to_xyxy(det_bbox_set[ann['file_name']]), width, height)
            else:
                xmin, ymin, xmax, ymax = bbox_clip_xyxy(
                    bbox_xywh_to_xyxy(ann['bbox']), width, height)

            R, t = np.array(ann['cam_param']['R'], dtype=np.float32), np.array(
                ann['cam_param']['t'], dtype=np.float32)
            f, c = np.array(ann['cam_param']['f'], dtype=np.float32), np.array(
                ann['cam_param']['c'], dtype=np.float32)

            joint_world = np.array(ann['keypoints_world'])
            joint_world = self.add_thorax(joint_world)
            joint_cam = np.zeros((self.num_joints, 3))
            for j in range(self.num_joints):
                joint_cam[j] = world2cam(joint_world[j], R, t)

            joint_img = cam2pixel(joint_cam, f, c)
            joint_img[:, 2] = joint_img[:, 2] - joint_cam[self.root_idx, 2]
            joint_vis = np.ones((self.num_joints, 3))

            root_cam = joint_cam[self.root_idx]

            abs_path = os.path.join(self._root, 'images', ann['file_name'])

            tot_bone_len = 0
            for parent, child in self.skeleton:
                bl = np.sqrt(np.sum((joint_cam[parent] - joint_cam[child]) ** 2))
                tot_bone_len += bl

            items.append(abs_path)
            labels.append({
                'bbox': (xmin, ymin, xmax, ymax),
                'img_id': image_id,
                'img_path': abs_path,
                'width': width,
                'height': height,
                'joint_img': joint_img,
                'joint_vis': joint_vis,
                'joint_cam': joint_cam,
                'root_cam': root_cam,
                'tot_bone_len': tot_bone_len,
                'f': f,
                'c': c
            })
            bbox_scale_list.append(max(xmax - xmin, ymax - ymin))

        return items, labels

    @property
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))

    @property
    def bone_pairs(self):
        """Bone pairs which defines the pairs of bone to be swapped
        when the image is flipped horizontally."""
        return ((0, 3), (1, 4), (2, 5), (10, 13), (11, 14), (12, 15))

    def _get_box_center_area(self, bbox):
        """Get bbox center"""
        c = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
        area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        return c, area

    def _get_keypoints_center_count(self, keypoints):
        """Get geometric center of all keypoints"""
        keypoint_x = np.sum(keypoints[:, 0, 0] * (keypoints[:, 0, 1] > 0))
        keypoint_y = np.sum(keypoints[:, 1, 0] * (keypoints[:, 1, 1] > 0))
        num = float(np.sum(keypoints[:, 0, 1]))
        return np.array([keypoint_x / num, keypoint_y / num]), num

    def add_thorax(self, joint_coord):
        thorax = (joint_coord[self.lshoulder_idx, :] +
                  joint_coord[self.rshoulder_idx, :]) * 0.5
        thorax = thorax.reshape((1, 3))
        joint_coord = np.concatenate((joint_coord, thorax), axis=0)
        return joint_coord

    def evaluate(self, preds, result_dir):
        print('Evaluation start...')
        gts = self._labels
        assert len(gts) == len(preds)
        sample_num = len(gts)

        pred_save = []
        error = np.zeros((sample_num, self.num_joints - 1))  # joint error
        error_pa = np.zeros((sample_num, self.num_joints - 1))  # joint error
        error_x = np.zeros((sample_num, self.num_joints - 1))  # joint error
        error_y = np.zeros((sample_num, self.num_joints - 1))  # joint error
        error_z = np.zeros((sample_num, self.num_joints - 1))  # joint error
        # error for each sequence
        error_action = [[] for _ in range(len(self.action_name))]
        for n in range(sample_num):
            gt = gts[n]
            image_id = gt['img_id']
            f = gt['f']
            c = gt['c']
            bbox = gt['bbox']
            gt_3d_root = gt['root_cam']
            gt_3d_kpt = gt['joint_cam']

            gt_3d_kpt = np.take(gt_3d_kpt, self.EVAL_JOINTS, axis=0)
            # gt_vis = gt['joint_vis']

            # restore coordinates to original space
            pred_2d_kpt = preds[image_id]['uvd'].copy()
            # pred_2d_kpt[:, 0] = pred_2d_kpt[:, 0] / self._output_size[1] * bbox[2] + bbox[0]
            # pred_2d_kpt[:, 1] = pred_2d_kpt[:, 1] / self._output_size[0] * bbox[3] + bbox[1]
            pred_2d_kpt[:, 2] = pred_2d_kpt[:, 2] * self.bbox_3d_shape[0] + gt_3d_root[2]

            vis = False
            if vis:
                import random
                cvimg = cv2.imread(
                    gt['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                filename = str(random.randrange(1, 500))
                tmpimg = cvimg.copy().astype(np.uint8)
                tmpkps = np.zeros((3, self.joint_num))
                tmpkps[0, :], tmpkps[1, :] = pred_2d_kpt[:, 0], pred_2d_kpt[:, 1]
                tmpkps[2, :] = 1
                tmpimg = vis_keypoints(tmpimg, tmpkps, self.skeleton)
                cv2.imwrite(filename + '_output.jpg', tmpimg)

            # back project to camera coordinate system
            pred_3d_kpt = pixel2cam(pred_2d_kpt, f, c)

            # root joint alignment
            pred_3d_kpt = pred_3d_kpt - pred_3d_kpt[self.root_idx]
            gt_3d_kpt = gt_3d_kpt - gt_3d_kpt[self.root_idx]

            # exclude thorax
            pred_3d_kpt = np.take(pred_3d_kpt, self.EVAL_JOINTS, axis=0)
            gt_3d_kpt = np.take(gt_3d_kpt, self.EVAL_JOINTS, axis=0)

            # if self.protocol == 1:
            # rigid alignment for PA MPJPE (protocol #1)
            pred_3d_kpt_pa = reconstruction_error(pred_3d_kpt.copy(), gt_3d_kpt.copy())

            # error calculate
            error[n] = np.sqrt(np.sum((pred_3d_kpt - gt_3d_kpt)**2, 1))
            error_pa[n] = np.sqrt(np.sum((pred_3d_kpt_pa - gt_3d_kpt)**2, 1))
            error_x[n] = np.abs(pred_3d_kpt[:, 0] - gt_3d_kpt[:, 0])
            error_y[n] = np.abs(pred_3d_kpt[:, 1] - gt_3d_kpt[:, 1])
            error_z[n] = np.abs(pred_3d_kpt[:, 2] - gt_3d_kpt[:, 2])
            img_name = gt['img_path']
            action_idx = int(img_name[img_name.find(
                'act') + 4:img_name.find('act') + 6]) - 2
            error_action[action_idx].append(error[n].copy())

            # prediction save
            pred_save.append({'image_id': image_id, 'joint_cam': pred_3d_kpt.tolist(
            ), 'bbox': bbox, 'root_cam': gt_3d_root.tolist()})  # joint_cam is root-relative coordinate

        # total error
        tot_err = np.mean(error)
        tot_err_pa = np.mean(error_pa)
        tot_err_x = np.mean(error_x)
        tot_err_y = np.mean(error_y)
        tot_err_z = np.mean(error_z)
        metric = 'PA MPJPE' if self.protocol == 1 else 'MPJPE'

        eval_summary = f'PA-MPJPE {tot_err_pa:2f}, Protocol {self.protocol} error ({metric}) >> tot: {tot_err:2f}, x: {tot_err_x:2f}, y: {tot_err_y:.2f}, z: {tot_err_z:2f}\n'

        # error for each action
        for i in range(len(error_action)):
            err = np.mean(np.array(error_action[i]))
            eval_summary += (self.action_name[i] + ': %.2f ' % err)

        print(eval_summary)

        # prediction save
        with open(result_dir, 'w') as f:
            json.dump(pred_save, f)
        print("Test result is saved at " + result_dir)
        return tot_err
    

@DATASET_REGISTRY.register()
class Mpii(data.Dataset):
    """ MPII Human Pose Dataset.
    Parameters
    ----------
    root: str, default './data/mpii'
        Path to the mpii dataset.
    train: bool, default is True
        If true, will set as training mode.
    dpg: bool, default is False
        If true, will activate `dpg` for data augmentation.
    """

    CLASSES = ['person']
    num_joints = 16
    bbox_3d_shape = (2000, 2000, 2000)
    joints_name = ('right_ankle', 'right_knee', 'right_hip',  # 2
                   'left_hip', 'left_knee', 'left_ankle',   # 5
                   'pelv', 'thrx', 'neck', 'head',  # 9
                   'right_wrist', 'right_elbow', 'right_shoulder',  # 12
                   'left_shoulder', 'left_elbow', 'left_wrist')  # 15
    skeleton = ((3, 6), (4, 3), (5, 4),
                (2, 6), (1, 2), (0, 1),
                (7, 6), (8, 7), (9, 8),
                (13, 7), (14, 13), (15, 14),
                (12, 7), (11, 12), (10, 11))
    mean_bone_len = None

    def __init__(self,
                 cfg,
                 ann_file,
                 root='./data/mpii',
                 train=True,
                 skip_empty=True,
                 dpg=False,
                 lazy_import=False):

        self._cfg = cfg
        self._preset_cfg = cfg['PRESET']
        self._ann_file = os.path.join(root, 'annotations', ann_file)
        self._lazy_import = lazy_import
        self._root = root
        self._skip_empty = skip_empty
        self._train = train
        self._dpg = dpg

        self._scale_factor = self._preset_cfg.SCALE_FACTOR
        self._color_factor = self._preset_cfg.COLOR_FACTOR
        self._rot = self._preset_cfg.ROT_FACTOR
        self._input_size = self._preset_cfg.IMAGE_SIZE
        self._output_size = self._preset_cfg.HEATMAP_SIZE

        self._occlusion = self._preset_cfg.OCCLUSION
        # self._occlusion = False

        self._sigma = self._preset_cfg.SIGMA
        self._img_prefix = 'images'

        self._check_centers = False

        self.num_class = len(self.CLASSES)

        self.num_joints_half_body = self._preset_cfg.NUM_JOINTS_HALF_BODY
        self.prob_half_body = self._preset_cfg.PROB_HALF_BODY

        self._loss_type = cfg['heatmap2coord']

        self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15)
        self.lower_body_ids = (0, 1, 2, 3, 4, 5, 6)

        self.transformation = SimpleTransform3D(
            self, scale_factor=self._scale_factor,
            color_factor=self._color_factor,
            occlusion=self._occlusion,
            input_size=self._input_size,
            output_size=self._output_size,
            bbox_3d_shape=self.bbox_3d_shape,
            rot=self._rot, sigma=self._sigma,
            train=self._train, add_dpg=self._dpg,
            loss_type=self._loss_type)

        self._items, self._labels = self._lazy_load_json()

    def __getitem__(self, idx):
        # get image id
        img_path = self._items[idx]
        img_id = int(os.path.splitext(os.path.basename(img_path))[0])

        # load ground truth, including bbox, keypoints, image size
        label = copy.deepcopy(self._labels[idx])
        img = scipy.misc.imread(img_path, mode='RGB')
        # transform ground truth into training label and apply data augmentation
        target = self.transformation(img, label)

        img = target.pop('image')
        bbox = target.pop('bbox')
        return img, target, img_id, bbox

    def __len__(self):
        return len(self._items)

    def _lazy_load_ann_file(self):
        if os.path.exists(self._ann_file + '.pkl') and self._lazy_import:
            print('Lazy load json...')
            with open(self._ann_file + '.pkl', 'rb') as fid:
                return pk.load(fid)
        else:
            _coco = COCO(self._ann_file)
            try:
                with open(self._ann_file + '.pkl', 'wb') as fid:
                    pk.dump(_coco, fid, pk.HIGHEST_PROTOCOL)
            except Exception as e:
                print(e)
                print('Skip writing to .pkl file.')
            return _coco

    def _lazy_load_json(self):
        if os.path.exists(self._ann_file + '_annot_keypoint.pkl') and self._lazy_import:
            print('Lazy load annot...')
            with open(self._ann_file + '_annot_keypoint.pkl', 'rb') as fid:
                items, labels = pk.load(fid)
        else:
            items, labels = self._load_jsons()
            try:
                with open(self._ann_file + '_annot_keypoint.pkl', 'wb') as fid:
                    pk.dump((items, labels), fid, pk.HIGHEST_PROTOCOL)
            except Exception as e:
                print(e)
                print('Skip writing to .pkl file.')

        return items, labels

    def _load_jsons(self):
        """Load all image paths and labels from annotation files into buffer."""
        items = []
        labels = []

        _mpii = self._lazy_load_ann_file()
        classes = [c['name'] for c in _mpii.loadCats(_mpii.getCatIds())]
        assert classes == self.CLASSES, "Incompatible category names with MPII. "

        # iterate through the annotations
        image_ids = sorted(_mpii.getImgIds())
        for entry in _mpii.loadImgs(image_ids):
            filename = entry['file_name']
            abs_path = os.path.join(self._root, self._img_prefix, filename)
            if False and not os.path.exists(abs_path):
                raise IOError('Image: {} not exists.'.format(abs_path))
            label = self._check_load_keypoints(_mpii, entry)
            if not label:
                continue

            # num of items are relative to person, not image
            for obj in label:
                items.append(abs_path)
                labels.append(obj)

        return items, labels

    def _check_load_keypoints(self, _mpii, entry):
        """Check and load ground-truth keypoints"""
        ann_ids = _mpii.getAnnIds(imgIds=entry['id'], iscrowd=False)
        objs = _mpii.loadAnns(ann_ids)
        # check valid bboxes
        valid_objs = []
        width = entry['width']
        height = entry['height']

        for obj in objs:
            if max(obj['keypoints']) == 0:
                continue
            # convert from (x, y, w, h) to (xmin, ymin, xmax, ymax) and clip bound
            xmin, ymin, xmax, ymax = bbox_clip_xyxy(
                bbox_xywh_to_xyxy(obj['bbox']), width, height)
            # require non-zero box area
            if xmax <= xmin or ymax <= ymin:
                continue
            if obj['num_keypoints'] == 0:
                continue
            # joints 3d: (num_joints, 3, 2); 3 is for x, y, z; 2 is for position, visibility
            joints_3d = np.zeros((self.num_joints, 3, 2), dtype=np.float32)
            for i in range(self.num_joints):
                joints_3d[i, 0, 0] = obj['keypoints'][i * 3 + 0]
                joints_3d[i, 1, 0] = obj['keypoints'][i * 3 + 1]
                # joints_3d[i, 2, 0] = 0
                visible = min(1, obj['keypoints'][i * 3 + 2])
                joints_3d[i, :2, 1] = visible
                # joints_3d[i, 2, 1] = 0

            if np.sum(joints_3d[:, 0, 1]) < 1:
                # no visible keypoint
                continue

            if self._check_centers and self._train:
                bbox_center, bbox_area = self._get_box_center_area(
                    (xmin, ymin, xmax, ymax))
                kp_center, num_vis = self._get_keypoints_center_count(
                    joints_3d)
                ks = np.exp(-2 * np.sum(np.square(bbox_center -
                                                  kp_center)) / bbox_area)
                if (num_vis / 80.0 + 47 / 80.0) > ks:
                    continue

            joint_img = np.zeros((self.num_joints, 3))
            joint_vis = np.ones((self.num_joints, 3))
            joint_img[:, 0] = joints_3d[:, 0, 0]
            joint_img[:, 1] = joints_3d[:, 1, 0]
            joint_vis[:, 2] = 0

            valid_objs.append({
                'bbox': (xmin, ymin, xmax, ymax),
                'width': width,
                'height': height,
                'joint_img': joint_img,
                'joint_vis': joint_vis,
            })

        if not valid_objs:
            if not self._skip_empty:
                # dummy invalid labels if no valid objects are found
                valid_objs.append({
                    'bbox': np.array([-1, -1, 0, 0]),
                    'width': width,
                    'height': height,
                    'joints_3d': np.zeros((self.num_joints, 2, 2), dtype=np.float32)
                })
        return valid_objs

    @property
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return [[0, 5], [1, 4], [2, 3],
                [10, 15], [11, 14], [12, 13]]

    def _get_box_center_area(self, bbox):
        """Get bbox center"""
        c = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
        area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        return c, area

    def _get_keypoints_center_count(self, keypoints):
        """Get geometric center of all keypoints"""
        keypoint_x = np.sum(keypoints[:, 0, 0] * (keypoints[:, 0, 1] > 0))
        keypoint_y = np.sum(keypoints[:, 1, 0] * (keypoints[:, 1, 1] > 0))
        num = float(np.sum(keypoints[:, 0, 1]))
        return np.array([keypoint_x / num, keypoint_y / num]), num


@DATASET_REGISTRY.register()
class H36mMpii(data.Dataset):
    CLASSES = ['person']
    EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    num_joints = 18
    num_bones = 17
    bbox_3d_shape = (2000, 2000, 2000)
    joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck',
                   'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'Thorax')
    action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases',
                   'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']
    skeleton = ((1, 0), (2, 0),
                (3, 1), (4, 2), (5, 3), (6, 4),
                (7, 0),
                (8, 0), (9, 0),
                (10, 8), (11, 9), (12, 10), (13, 11)
                )
    data_domain = set([
        'type',
        'target_uvd',
        'target_uvd_weight'
    ])

    def __init__(self,
                 train=True,
                 skip_empty=True,
                 dpg=False,
                 lazy_import=False,
                 **cfg):
        self._train = train
        self._preset_cfg = cfg['PRESET']

        cfg = EasyDict(cfg)
        if train:
            self.db0 = H36m(
                cfg=cfg,
                ann_file=cfg.SET_LIST[0].TRAIN_SET,
                train=train)
            self.db1 = Mpii(
                cfg=cfg,
                ann_file=f'{cfg.SET_LIST[1].TRAIN_SET}.json',
                train=True)

            self._subsets = [self.db0, self.db1]
        else:
            self.db0 = H36m(
                cfg=cfg,
                ann_file=cfg.TEST_SET,
                train=train)

            self._subsets = [self.db0]

        self._subset_size = [len(item) for item in self._subsets]
        self.cumulative_sizes = self.cumsum(self._subset_size)
        self._db0_size = len(self.db0)

        if train:
            self._db1_size = len(self.db1)
            self.tot_size = 2 * self._db0_size
        else:
            self.tot_size = self._db0_size
        self.joint_pairs = self.db0.joint_pairs
        self.evaluate = self.db0.evaluate

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            r.append(e + s)
            s += e
        return r

    def __len__(self):
        return self.tot_size

    def __getitem__(self, idx):
        assert idx >= 0
        if idx < self._db0_size:
            dataset_idx = 0
            sample_idx = idx
        else:
            assert self._train
            dataset_idx = 1

            _rand = random.uniform(0, 1)
            sample_idx = int(_rand * self._db1_size)

        sample = self._subsets[dataset_idx][sample_idx]
        img, target, img_id, bbox = sample

        if dataset_idx == 1:
            # Mpii
            target_uvd_origin = target.pop('target_uvd')
            target_uvd_weight_origin = target.pop('target_uvd_weight')

            target_uvd = torch.zeros(self.num_joints, 3)
            target_uvd_weight = torch.zeros(self.num_joints, 3)

            assert target_uvd_origin.dim() == 1 and target_uvd_origin.shape[0] == 16 * 3, target_uvd_origin.shape
            target_uvd_origin = target_uvd_origin.reshape(16, 3)
            target_uvd_weight_origin = target_uvd_weight_origin.reshape(16, 3)
            for i in range(s_36_jt_num):
                id1 = i
                id2 = s_mpii_2_hm36_jt[i]
                if id2 >= 0:
                    target_uvd[id1, :2] = target_uvd_origin[id2, :2].clone()
                    target_uvd_weight[id1, :2] = target_uvd_weight_origin[id2, :2].clone()

            target['target_uvd'] = target_uvd.reshape(-1)
            target['target_uvd_weight'] = target_uvd_weight.reshape(-1)

        assert set(target.keys()).issubset(self.data_domain), (set(target.keys()), self.data_domain)
        target.pop('type')

        return img, target, img_id, bbox