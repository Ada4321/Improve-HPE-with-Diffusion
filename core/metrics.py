import os
import math
import numpy as np
import cv2
import torch
from torchvision.utils import make_grid
import sys
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(img_path, img)


def evaluate_mAP(res_file, ann_type='bbox', ann_file='person_keypoints_val2017.json', silence=True, img_ids=None):
    """Evaluate mAP result for coco dataset.

    Parameters
    ----------
    res_file: str
        Path to result json file.
    ann_type: str
        annotation type, including: `bbox`, `segm`, `keypoints`.
    ann_file: str
        Path to groundtruth file.
    silence: bool
        True: disable running log.

    """
    class NullWriter(object):
        def write(self, arg):
            pass

    ann_file = os.path.join('/root/Improve-HPE-with-Diffusion/data/coco/annotations/', ann_file)

    if silence:
        nullwrite = NullWriter()
        oldstdout = sys.stdout
        sys.stdout = nullwrite  # disable output

    cocoGt = COCO(ann_file)
    cocoDt = cocoGt.loadRes(res_file)

    cocoEval = COCOeval(cocoGt, cocoDt, ann_type, img_ids=img_ids)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    if silence:
        sys.stdout = oldstdout  # enable output

    stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)',
                   'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']
    info_str = {}
    for ind, name in enumerate(stats_names):
        info_str[name] = cocoEval.stats[ind]

    return info_str


# return mpjpe by action
def mpjpe_by_action(predicted, target, action, error_sum):
    error_sum = mpjpe_by_action_p1(predicted, target, action, error_sum)
    error_sum = mpjpe_by_action_p2(predicted, target, action, error_sum)

    return error_sum

def mpjpe_by_action_p1(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape  # (b,num_kps,num_axis)
    predicted = predicted.reshape(-1, predicted.shape[-2], predicted.shape[-1])
    target = target.reshape(-1, target.shape[-2], target.shape[-1])
    dist = torch.mean(torch.norm(predicted - target, dim=-1), dim=-1)  # (b,)

    for i in range(predicted.shape[0]):
    #     action_name = action[i]
        action_error_sum[action]['p1'].update(dist[i].item(), 1)
    # action_error_sum[action]['p1'].update(dist[i].item(), 1)
            
    return action_error_sum

# compute j-mpjpe
def mpjpe_by_action_p2(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    predicted = predicted.reshape(-1, predicted.shape[-2], predicted.shape[-1])
    target = target.reshape(-1, target.shape[-2], target.shape[-1])
    predicted = predicted.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    dist = p_mpjpe(predicted, target)  #(b,)

    for i in range(predicted.shape[0]):
        # action_name = action[i]
        action_error_sum[action]['p2'].update(dist[i], 1)
            
    return action_error_sum


def mpjpe(predicted, target):
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))

def p_mpjpe(predicted, target):
    assert predicted.shape == target.shape  # (b,num_kps,num_axis)

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY
    t = muX - a * np.matmul(muY, R)

    predicted_aligned = a * np.matmul(predicted, R) + t

    return np.mean(np.linalg.norm(predicted_aligned - target, axis=-1), axis=-1)  # (b,)


class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

def define_error_list(actions):
    error_sum = {}
    error_sum.update({actions[i]: 
        {'p1':AccumLoss(), 'p2':AccumLoss()} 
        for i in range(len(actions))})
    return error_sum