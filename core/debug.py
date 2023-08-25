import os
import numpy as np
import torch
import cv2
from imageio.v2 import imread, imwrite
import sys
import argparse
from torch.utils.data import DataLoader

sys.path.append('/root/Improve-HPE-with-Diffusion')
import model as Model
import logger as Logger
from datasets import build_datasets_debug

IMAGE_ROOT = '/root/autodl-tmp/coco/images/train2017'
SAVE_ROOT = '/root/Improve-HPE-with-Diffusion/vis'
NUM_KPTS = 17
#CKPT_PATH = '/root/Improve-HPE-with-Diffusion/experiments/fixed_res_and_diff_230820_092507/checkpoint/last_gen.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_HEIGHT = 256
IMG_WIDTH = 192

def train_pred(model, img, gt):
    preds, im_feats = model.netG.regress(images=img)
    pred_jts = preds['raw_pred_jts']
    x_start = gt - pred_jts

    b = img.shape[0]
    t = np.random.randint(model.netG.num_timesteps - 1, model.netG.num_timesteps + 1)  # sample a 't' value -- total diffusion step T
    continuous_sqrt_alpha_cumprod = torch.FloatTensor( # sample one alpha value for each sample in the batch
        np.random.uniform(
            model.netG.sqrt_alphas_cumprod_prev[t-1],
            model.netG.sqrt_alphas_cumprod_prev[t],
            size=b
        )
    ).to(DEVICE)
    continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)

    noise = torch.randn_like(x_start) # sample a noise from N(0,1)
    x_noisy = model.netG.q_sample(                                  # sample X_t
        x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod, noise=noise)

    if not model.netG.condition_on_preds:
        x_recon = model.netG.denoise_fn(x_noisy, im_feats, continuous_sqrt_alpha_cumprod) # x_recon -- reconstructed noise
    else:
        x_recon = model.netG.denoise_fn(
            x_noisy, im_feats, continuous_sqrt_alpha_cumprod, pred_jts)
    
    preds = pred_jts + x_recon
    return {'raw_preds': pred_jts, 'preds': preds}

def val_pred(model, img):
    ret = model.netG.sample(img)
    raw_preds = ret['preds']['raw_pred_jts']
    if 'res' not in ret:
        return {'raw_preds': raw_preds}
    preds = raw_preds + ret['res']
    return {'raw_preds': raw_preds, 'preds': preds}

def show_skeleton(kpts, img, color=(0,0,255)):
    img = img.permute(1,2,0).cpu().numpy()
    img = np.ascontiguousarray((img+0.5) * 255, dtype=np.uint8)
    kpts = kpts.reshape(-1,2)
    kpts[..., 0] = (kpts[..., 0] + 0.5) * IMG_WIDTH
    kpts[..., 1] = (kpts[..., 1] + 0.5) * IMG_HEIGHT
    for i in range(NUM_KPTS):
        pos = (int(kpts[i][0]),int(kpts[i][1]))
        if pos[0] > 0 and pos[1] > 0:
            cv2.circle(img, pos, 1, (0,0,255), -1) #为肢体点画红色实心圆
    return img

def horizon_concate(inp0, inp1):
    h0, w0 = inp0.shape[:2]
    h1, w1 = inp1.shape[:2]
    if inp0.ndim == 3:
        inp = np.zeros((max(h0, h1), w0 + w1, 3), dtype=inp0.dtype)
        inp[:h0, :w0, :] = inp0
        inp[:h1, w0:(w0 + w1), :] = inp1
    else:
        inp = np.zeros((max(h0, h1), w0 + w1), dtype=inp0.dtype)
        inp[:h0, :w0] = inp0
        inp[:h1, w0:(w0 + w1)] = inp1
    return inp

def vis(train_preds, val_preds, imgs, color=(255,128,128)):
    for i, img in enumerate(imgs):
        train_img_raw = show_skeleton(train_preds['raw_preds'][i], img, color)
        val_img_raw = show_skeleton(val_preds['raw_preds'][i], img, color)

        if not os.path.exists(SAVE_ROOT):
            os.makedirs(SAVE_ROOT)
        save_path_raw = os.path.join(SAVE_ROOT, 'res_{}_raw.jpg'.format(i))
        imwrite(save_path_raw, horizon_concate(train_img_raw, val_img_raw))

        if 'preds' in val_preds:
            train_img = show_skeleton(train_preds['preds'][i], img, color)
            val_img = show_skeleton(val_preds['preds'][i], img, color)
            save_path = os.path.join(SAVE_ROOT, 'res_{}.jpg'.format(i))
            imwrite(save_path, horizon_concate(train_img, val_img))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/root/Improve-HPE-with-Diffusion/config/fixed_res_and_diff_debug.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                         help='Run either train(training) or val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    train_dataset, _, _ = build_datasets_debug(opt['datasets'], opt['data_preset'])
    train_loader = DataLoader(train_dataset, batch_size=opt['train']['batch_size'], shuffle=True)
    model = Model.create_model(opt)

    # all_imgs = os.listdir(IMAGE_ROOT)
    # img_path = all_imgs[np.random.choice(len(all_imgs))]
    # img = torch.from_numpy(np.array(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)))
    
    for _, (inps, labels, img_ids, bboxes) in enumerate(train_loader):
        inps = inps.to(DEVICE)
        gt = labels['target_uv'].to(DEVICE)
        print(labels.keys())
        #labels = {k:v.to(DEVICE) for k,v in labels.items()}
        with torch.no_grad():
            model.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
            train_preds = train_pred(model, inps, gt)
            model.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
            val_preds = val_pred(model, inps)
            print(img_ids)
        break

    
    vis(train_preds, val_preds, inps)