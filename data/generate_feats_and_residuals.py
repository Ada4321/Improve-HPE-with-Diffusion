import torch
from argparse import ArgumentParser
import os
import numpy as np

import sys
sys.path.append("/root/Improve-HPE-with-Diffusion")
from datasets import build_datasets
from model.regression_modules import MODEL_REGISTRY
from data.utils import eval_data_prepare

ROOT = "/root/autodl-tmp"
SAVE_PATH = os.path.join(ROOT, "human3.6m/pretrained/feats_res_2d_{}_{}.npz")


def gen_train(generator, model, data_type):
    assert generator.shuffle == False
    
    output = {"residual": {}, "feature": {}, "keypoints_2d": {}}
    for i, (_, _, batch_3d, batch_2d) in enumerate(generator.next_epoch()):
        print("train", i)
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
        output["residual"][i] = residual.detach().cpu().numpy()
        output["feature"][i] = feats.detach().cpu().numpy()
        output["keypoints_2d"][i] = inputs_2d.detach().cpu().numpy()

    print("Saving {}, train...".format(data_type))
    np.savez_compressed(SAVE_PATH.format(data_type, "train"), pretrained_items=output)
    print("Done saving {}, train.".format(data_type))

def gen_val(generator, model, data_type, kps_left, kps_right, joints_left, joints_right, receptive_field):
    output = {"residual": {}, "feature": {}, "keypoints_2d": {}, "action": {}}
    for i, (_, cano_action, batch, batch_2d) in enumerate(generator.next_epoch()):
        print("val", i)
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

    print("Saving {}, val...".format(data_type))
    np.savez_compressed(SAVE_PATH.format(data_type, "val"), pretrained_items=output)
    print("Done saving {}, val.".format(data_type))
        

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--dataset", default="H36M")
    parser.add_argument("--ckpt", default="checkpoint_cpn_243f.bin")
    parser.add_argument("--model", default="MixSTE2")
    parser.add_argument("--keypoint_type_2d", default="cpn_ft_h36m_dbb")
    parser.add_argument("--receptive_field", default=243, type=int)

    args = parser.parse_args()

    # dataset
    if args.dataset == "H36M":
        train_dataset, val_dataset = build_datasets(
            dataset_cfg={
                "type": "H36M",
                "root": "/root/autodl-tmp/human3.6m",
                "keypoints_type_2d": args.keypoint_type_2d,
                "remove_static_joints": True,
                "batch_size": 1024,
                "train_downsample": 1,
                "val_downsample": 1,
                "augmentation": True,
                "test_augmentation": False,
                "pad": 0,
                "action_filter": True,
                "num_frames": args.receptive_field,
                "num_joints": 17}
        )
    elif args.dataset == "3DHP":
        train_dataset, val_dataset = build_datasets(

        )
    print("Datasets initialized.")
    train_generator = train_dataset.get_generator()
    train_generator.shuffle = False
    val_generator = val_dataset.get_generator()

    # model
    if args.model == "MixSTE2":
        model = MODEL_REGISTRY.get("MixSTE2")(
            num_frame=243,
            num_joints=17,
            in_chans=2.0,
            embed_dim_ratio=512,
            depth=8,
            num_heads=8,
            mlp_ratio=2,
            qkv_bias=True,
            drop_path_rate=0.1
        )
    elif args.model == "SimpleBaseline":
        model = MODEL_REGISTRY.get("SimpleBaseline")(

        )
    else:
        raise NotImplementedError
    
    # load ckpt
    ckpt_path = os.path.join(ROOT, "ckpt", args.ckpt)
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    #print(model.state_dict().keys())
    kvs = {".".join(k.split(".")[1:]):v for k,v in checkpoint["model_pos"].items()}
    model.load_state_dict(kvs, strict=True)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    print("Model initialized.")
    
    # generate data
    gen_train(generator=train_generator, model=model, data_type=args.keypoint_type_2d)
    gen_val(generator=val_generator, model=model, data_type=args.keypoint_type_2d, \
            kps_left=val_dataset.kps_left, kps_right=val_dataset.kps_right, \
            joints_left=val_dataset.joints_left, joints_right=val_dataset.joints_right, \
            receptive_field=args.receptive_field)