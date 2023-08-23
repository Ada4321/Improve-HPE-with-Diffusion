export CUDA_LAUNCH_BLOCKING=1
nohup python /root/Improve-HPE-with-Diffusion/main.py \
--phase 'train' \
--gpu_ids 0 \
--config /root/Improve-HPE-with-Diffusion/config/fixed_res_and_diff.json >/root/Improve-HPE-with-Diffusion/logs/train_fixed_rle_prestart_lr2e-3_explr.log 2>& 1&