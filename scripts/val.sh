export CUDA_LAUNCH_BLOCKING=1
nohup python /root/Improve-HPE-with-Diffusion/main.py \
--phase 'val' \
--gpu_ids 1 \
--config /root/Improve-HPE-with-Diffusion/config/fixed_res_and_diff_val.json >/root/Improve-HPE-with-Diffusion/logs/val_fixed_clip.log 2>& 1&