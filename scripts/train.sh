export CUDA_LAUNCH_BLOCKING=1
nohup python /root/Improve-HPE-with-Diffusion/main.py \
--phase 'train' \
--gpu_ids 0 \
--config /root/Improve-HPE-with-Diffusion/config/res_diff_plus.json >/root/Improve-HPE-with-Diffusion/logs/train_res_diff_plus.log 2>& 1&