export CUDA_LAUNCH_BLOCKING=1
nohup python /root/Improve-HPE-with-Diffusion/main_debug.py \
--phase 'train' \
--gpu_ids 1 \
--config /root/Improve-HPE-with-Diffusion/config/fixed_res_and_diff_debug.json >/root/Improve-HPE-with-Diffusion/logs/debug_with_valset.log 2>& 1&