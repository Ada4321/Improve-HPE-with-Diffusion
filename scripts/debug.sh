export CUDA_LAUNCH_BLOCKING=1
nohup python /root/Improve-HPE-with-Diffusion/main_debug.py \
--phase 'train' \
--gpu_ids 0 \
--sweep \
--config /root/Improve-HPE-with-Diffusion/config/fixed_res_and_diff_debug.json >/root/Improve-HPE-with-Diffusion/logs/debug_with_valset_1.log 2>& 1&