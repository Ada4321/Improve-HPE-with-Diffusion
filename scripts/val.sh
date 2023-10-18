export CUDA_LAUNCH_BLOCKING=1
nohup python /root/Improve-HPE-with-Diffusion/main.py \
--phase 'val' \
--gpu_ids 0 \
--config /root/Improve-HPE-with-Diffusion/config/mixste/two_stage_mixste_h36m_train_diff_u.json >/root/Improve-HPE-with-Diffusion/logs/val_h36m.log 2>& 1&