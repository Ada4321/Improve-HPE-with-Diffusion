export CUDA_LAUNCH_BLOCKING=1
nohup python /root/Improve-HPE-with-Diffusion/main.py \
--phase 'train' \
--gpu_ids 1 \
--config /root/Improve-HPE-with-Diffusion/config/mixste/two_stage_mixste_h36m_train.json >/root/Improve-HPE-with-Diffusion/logs/train_h36m.log 2>& 1&