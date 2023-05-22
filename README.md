# PredVDM: A Video Prediction Model based on Latent Diffusion

![smmnist-compare-0](https://github.com/JunyaoHu/pred-vdm/assets/67564714/c7bb9951-5734-48cd-b334-f0efac9d789e)

![kth-compare-4](https://github.com/JunyaoHu/pred-vdm/assets/67564714/533a3293-50bc-4b76-ac10-f70b87590887)

# conda 环境配置
```

conda create -n pred-vdm python==3.9.15
conda activate pred-vdm

# CUDA 11.3
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# CUDA 11.6
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

conda install pytorch-lightning -c conda-forge
conda install h5py
conda install ffmpeg

pip install opencv-python==4.7.0.72 omegaconf mediapy einops wandb lpips progressbar scikit-image albumentations==1.3.0


pip install -e .
cd taming-transformers/
pip install -e .
cd ..

# 下载 vq-f4 模型
cd code/videoprediction/pred-vdm/
bash scripts/download_first_stages.sh

# 下载 fvd 模型 到 pred-vdm/models/fvd
# 需手动下载 联网有问题 任选一个即可
https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt
https://onedrive.live.com/download?cid=78EEF3EB6AE7DBCB&resid=78EEF3EB6AE7DBCB%21199&authkey=AApKdFHPXzWLNyI

```
# 训练推理
```
# [for training like]
# CUDA_VISIBLE_DEVICES=0,1 python main.py --base configs/smmnist64.yaml     -l /root/autodl-tmp/training_logs --train --gpus 0,1 -f 230515_test
# CUDA_VISIBLE_DEVICES=0,1 python main.py --base configs/kth64.yaml         -l /root/autodl-tmp/training_logs --train --gpus 0,1 -f 230515_test
# CUDA_VISIBLE_DEVICES=0,1 python main.py --base configs/cityscapes128.yaml -l /root/autodl-tmp/training_logs --train --gpus 0,1 -f test
# [for resume from a checkpoint like]
# CUDA_VISIBLE_DEVICES=0,1 python main.py --resume logs_training/20230220-213917_kth-ldm-vq-f4 --train --gpus 0,1
# [for test(sampling) like] wait for edit
# CUDA_VISIBLE_DEVICES=0 python main.py --resume /root/autodl-tmp/training_logs/smmnist64_230519_tcct_para1     --gpus 0, -f test
# CUDA_VISIBLE_DEVICES=1 python main.py --resume /root/autodl-tmp/training_logs/smmnist64_230522_tcct_para_attn --gpus 0, -f test
# CUDA_VISIBLE_DEVICES=0 python main.py --resume /root/autodl-tmp/training_logs/kth64_230516_baseline        --gpus 0, -f test
# CUDA_VISIBLE_DEVICES=0 python main.py --resume /root/autodl-tmp/training_logs/kth64_230518_ct              --gpus 0, -f test
# CUDA_VISIBLE_DEVICES=0 python main.py --resume /root/autodl-tmp/training_logs/kth64_230517_tcct_para       --gpus 0, -f test
# CUDA_VISIBLE_DEVICES=0 python main.py --resume /root/autodl-tmp/training_logs/kth64_230520_tcct_para_attn2 --gpus 0, -f test
# CUDA_VISIBLE_DEVICES=1 python main.py --resume /root/autodl-tmp/training_logs/kth64_230520_tcct_para_attn2 --gpus 0, -f test

# 主函数main.py
# 训练和推理进入到./ldm/models/diffusion/ddpm.py

# 数据格式、范围
save in h5 dataset: [0,255] uint8 numpy (b t c h w)
read in dataloader: [-1,1] float32 torch (b t c h w)

```
