# PredVDM: Video Diffusion Model Alleviating Error Accumulation in Prediction

introduction

architecture figure

prediction performance

prediction examples

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

pip install opencv-python omegaconf mediapy einops wandb lpips progressbar scikit-image albumentations

# mcvd need
pip install ninja prettytable 

# pred-vdm 还需要的步骤
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


# mcvd 涉及的主要文件
## 配置文件
mcvd/configs/kth64_big.yml
## 主函数
mcvd/main.py
## 运行函数
mcvd/runners/ncsn_runner.py
## 基于NCSNpp的diffusion结构
mcvd/models/better/ncsnpp_more.py
## 训练
CUDA_VISIBLE_DEVICES=0,1 python main.py --config ./configs/kth64_big.yml --data_path ../data/KTH/processed --exp checkpoints/kth64-cond10-pred5-my-train --seed 0 --ni
## 测试和采样
python code/videoprediction/mcvd/MCVD_demo_KTH.py

# mcvd 涉及的主要文件
## 配置文件
pred-vdm/configs/latent-diffusion/kth-ldm-vq-f4.yaml
## 主函数
pred-vdm/main.py
## 运行函数
mcvd/runners/ncsn_runner.py
## 基于NCSNpp的diffusion结构
mcvd/models/better/ncsnpp_more.py
## 训练
CUDA_VISIBLE_DEVICES=0 python main.py --base configs/latent-diffusion/kth-ldm-vq-f4.yaml --train --gpus 0,
## 恢复训练
CUDA_VISIBLE_DEVICES=0,1 python main.py --resume logs_training/20230220-213917_kth-ldm-vq-f4 --train --gpus 0,1
## 测试
CUDA_VISIBLE_DEVICES=0,1 python main.py --resume logs_training/20230220-213917_kth-ldm-vq-f4 --gpus 0,1
## 采样
CUDA_VISIBLE_DEVICES=0 python scripts/sample_diffusion.py -r models/ldm/kth_64/checkpoints/last.ckpt -l ./logs_sampling -n 20

# 数据格式、范围
save in h5 dataset: [0,255] uint8 numpy (b t c h w)

read in dataloader: [-1,1] float32 torch (b t c h w)