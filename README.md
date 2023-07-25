theme: minima

# 基于潜在扩散过程的视频预测模型

![smmnist-compare-2](https://github.com/JunyaoHu/pred-vdm/assets/67564714/004a8860-3194-4e43-8d86-cce0e29508f6)

![image](https://github.com/JunyaoHu/pred-vdm/assets/67564714/7a5cae3a-38d1-48ef-a313-369124f7fc3e)

![kth-compare-5](https://github.com/JunyaoHu/pred-vdm/assets/67564714/f9cb3774-81b6-4961-8d35-768904b30a33)

![image (2)](https://github.com/JunyaoHu/pred-vdm/assets/67564714/8951d9e8-7663-4107-9e7c-596cdf378f88)

![image](https://github.com/JunyaoHu/pred-vdm/assets/67564714/47841e09-41a2-4ccc-bae2-76ff628fefba)

![image (1)](https://github.com/JunyaoHu/pred-vdm/assets/67564714/2593a0c9-5ed9-4883-ad4d-cf66791cfd55)

![image](https://github.com/JunyaoHu/pred-vdm/assets/67564714/a559dc13-b095-4262-926d-9dc02d44ac9d)

![image](https://github.com/JunyaoHu/pred-vdm/assets/67564714/856b79ed-b6ac-4418-a5bc-51dd551f110b)


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
