import mediapy as media
import einops
import numpy as np
import torch
import torch.nn.functional as F

root = "/home/ubuntu11/zzc/code/videoprediction/pred-vdm/logs_training/20230405-042152_kth-ldm-vq-f4/videos/test/step-022400(epoch-005599)/batch-000000"

grounds=[]
samples=[]
for i in range(8):
    ground = media.read_video(f"{root}/x_origin/video-{i:06}.gif")
    grounds.append(ground)
    sample = media.read_video(f"{root}/x_sample/video-{i:06}.gif")
    samples.append(sample)
grounds = np.stack(grounds)
samples = np.stack(samples)

grounds = torch.tensor(grounds)
samples = torch.tensor(samples)

grounds = F.pad(grounds, (0, 0, 2, 0, 2, 2), 'constant', 255)
samples = F.pad(samples, (0, 0, 0, 2, 2, 2), 'constant', 255)

# output add white border

print(grounds.shape)
print(samples.shape)
all = torch.cat([grounds, samples])
# b t h w c

all = einops.rearrange(all, "(b1 b2) t h w c -> t (b2 h) (b1 w) c", b1=2) 
all = einops.rearrange(all, "t (a h) w c -> t h (a w) c", a=4) 
all = F.pad(all, (0, 0, 2, 2, 2, 2), 'constant', 255)
media.write_video(f"{root}/result-predvdm.gif", all.numpy(), fps=20, codec='gif')
