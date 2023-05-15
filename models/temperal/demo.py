import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

conv1 = nn.Conv1d(in_channels=256,out_channels=100,kernel_size=3,padding=1)
input = torch.randn(32, 256, 35)
out = conv1(input)
print(out.size())