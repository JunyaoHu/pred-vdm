import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from models.temperal.TemporalShift import TemporalShift

import einops

class TemperalFixModule(nn.Module):

  def __init__(self, temperal_channels=64):
    super(TemperalFixModule, self).__init__()

    self.channels = nn.Sequential(
      nn.Conv1d(in_channels=3, out_channels=temperal_channels, kernel_size=3, padding=1),
      nn.Conv1d(in_channels=temperal_channels, out_channels=temperal_channels, kernel_size=3, padding=1),
      nn.Conv1d(in_channels=temperal_channels, out_channels=3, kernel_size=3, padding=1)
    )

  def forward(self, x):
    b, _, _, h, w = x.shape

    x = einops.rearrange(x, "b t c h w -> (b h w) c t")
    x = self.channels(x)
    x = einops.rearrange(x, "(b h w) c t -> b t c h w ", b=b, h=h, w=w)

    return x

model = TemperalFixModule(temperal_channels=64)
input = th.randn(64, 10, 3, 16, 16)
output = model(input)
print(output.shape)