import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from models.temperal.TemporalShift import TemporalShift

import einops

class TemperalFixModule(nn.Module):

  def __init__(self, temperal_channels=64, shift_conf=0.5, n_segment=5, n_div=8):
    super(TemperalFixModule, self).__init__()

    self.shift_conf = shift_conf
    self.temperal_channels = temperal_channels

    self.in_block = nn.Sequential(
      nn.Conv1d(in_channels=3, out_channels=temperal_channels, kernel_size=3, padding=1),
      nn.Conv1d(in_channels=temperal_channels, out_channels=temperal_channels, kernel_size=3, padding=1),
    )

    self.time_shift_block = TemporalShift(nn.Sequential(), n_segment=n_segment, n_div=n_div, inplace=False)

    self.out_block = nn.Sequential(
      nn.Conv1d(in_channels=temperal_channels, out_channels=3, kernel_size=3, padding=1)
    )


  def forward(self, x):
    b, _, _, h, w = x.shape

    out1 = einops.rearrange(x, "b t c h w -> (b h w) c t")   
    out1 = self.in_block(out1)

    in2  = einops.rearrange(out1, "(b h w) c t -> b t c h w ", b=b, h=h, w=w)
    out2 = self.time_shift_block(in2)
    out2 = einops.rearrange(out2, "b t c h w -> (b h w) c t")
    out2 = self.out_block(out2)
    out2 = einops.rearrange(out2, "(b h w) c t -> b t c h w ", b=b, h=h, w=w)

    out1 = self.out_block(out1)
    out1 = einops.rearrange(out1, "(b h w) c t -> b t c h w ", b=b, h=h, w=w)

    out = (1-self.shift_conf)*out1 + self.shift_conf*out2

    return out

model = TemperalFixModule(temperal_channels=64, n_segment=10)
input = th.randn(64, 5, 3, 16, 16)
output = model(input)
print(output.shape)