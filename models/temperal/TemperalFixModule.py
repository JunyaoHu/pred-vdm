import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from models.temperal.TemporalShift import TemporalShift

import einops

class TemperalFixModule(nn.Module):

  def __init__(self, n_segment=5, n_div=8):
    super(TemperalFixModule, self).__init__()

    self.time_shift_block = TemporalShift(nn.Sequential(), n_segment=n_segment, n_div=n_div, inplace=False)
    self.conv = nn.Conv1d(in_channels=3, out_channels=3, kernel_size=3, padding=1)


  def forward(self, x):
    b, _, _, h, w = x.shape
    out1 = self.time_shift_block(x)
    out1 = einops.rearrange(x, "b t c h w -> (b h w) c t")   
    out1 = self.conv(out1)
    out1 = einops.rearrange(out1, "(b h w) c t -> b t c h w ", b=b, h=h, w=w)
    out = x + out1
    return out

# model = TemperalFixModule(n_segment=10)
# input = th.randn(64, 5, 3, 16, 16)
# output = model(input)
# print(output.shape)