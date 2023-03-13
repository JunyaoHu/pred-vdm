from os.path import join
import numpy as np

from models.motion_state.transforms import *
from models.motion_state.settings import *
from models.motion_state.visualize import *

import torch

def motion_state(batches, t_length, embed_dim):
    batches = np.array(batches.cpu())
    # ([64, 20, 1, 64, 64])
    # COMPUTE DESCRIPTORS
    # task 1
    trajectories = trajectories_from_video(batches)
    # trajectories = trajectories_from_video(batches, vis_flow=False, vis_trajectories=True)
    # trajectories = trajectories_from_video(batches, vis_flow=True, vis_trajectories=False)
    # trajectories = trajectories_from_video(batches)
    # task 2
    # get descriptors
    tt = descriptors_from_trajectories(trajectories, t_length, len_descriptor=embed_dim)
    tt = torch.tensor(tt)
    return tt
