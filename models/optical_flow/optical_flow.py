import numpy as np
import cv2
import torch
import einops

def optical_flow(video):
    if video.shape[-3] == 1:
        video  = einops.repeat(video, "b t c h w -> b t (n c) h w", n = 3)
    video  = einops.rearrange(video, "b t c h w -> b t h w c")
    # print("video", video.shape)
    batches = np.array(video.cpu())
    results = []
    for i in range(len(batches)):
        result = []
        for j in range(len(batches[i])-1):
            prvs = cv2.cvtColor(batches[i][j],cv2.COLOR_RGB2GRAY)
            next = cv2.cvtColor(batches[i][j+1],cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, pyr_scale=0.5, levels=5, winsize=15, iterations=3, poly_n=7, poly_sigma=1.5, flags=0)
            result.append(flow)
        results.append(result)
    results = torch.tensor(np.array(results), requires_grad=True)
    # print("results", results.shape)
    return results



