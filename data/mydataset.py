# import os
# import numpy as np
# import albumentations as A
from torch.utils.data import Dataset

# from data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex
# from data.base import VideoPaths, HDF5InterfaceDataset

from data.base import ConcatDatasetWithIndex
from data.base import HDF5InterfaceDataset

class DatasetBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.keys = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        ex = {}
        if self.keys is not None:
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example
        return ex


class KTH(DatasetBase):
    def __init__(self, data_dir, frames_per_sample=15, total_videos=-1, keys=None):
        super().__init__()
        self.keys = keys
        self.data = HDF5InterfaceDataset(data_dir, frames_per_sample, total_videos=total_videos)

# class FacesHQTrain(Dataset):
#     # CelebAHQ [0] + FFHQ [1]
#     def __init__(self, size, keys=None, crop_size=None, coord=False):
#         d1 = CelebAHQTrain(size=size, keys=keys)
#         d2 = FFHQTrain(size=size, keys=keys)
#         self.data = ConcatDatasetWithIndex([d1, d2])
#         self.coord = coord
#         if crop_size is not None:
#             self.cropper = albumentations.RandomCrop(height=crop_size,width=crop_size)
#             if self.coord:
#                 self.cropper = albumentations.Compose([self.cropper],
#                                                       additional_targets={"coord": "image"})

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, i):
#         ex, y = self.data[i]
#         if hasattr(self, "cropper"):
#             if not self.coord:
#                 out = self.cropper(image=ex["image"])
#                 ex["image"] = out["image"]
#             else:
#                 h,w,_ = ex["image"].shape
#                 coord = np.arange(h*w).reshape(h,w,1)/(h*w)
#                 out = self.cropper(image=ex["image"], coord=coord)
#                 ex["image"] = out["image"]
#                 ex["coord"] = out["coord"]
#         ex["class"] = y
#         return ex


# class FacesHQValidation(Dataset):
#     # CelebAHQ [0] + FFHQ [1]
#     def __init__(self, size, keys=None, crop_size=None, coord=False):
#         d1 = CelebAHQValidation(size=size, keys=keys)
#         d2 = FFHQValidation(size=size, keys=keys)
#         self.data = ConcatDatasetWithIndex([d1, d2])
#         self.coord = coord
#         if crop_size is not None:
#             self.cropper = albumentations.CenterCrop(height=crop_size,width=crop_size)
#             if self.coord:
#                 self.cropper = albumentations.Compose([self.cropper],
#                                                       additional_targets={"coord": "image"})

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, i):
#         ex, y = self.data[i]
#         if hasattr(self, "cropper"):
#             if not self.coord:
#                 out = self.cropper(image=ex["image"])
#                 ex["image"] = out["image"]
#             else:
#                 h,w,_ = ex["image"].shape
#                 coord = np.arange(h*w).reshape(h,w,1)/(h*w)
#                 out = self.cropper(image=ex["image"], coord=coord)
#                 ex["image"] = out["image"]
#                 ex["coord"] = out["coord"]
#         ex["class"] = y
#         return ex

if __name__ == "__main__":

    #################### KTH ########################
    dataset_root = "/home/ubuntu15/zzc/data/KTH/pred-vdm/processed"
    dataset1 = KTH(f"{dataset_root}/train", 20)
    print(len(dataset1))

    dataset2 = KTH(f"{dataset_root}/valid", 50, 256)
    print(len(dataset2))

    print(dataset1[len(dataset2)-1]['video'].shape)
    print(dataset2[len(dataset2)-1]['video'].shape)

    import mediapy as media
    video = dataset2[len(dataset2)-1]['video']
    # [-1,1] float32
    # torch.Size([20, 1, 64, 64])
    media.show_video(((video+1)/2).squeeze().numpy(), fps=20)

