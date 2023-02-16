import os
import numpy as np
import albumentations as A
from torch.utils.data import Dataset

# from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex
from taming.data.base import VideoPaths


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
    def __init__(self, size, type, keys=None):
        super().__init__()
        self.trans = A.Compose([
            A.Resize(size,size),
        ])
        self.keys = keys

        root = "/home/ubuntu16/hjy/mcvd-pytorch/datasets/KTH/raw"
        with open(f"./data/KTH/{type}-2splits.txt", "r") as f:
            lines = f.read().splitlines()

        # it means only have file path if a line without ' '
        # like 'walking/person18_walking_d4_uncomp.avi'
        if ' ' not in lines[0]:
            paths = [os.path.join(root, relpath) for relpath in lines]
            self.data = VideoPaths(paths=paths, start_idxs=None, end_idxs=None, trans=self.trans)
            
        # it means a line should have filename, start_idx and end_idx
        # like 'walking/person18_walking_d4_uncomp.avi 1 80'
        else:
            assert len(lines[0].split()) == 3, "dataset should be [filepath] or [filepath, start_idx, end_idx]"
            paths, start_idxs, end_idxs = [], [], []
            for line in lines:
                path, start_idx, end_id = line.split()
                paths.append(os.path.join(root, path))
                start_idxs.append(start_idx)
                end_idxs.append(end_id)
            self.data = VideoPaths(paths=paths, start_idxs=start_idxs, end_idxs=end_idxs, trans=self.trans)

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
    dataset1 = KTH(64, "train")
    print(len(dataset1))

    dataset2 = KTH(64, "valid")
    print(len(dataset2))

    dataset3 = KTH(64, "test")
    print(len(dataset3))

    print(dataset1[0]['video'].shape)
