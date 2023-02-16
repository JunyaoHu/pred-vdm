import bisect
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
import mediapy as media

# for image dataset
import albumentations
from PIL import Image


class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


class VideoPaths(Dataset):
    def __init__(self, paths=None, start_idxs=None, end_idxs=None, trans=None, labels=None):
        self._length = len(paths)
        self._trans = trans

        if labels is None:
            self.labels = dict() 
        else:
            self.labels = labels

        self.labels["file_path"] = paths

        
        if start_idxs is not None and end_idxs is not None:
            self.given_start_and_end = True
            self.labels["start_idx"] = start_idxs
            self.labels["end_idx"] = end_idxs
        else:
            self.given_start_and_end = False

    def __len__(self):
        return self._length

    def preprocess_video(self, video_path, start_idx=None, end_idx=None):
        if start_idx is not None and end_idx is not None:
            video = media.read_video(video_path)[start_idx:start_idx+50]
        else:
            video = media.read_video(video_path)[:50]
        video = np.array(video).astype(np.uint8)
        
        tmp_video = []

        # select true frame idxs what we want
        for i in range(len(video)):
            tmp_video.append(self._trans(image=video[i])["image"])

        video = np.array(tmp_video)
        video = (video/127.5 - 1.0).astype(np.float32)
        # [0,255] -> [-1,1]
        return video

    def __getitem__(self, i):
        video = dict()
        if self.given_start_and_end:
            video["video"] = self.preprocess_video(self.labels["file_path"][i], int(self.labels["start_idx"][i]), int(self.labels["end_idx"][i]))
        else:
            video["video"] = self.preprocess_video(self.labels["file_path"][i])
        for k in self.labels:
            video[k] = self.labels[k][i]
        return video

class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path"] = paths
        self._length = len(paths)

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example
    
class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
