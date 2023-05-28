import einops
from torch.utils.data import Dataset
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


class VideoDataset(DatasetBase):
    def __init__(self, data_dir, frames_per_sample=15, total_videos=-1, augmentation_params=None, keys=None):
        super().__init__()
        self.keys = keys
        self.data = HDF5InterfaceDataset(data_dir, frames_per_sample, total_videos=total_videos, augmentation_params=augmentation_params)

if __name__ == "__main__":
    import mediapy as media

    # #################### SMMNIST ########################
    # dataset_root = "/root/autodl-tmp/data/SMMNIST/SMMNIST_h5"
    # dataset1 = VideoDataset(f"{dataset_root}/train", 20)
    # print(len(dataset1))
    # dataset2 = VideoDataset(f"{dataset_root}/test", 50, 256)
    # print(len(dataset2))
    # video = dataset1[len(dataset1)-1]['video']
    # video = einops.rearrange(video, "t c h w -> t h w c")

    # print(video.shape)
    # # [20, 64, 64, 1] [-1,1] float32
    # media.show_video(((video+1)/2).squeeze().numpy(), fps=20)

    # #################### KTH ########################
    # dataset_root = "/root/autodl-tmp/data/KTH/KTH_h5"
    # dataset1 = VideoDataset(f"{dataset_root}/train", 20)
    # print(len(dataset1))
    # dataset2 = VideoDataset(f"{dataset_root}/valid", 50, 256)
    # print(len(dataset2))
    # # video = dataset1[len(dataset1)-1]['video']
    # video = dataset2[47]['video']
    # video = einops.rearrange(video, "t c h w -> t h w c")
    # print(video.shape)
    # # [20, 64, 64, 1] [-1,1] float32
    # media.show_video(((video+1)/2).squeeze().numpy(), fps=20)

    #################### Cityscapes ########################
    # dataset_root = "/root/autodl-tmp/data/Cityscapes/Cityscapes_h5"
    # dataset1 = VideoDataset(f"{dataset_root}/train", 7)
    # print(len(dataset1))
    # dataset2 = VideoDataset(f"{dataset_root}/val", 50, 256)
    # print(len(dataset2))
    # video = dataset1[len(dataset1)-1]['video']
    # video = einops.rearrange(video, "t c h w -> t h w c")
    # print(video.shape)
    # # [20, 64, 64, 3] [-1,1] float32
    # media.show_video(((video+1)/2).squeeze().numpy(), fps=20)
    # video = dataset2[len(dataset2)-1]['video']
    # video = einops.rearrange(video, "t c h w -> t h w c")
    # print(video.shape)
    # # [20, 64, 64, 3] [-1,1] float32
    # media.show_video(((video+1)/2).squeeze().numpy(), fps=20)

    dataset_root = "/root/autodl-tmp/data/Cityscapes/Cityscapes_h5"
    augmentation_params={
        "flip_param":{
            "horizontal_flip": True,
            "time_flip": False,
        },
        "jitter_param":{
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.2,
        }
    }
    dataset1 = VideoDataset(f"{dataset_root}/train", 7, augmentation_params=augmentation_params)
    video = dataset1[len(dataset1)-1]['video']
    video = einops.rearrange(video, "t c h w -> t h w c")
    print(video.shape)
    # [20, 64, 64, 3] [-1,1] float32
    media.show_video(((video+1)/2).squeeze().numpy(), fps=20)

