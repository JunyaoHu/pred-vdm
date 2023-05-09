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
    def __init__(self, data_dir, frames_per_sample=15, total_videos=-1, keys=None):
        super().__init__()
        self.keys = keys
        self.data = HDF5InterfaceDataset(data_dir, frames_per_sample, total_videos=total_videos)

if __name__ == "__main__":
    import mediapy as media

    #################### SMMNIST ########################
    dataset_root = "/root/autodl-tmp/SMMNIST/SMMNIST_h5"
    dataset1 = VideoDataset(f"{dataset_root}/train", 20)
    print(len(dataset1))
    dataset2 = VideoDataset(f"{dataset_root}/test", 50, 256)
    print(len(dataset2))
    video = dataset1[len(dataset1)-1]['video']
    # [20, 1, 64, 64] [-1,1] float32
    media.show_video(((video+1)/2).squeeze().numpy(), fps=20)

    #################### KTH ########################
    dataset_root = "/root/autodl-tmp/KTH/KTH_h5"
    dataset1 = VideoDataset(f"{dataset_root}/train", 20)
    print(len(dataset1))
    dataset2 = VideoDataset(f"{dataset_root}/valid", 50, 256)
    print(len(dataset2))
    video = dataset1[len(dataset1)-1]['video']
    # [20, 1, 64, 64] [-1,1] float32
    media.show_video(((video+1)/2).squeeze().numpy(), fps=20)

