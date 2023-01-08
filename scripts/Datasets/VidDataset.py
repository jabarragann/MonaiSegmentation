from pathlib import Path
from torch.utils.data import Dataset
import torch
from monai.data.video_dataset import VideoFileDataset, CameraDataset
import monai.transforms as mt

default_vid_transforms = mt.Compose(
    [
        mt.DivisiblePad(32),
        mt.ScaleIntensity(),
        mt.CastToType(torch.float32),
    ]
)

default_seg_transforms = mt.Compose(
    [default_vid_transforms, mt.Lambda(lambda x: x[:1])]
)  # rgb -> 1 chan


class VidDataset(Dataset):
    def __init__(self, vid_file: Path, vid_transforms=None):
        self.vid_transforms = (
            vid_transforms if vid_transforms is not None else default_vid_transforms
        )
        vid_file = str(vid_file)
        self.ds_img = VideoFileDataset(vid_file, self.vid_transforms)

    def __len__(self):
        return len(self.ds_img)

    def __getitem__(self, idx):
        return {"image": self.ds_img[idx], "label": None}


class CombinedVidDataset(Dataset):
    def __init__(self, vid_filepath, seg_filepath):
        self.ds_img = VideoFileDataset(vid_filepath, default_vid_transforms)
        self.ds_lbl = VideoFileDataset(seg_filepath, default_seg_transforms)

    def __len__(self):
        return len(self.ds_img)

    def __getitem__(self, idx):
        return {"image": self.ds_img[idx], "label": self.ds_lbl[idx]}
