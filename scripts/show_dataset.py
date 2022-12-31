import os
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import torch

# monai
import monai
from monai.data.video_dataset import VideoFileDataset, CameraDataset
from monai.data import ThreadDataLoader, decollate_batch
import monai.transforms as mt
from monai.utils import set_determinism
from monai.visualize.utils import blend_images


set_determinism(seed=0)
monai.config.print_config()

data_dir = "/home/juan1995/research_juan/accelnet_grant/monai_tutorials/data"
data_dir = Path(data_dir)

if not data_dir.exists():
    print("no data directory")
    exit(0)

endo_dir = os.path.join(data_dir, "endo_vid")


vid_filepath = os.path.join(endo_dir, "endo.mp4")
seg_filepath = os.path.join(endo_dir, "endo_seg.mp4")

# Dataset
vid_transforms = mt.Compose(
    [
        mt.DivisiblePad(32),
        mt.ScaleIntensity(),
        mt.CastToType(torch.float32),
    ]
)
seg_transforms = mt.Compose([vid_transforms, mt.Lambda(lambda x: x[:1])])  # rgb -> 1 chan


class CombinedVidDataset(Dataset):
    def __init__(self, img_file, seg_file, vid_transforms, seg_transforms):
        self.ds_img = VideoFileDataset(vid_filepath, vid_transforms)
        self.ds_lbl = VideoFileDataset(seg_filepath, seg_transforms)

    def __len__(self):
        return len(self.ds_img)

    def __getitem__(self, idx):
        return {"image": self.ds_img[idx], "label": self.ds_lbl[idx]}


ds = CombinedVidDataset(vid_filepath, seg_filepath, vid_transforms, seg_transforms)
dl = ThreadDataLoader(ds, batch_size=4, num_workers=0, shuffle=True)

print(f"Number of frames in vid: {len(ds)}")


nrow, ncol = 4, 3
fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 3), facecolor="white")
nexamples = nrow * ncol
frames = sorted(np.random.choice(len(ds), size=nexamples, replace=False))
for frame, ax in zip(frames, axes.flatten()):
    _ds = ds[frame]
    img, lbl = _ds["image"], _ds["label"]
    blended = blend_images(img, lbl, cmap="viridis", alpha=0.2)
    blended = np.moveaxis(blended, 0, -1)  # RGB to end
    ax.imshow(blended)
    ax.set_title(f"Frame: {frame}")
    ax.axis("off")

plt.show()
