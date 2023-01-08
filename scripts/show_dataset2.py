from Datasets.VidDataset import VidDataset
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

data_dir = "/home/juan1995/research_juan/accelnet_grant/monai_tutorials/data/suturing_vid/suturing_recording01_right.avi"
data_dir = Path(data_dir)

if not data_dir.exists():
    print("no data directory")
    exit(0)


# Dataset
ds = VidDataset(data_dir)
dl = ThreadDataLoader(ds, batch_size=4, num_workers=0, shuffle=True)

print(f"Number of frames in vid: {len(ds)}")


nrow, ncol = 4, 3
fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 3), facecolor="white")
nexamples = nrow * ncol
frames = sorted(np.random.choice(len(ds), size=nexamples, replace=False))
for frame, ax in zip(frames, axes.flatten()):
    _ds = ds[frame]
    img, lbl = _ds["image"], _ds["label"]
    img = np.transpose(img, [1, 2, 0])
    # img = img[:,:,::-1]
    # blended = blend_images(img, lbl, cmap="viridis", alpha=0.2)
    # blended = np.moveaxis(blended, 0, -1)  # RGB to end
    ax.imshow(img)
    ax.set_title(f"Frame: {frame}")
    ax.axis("off")

plt.show()
