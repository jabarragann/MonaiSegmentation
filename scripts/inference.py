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
from pathlib import Path

import monai
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, MeanIoU
from monai.networks.nets import FlexibleUNet

from monai.utils import set_determinism
from monai.visualize.utils import blend_images


set_determinism(seed=0)
###########################
#### Data Loading #########
###########################

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

###########################
#### Model Loading ########
###########################

dst_dir = Path("/home/juan1995/research_juan/accelnet_grant/monai_tutorials")
dst_dir = dst_dir / "data"
device = "cuda"

pretrained_weights = monai.bundle.load(
    name="endoscopic_tool_segmentation", bundle_dir=dst_dir, version="0.2.0"
)

model = FlexibleUNet(
    in_channels=3,
    out_channels=1,
    backbone="efficientnet-b0",
    pretrained=True,
    is_pad=False,
).to(device)


loss_function = DiceLoss(sigmoid=True)

model_weight = model.state_dict()
weights_no_head = {k: v for k, v in pretrained_weights.items() if not "segmentation_head" in k}
model_weight.update(weights_no_head)
model.load_state_dict(model_weight)

for l in model.parameters():
    l.requires_grad = False

for l in model.segmentation_head.parameters():
    l.requires_grad = True

print("model loading successful")

###########################
#### Inference     ########
###########################


def infer_seg(im, model):
    """Infer single model and threshold."""
    inferred = model(im[None]) > 0
    return inferred[0].detach()


nrow, ncol = 4, 2
fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 3), facecolor="white")
nexamples = nrow
frames = sorted(np.random.choice(len(ds), size=nexamples, replace=False))

with torch.no_grad():
    for row, frame in enumerate(frames):
        _ds = ds[frame]
        img, lbl = _ds["image"], _ds["label"]
        inferred = infer_seg(img.to(device), model).cpu()
        # inferred = np.expand_dims(inferred[1], 0)

        for col, (_lbl, title) in enumerate(zip((lbl, inferred), ("GT", "inferred"))):
            blended = blend_images(img, _lbl, cmap="viridis", alpha=0.2)
            blended = np.moveaxis(blended, 0, -1)  # RGB to end
            axes[row, col].imshow(blended)
            axes[row, col].set_title(f"Frame: {frame} ({title})")
            axes[row, col].axis("off")

plt.show()
