from pathlib import Path
import cv2
import os
import numpy as np
import torch 

import monai
from monai.data.video_dataset import VideoFileDataset
from monai.visualize.utils import blend_images
from monai.networks.nets import FlexibleUNet
from monai.utils import set_determinism

from Datasets.VidDataset import VidDataset


def infer_seg(im, model):
    """Infer single model and threshold."""
    inferred = model(im[None]) > 0
    inferred = inferred[0].detach().cpu()
    inferred = np.expand_dims(inferred[1], 0)

    return inferred


def create_video(ds, output_file, fps, codec, ext, check_codec=True):
    fourcc = cv2.VideoWriter_fourcc(*codec)
    if check_codec:
        codec_success = cv2.VideoWriter().open("test" + ext, fourcc, 1, (10, 10))
        if not codec_success:
            raise RuntimeError("failed to open video.")
        os.remove("test" + ext)

    print(f"{len(ds)} frames @ {fps} fps: {output_file}...")
    for idx in range(len(ds)):
        img = ds[idx]["image"]
        inferred = infer_seg(img.to(device), model)
        blended = blend_images(img, inferred, cmap="viridis", alpha=0.2)
        if idx == 0:
            width_height = blended.shape[1:][::-1]
            video = cv2.VideoWriter(output_file, fourcc, fps, width_height)
        blended = (np.moveaxis(blended, 0, -1) * 254).astype(np.uint8)
        blended = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
        video.write(blended)
    video.release()
    if not os.path.isfile(output_file):
        raise RuntimeError("video not created:", output_file)
    print("Success!")


device = "cuda"
codecs = VideoFileDataset.get_available_codecs()
codec, ext = next(iter(codecs.items()))
print(codec, ext)

## Load model
model = FlexibleUNet(
    in_channels=3,
    out_channels=2,
    backbone="efficientnet-b0",
    pretrained=True,
    is_pad=False,
).to(device)

dst_dir = Path("/home/juan1995/research_juan/accelnet_grant/monai_tutorials")
dst_dir = dst_dir / "data"
pretrained_weights = monai.bundle.load(
    name="endoscopic_tool_segmentation", bundle_dir=dst_dir, version="0.2.0"
)
model.load_state_dict(pretrained_weights)

## Data loading
data_dir = "/home/juan1995/research_juan/accelnet_grant/monai_tutorials/data/suturing_vid"
data_dir = Path(data_dir)
video_file = "suturing_recording01_right.avi"
if not data_dir.exists():
    print("no data directory")
    exit(0)
ds = VidDataset(data_dir / video_file)

# create video
fps = ds.ds_img.get_fps()
print(fps)
inferred_vid = os.path.join(data_dir, "inferred.mp4")

with torch.no_grad():
    create_video(ds, inferred_vid, fps, codec, ext)
    