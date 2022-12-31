import torch
import monai
import os
from monai.apps import download_url
from pathlib import Path


data_dir = "/home/juan1995/research_juan/accelnet_grant/monai_tutorials/data"
data_dir = Path(data_dir)

if not data_dir.exists():
    print("no data directory")
    exit(0)

endo_dir = os.path.join(data_dir, "endo_vid")


vid_url = "https://github.com/rijobro/real_time_seg/raw/main/example_data/EndoVis2017/d1_im.mp4"
vid_hash = "9b103c07326439b0ea376018d7189384"
seg_url = "https://github.com/rijobro/real_time_seg/raw/main/example_data/EndoVis2017/d1_seg.mp4"
seg_hash = "da4eb17b6f8e4155fd81b962d46e5eff"
vid_filepath = os.path.join(endo_dir, "endo.mp4")
seg_filepath = os.path.join(endo_dir, "endo_seg.mp4")

download_url(vid_url, vid_filepath, vid_hash)
download_url(seg_url, seg_filepath, seg_hash)
