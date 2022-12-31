from pathlib import Path
import monai
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, MeanIoU
from monai.networks.nets import FlexibleUNet

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

model.load_state_dict(pretrained_weights)

print("model loading successful")
