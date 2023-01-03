from pathlib import Path
import monai
from monai.losses import DiceLoss
from monai.networks.nets import FlexibleUNet


monai.utils.set_determinism(seed=0)
monai.config.print_config()

dst_dir = Path("/home/juan1995/research_juan/accelnet_grant/monai_tutorials")
dst_dir = dst_dir / "data"
device = "cuda"


model = FlexibleUNet(
    in_channels=3,
    out_channels=1,
    backbone="efficientnet-b0",
    pretrained=True,
    is_pad=False,
).to(device)

loss_function = DiceLoss(sigmoid=True)


pretrained_weights = monai.bundle.load(
    name="endoscopic_tool_segmentation", bundle_dir=dst_dir, version="0.2.0"
)

model_weight = model.state_dict()
weights_no_head = {k: v for k, v in pretrained_weights.items() if not "segmentation_head" in k}
model_weight.update(weights_no_head)
model.load_state_dict(model_weight)

print("model loading successful")
