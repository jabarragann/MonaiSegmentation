from pathlib import Path
from typing import List
import cv2
import sys
from enum import Enum
import numpy as np


class ColorMap(Enum):
    Shaft = [54, 54, 54]


class ColorMapGray(Enum):
    Shaft = [48, 64]
    Gripper = [85, 95]


def extract_path_within_range(gray_img: np.ndarray, range: List[int]):
    result_img = np.zeros_like(gray_img)
    result_img[:, :] = ((gray_img[:, :] > range[0]) & (gray_img[:, :] < range[1])) * 255
    result_img[:, :] = result_img.astype(np.uint8)
    return result_img


img_path = Path("./data/ambf_data/initial_segmentations/foo.png").resolve()

print(ColorMap.Shaft.value)
img = cv2.imread(str(img_path))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# shaft_img = np.zeros_like(img)
# shaft_img[:, :, 0] = (img[:, :, 0] == ColorMap.Shaft.value[0]) * 254
# shaft_img = shaft_img.astype(np.uint8)

# shaft_img = np.zeros_like(img_gray)
# shaft_img[:, :] = (img_gray[:, :] == ColorMap.Shaft.value[0]) * 254
# shaft_img = shaft_img.astype(np.uint8)
shaft_img = extract_path_within_range(img_gray, ColorMapGray.Shaft.value)
gripper_img = extract_path_within_range(img_gray, ColorMapGray.Gripper.value)

result = np.clip(shaft_img + gripper_img, a_min=0, a_max=255)

print(img.shape)
cv2.imshow("img", img)
# cv2.imshow("img_gray", img_gray)
cv2.imshow("img_seg", result)
cv2.waitKey(0)

cv2.destroyAllWindows()
