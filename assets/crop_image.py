# ===========================
# crop images to desired size
# ===========================

import os
import cv2
import numpy as np
import glob
from PIL import Image
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input images")
ap.add_argument("-o", "--output", required=True, help="path to output images")
ap.add_argument("-w", "--width", required=True, help="width of cropped image")
ap.add_argument("-hi", "--height", required=True, help="height of cropped image")
ap.add_argument("-e", "--ext", default=".jpg", help="file extension (dot in front)")
args = vars(ap.parse_args())


data_dir = args["input"]
new_dir = args["output"]
ext = args["ext"]
new_width = int(args["width"])
new_height = int(args["height"])


def crop(img, new_w=new_width, new_h=new_height):
    "Crop center of image."
    h = img.shape[0]
    w = img.shape[1]
    assert h >= new_h and w >= new_w, img_name
    left = int(np.ceil((w - new_w) / 2))
    right = w - int(np.floor((w - new_w) / 2))
    top = int(np.ceil((h - new_h) / 2))
    bottom = h - int(np.floor((h - new_h) / 2))
    cropped_img = img[top:bottom, left:right, ...]
    return cropped_img


i = 1
for img_name in glob.glob(os.path.join(data_dir, "*" + ext)):
    img = np.asarray(Image.open(img_name))
    img = crop(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(new_dir, "{}".format(i) + ext), img)
    print(i, end=" ")
    i += 1
print()