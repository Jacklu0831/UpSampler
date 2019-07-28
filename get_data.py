"""
Python script for getting COCO images that are big enough (400x400)
"""

import numpy as np
from PIL import Image
import cv2
import glob

coco = []        # COCO images
images = []      # images with the right size
size = 360       # size threshold

for img_name in glob.glob("coco/*.jpg"):
	img = np.asarray(Image.open(img_name))
	coco.append(img)

i = 0
while len(images) < 1000:
	shape = np.shape(coco[i])
	if len(shape) == 3:
		if shape[0]>=size and shape[1]>=size and shape[2]==3:
			images.append(coco[i])
	i += 1

for img in range(len(images)):
	images[img] = cv2.cvtColor(images[img], cv2.COLOR_BGR2RGB)
	cv2.imwrite("raw_data/{}.jpg".format(img+1), images[img])