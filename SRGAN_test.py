# python SRGAN_test.py -i i -o 1 -t 1

"""
TYPE:

1	SR
2	LR-SR-HR
3	COCO vs. FACE
4   COCO vs. BICUBIC
5	FACE vs. BICUBIC
6	
"""

import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob
import os

from keras.models import load_model, Model
from keras.applications.vgg19 import VGG19
import keras.backend as K

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="input image directory")
ap.add_argument("-o", "--output", required=True, help="output image directory")
ap.add_argument("-m1", "--model1", default="face_g_model2500.h5", help="generator model path (face)")
ap.add_argument("-m2", "--model2", default="coco_g_model2500.h5", help="generator model path (coco)")
ap.add_argument("-t", "--type", required=True, help="type of output image")
args = vars(ap.parse_args())


hr_ims = []
lr_ims = []
sr_ims = []
sr_ims_2 = []

# get Loss
def content_loss(y, y_pred):
    vgg19_model = VGG19(include_top=False, weights="imagenet", input_shape=(176, 176, 3))
    vgg19_model.trainable = False
    for layer in vgg19_model.layers:
        layer.trainable = False
    model = Model(inputs=vgg19_model.input, outputs=vgg19_model.get_layer("block5_conv4").output)
    return K.mean(K.square(model(y) - model(y_pred)))


# get and process images
def get_images():
	for img_name in glob.glob(os.path.join(args["input"], "*.jpg")):
		im = Image.open(img_name)
		hr_ims.append(np.asarray(im))

		im = np.asarray(im.resize((44, 44), Image.BICUBIC))
		lr_ims.append(im)

		im = np.divide(im.astype(np.float32), 127.5) - np.ones_like(im, dtype=np.float32)
		im = np.expand_dims(im, axis=0)
		im_f = np.asarray(g_1.predict(im))
		im_f = ((im_f + 1) * 127.5).astype(np.uint8)
		sr_ims.append(im_f[0])

		# also do the same for coco model
		if args["type"] == "3":
			im_c = np.asarray(g_2.predict(im))
			im_c = ((im_c + 1) * 127.5).astype(np.uint8)
			sr_ims_2.append(im_c[0])		


def plot_1_image():
	for i in range(len(sr_ims)):
		plt.imsave(os.path.join(args["output"], "{}.jpg".format(i+1)), sr_ims[i])

def plot_3_image():
	dim = (1, 3)
	for i in range(len(sr_ims)):
		plt.figure(figsize=(15, 5))
		plt.subplot(dim[0], dim[1], 1)
		plt.imshow(lr_ims[i], interpolation="nearest")
		plt.axis("off")
		plt.subplot(dim[0], dim[1], 2)
		plt.imshow(sr_ims[i], interpolation="nearest")
		plt.axis("off")
		plt.subplot(dim[0], dim[1], 3)
		plt.imshow(hr_ims[i], interpolation="nearest")
		plt.axis("off")
		plt.tight_layout()
		plt.savefig(os.path.join(args["output"], "{}.jpg".format(i+1)))
		plt.close()

def plot_face_vs_coco():
	dim = (1, 2)
	for i in range(len(sr_ims)):
		plt.figure(figsize=(10, 5))
		plt.subplot(dim[0], dim[1], 1)
		plt.imshow(sr_ims_2[i], interpolation="nearest")
		plt.axis("off")
		plt.subplot(dim[0], dim[1], 2)
		plt.imshow(sr_ims[i], interpolation="nearest")
		plt.axis("off")
		plt.tight_layout()
		plt.savefig(os.path.join(args["output"], "{}.jpg".format(i+1)))
		plt.close()

# def plot_face_vs_coco():
# 	dim = (1, 2)
# 	for i in range(len(sr_ims)):
# 		plt.figure(figsize=(10, 5))
# 		plt.subplot(dim[0], dim[1], 1)
# 		plt.imshow(sr_ims_2[i], interpolation="nearest")
# 		plt.axis("off")
# 		plt.subplot(dim[0], dim[1], 2)
# 		plt.imshow(sr_ims[i], interpolation="nearest")
# 		plt.axis("off")
# 		plt.tight_layout()
# 		plt.savefig(os.path.join(args["output"], "{}.jpg".format(i+1)))
# 		plt.close()

# def plot_face_vs_coco():
# 	dim = (1, 2)
# 	for i in range(len(sr_ims)):
# 		plt.figure(figsize=(10, 5))
# 		plt.subplot(dim[0], dim[1], 1)
# 		plt.imshow(sr_ims_2[i], interpolation="nearest")
# 		plt.axis("off")
# 		plt.subplot(dim[0], dim[1], 2)
# 		plt.imshow(sr_ims[i], interpolation="nearest")
# 		plt.axis("off")
# 		plt.tight_layout()
# 		plt.savefig(os.path.join(args["output"], "{}.jpg".format(i+1)))
# 		plt.close()


g_1 = load_model(args["model1"], custom_objects={'content_loss': content_loss})
if args["type"] == "3":
	g_2 = load_model(args["model2"], custom_objects={'content_loss': content_loss})

get_images()

if args["type"] == "1":
	plot_1_image()
elif args["type"] == "2":
	plot_3_image()
elif args["type"] == "3":
	plot_face_vs_coco()
elif args["type"] == "4":
	plot_coco_vs_bicubic()
elif args["type"] == "5":
	plot_face_vs_bicubic()
elif args["type"] == "6":
	pass