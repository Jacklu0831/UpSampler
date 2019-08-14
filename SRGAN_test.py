"""
OUTPUT TYPES
1	Choose LR, SR or HR
2	LR - SR - HR
3	COCO - FACE
4   FACE - BICUBIC
5	COCO - BICUBIC
6&7 LR - BICUBIC - COCO - FACE - HR
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
		# HR
		im = Image.open(img_name)
		hr_ims.append(np.asarray(im))
		# LR
		im = np.asarray(im.resize((44, 44), Image.BICUBIC))
		lr_ims.append(im)
		# SR
		im = np.divide(im.astype(np.float32), 127.5) - np.ones_like(im, dtype=np.float32)
		im = np.expand_dims(im, axis=0)
		im_f = np.asarray(g_1.predict(im))
		im_f = ((im_f + 1) * 127.5).astype(np.uint8)
		sr_ims.append(im_f[0])
		# SR, second model
		if args["type"] == "3" or args["type"] == "6":
			im_c = np.asarray(g_2.predict(im))
			im_c = ((im_c + 1) * 127.5).astype(np.uint8)
			sr_ims_2.append(im_c[0])

# 1
def plot_sr():
	for i in range(len(sr_ims)/100):
		# plt.imsave(os.path.join(args["output"], "{}.jpg".format(i+1)), sr_ims[i])
		# plt.imsave(os.path.join(args["output"], "{}.jpg".format(i+1)), lr_ims[i])
		plt.figure(figsize=(5, 5))
		plt.imshow(lr_ims[i], interpolation="BICUBIC")
		plt.axis("off")
        plt.tight_layout()
		plt.savefig(os.path.join(args["output"], "{}.jpg".format(i+1)))
		plt.close()

# 2
def plot_lr_sr_hr():
	dim = (1, 3)
	for i in range(len(sr_ims)):
		plt.figure(figsize=(15, 6))
		plt.subplot(dim[0], dim[1], 1)
		plt.title("LR", fontsize=16)
		plt.imshow(lr_ims[i], interpolation="nearest")
		plt.axis("off")
		plt.subplot(dim[0], dim[1], 2)
		plt.title("SR", fontsize=16)
		plt.imshow(sr_ims[i], interpolation="nearest")
		plt.axis("off")
		plt.subplot(dim[0], dim[1], 3)
		plt.title("HR", fontsize=16)
		plt.imshow(hr_ims[i], interpolation="nearest")
		plt.axis("off")
		plt.tight_layout()
		plt.savefig(os.path.join(args["output"], "{}.jpg".format(i+1)))
		plt.close()

# 3
def plot_face_vs_coco():
	dim = (1, 4)
	for i in range(len(sr_ims)):
		plt.figure(figsize=(20, 6))
		plt.subplot(dim[0], dim[1], 1)
		plt.title("LR", fontsize=16)
		plt.imshow(lr_ims[i], interpolation="nearest")
		plt.axis("off")
		plt.subplot(dim[0], dim[1], 2)
		plt.title("COCO", fontsize=16)
		plt.imshow(sr_ims_2[i], interpolation="nearest")
		plt.axis("off")
		plt.subplot(dim[0], dim[1], 3)
		plt.title("CelebA", fontsize=16)
		plt.imshow(sr_ims[i], interpolation="nearest")
		plt.axis("off")
		plt.subplot(dim[0], dim[1], 4)
		plt.title("HR", fontsize=16)
		plt.imshow(hr_ims[i], interpolation="nearest")
		plt.axis("off")
		plt.tight_layout()
		plt.savefig(os.path.join(args["output"], "{}.jpg".format(i+1)))
		plt.close()

# 4, 5
def plot_gan_vs_bicubic():
	dim = (1, 4)
	for i in range(len(sr_ims)):
		plt.figure(figsize=(20, 6))
		plt.subplot(dim[0], dim[1], 1)
		plt.title("LR", fontsize=16)
		plt.imshow(lr_ims[i], interpolation="nearest")
		plt.axis("off")
		plt.subplot(dim[0], dim[1], 2)
		plt.title("BICUBIC", fontsize=16)
		plt.imshow(lr_ims[i], interpolation="BICUBIC")
		plt.axis("off")
		plt.subplot(dim[0], dim[1], 3)
		plt.title("CelebA" if args["type"] == "4" else "COCO", fontsize=16)
		plt.imshow(sr_ims[i], interpolation="nearest")
		plt.axis("off")
		plt.subplot(dim[0], dim[1], 4)
		plt.title("HR", fontsize=16)
		plt.imshow(hr_ims[i], interpolation="nearest")
		plt.axis("off")
		plt.tight_layout()
		plt.savefig(os.path.join(args["output"], "{}.jpg".format(i+1)))
		plt.close()

# 6, 7
def all():
	dim = (1, 5)
	for i in range(len(sr_ims)):
		plt.figure(figsize=(25, 6))
		plt.subplot(dim[0], dim[1], 1)
		plt.title("LR", fontsize=16)
		plt.imshow(lr_ims[i], interpolation="nearest")
		plt.axis("off")
		plt.subplot(dim[0], dim[1], 2)
		plt.title("BICUBIC", fontsize=16)
		plt.imshow(lr_ims[i], interpolation="BICUBIC")
		plt.axis("off")
		plt.subplot(dim[0], dim[1], 3)
		plt.title("COCO", fontsize=16)
		plt.imshow(sr_ims_2[i], interpolation="nearest")
		plt.axis("off")
		plt.subplot(dim[0], dim[1], 4)
		plt.title("CelebA", fontsize=16)
		plt.imshow(sr_ims[i], interpolation="nearest")
		plt.axis("off")
		plt.subplot(dim[0], dim[1], 5)
		plt.title("HR", fontsize=16)
		plt.imshow(hr_ims[i], interpolation="nearest")
		plt.axis("off")
		plt.tight_layout()
		plt.savefig(os.path.join(args["output"], "{}.jpg".format(i+1)))
		plt.close()


# load model based on types of operations
if args["type"] == "1" or args["type"] == "2" or args["type"] == "4":
	g_1 = load_model(args["model1"], custom_objects={'content_loss': content_loss})
elif args["type"] == "3" or args["type"] == "6":
	g_1 = load_model(args["model1"], custom_objects={'content_loss': content_loss})
	g_2 = load_model(args["model2"], custom_objects={'content_loss': content_loss})
elif args["type"] == "5":
	g_1 = load_model(args["model2"], custom_objects={'content_loss': content_loss})

# get and process images
get_images()

# call respective functions to create and save plots
if args["type"] == "1":
	plot_sr()
elif args["type"] == "2":
	plot_lr_sr_hr()
elif args["type"] == "3":
	plot_face_vs_coco()
elif args["type"] == "4" or args["type"] == "5":
	plot_gan_vs_bicubic()
elif args["type"] == "6" or args["type"] == "7":
	all()