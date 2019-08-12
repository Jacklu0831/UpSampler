# Photo-Realistic SISR GAN

<pre>                      Input (LR)              Output (SR)           Ground Truth (HR)</pre>

<p align="center">
  <image src="assets/result_245.png" height="70%" width="70%"></image>
  <image src="assets/result_255.png" height="70%" width="70%"></image>
  <image src="assets/result_261.png" height="70%" width="70%"></image>
  <image src="assets/result_266.png" height="70%" width="70%"></image>
</p>

Implemented a **Photo-Realistic Single Image Super-Resolution Generative Adversial Network** (Tensorflow, Keras) that maps (64, 64, 3) image to size (256, 256, 3) while keeping the texture and perceptual details. The same model was trained separately on the COCO 2017 dataset and CelebA dataset, this was done to investigate how training SRGAN on domain specific dataset (faces) improve its performance on the domain of images it was trained on. The SRGAN network learns a mapping from the low-resolution patch through a series of convolutional, fully-connected, and transposed/upsampling convolutional layers into the high-resolution patch while keeping texture/perceptual details. Basically, I built and trained a deep neural network that asks for an image, then gives me back a clearer version of it. 

*GAN is hard to train, but SRGAN is even harder*. Google Colab provided me with free Tesla T4 GPU. However, even after decreasing the image size to free up storage and decreasing the training size to 2500 images, at 4+ min/epoch for 2500 epochs, the total training time was still more than a solid week for each of the COCO one and the face one. I highly recommend increasing the batch size and training size if you have access to stronger GPUs. For more details on the parameters I used, I made a pretty complete list of them in `parameters.txt`.

This is not really my first dip into GAN. For my previous work on making Celebrity Face Generator and seasonal CycleGAN, visit [this repo](https://github.com\Jacklu0831/GAN-Projects).

---

## Background + the Math

Invented by Ian GoodFellow in 2014, GAN showed amazing image generative abilities from road scenes to faces. However, generating images out of random noise is only a fraction of its capability. From switching images between domains (CycleGAN) to music generation (MidiNet), the breadth of tasks GAN could take on is still being rapidly discovered. Image super resolution can be defined as increasing the size of small images while keeping the drop in quality to minimum, or restoring high resolution images from rich details obtained from low resolution images. It has its applications in the fields of surveillance, forensics, medical imaging, satellite imaging, and consumer photography. Below is my attempt in concisely explaining SRGAN from [this paper](https://arxiv.org/abs/1609.04802).

 ### Neural Network Architecture

<p align="center"><image src="assets/architecture.png"></image></p>

The high level architecture of SRGAN is quite simple
1. High resolution (HR) ground truth images are selected from the training set
2. Low resolution (LR) images corresponding to the HR images are created with bi-cubic downsampling 
3. The generator upsamples the LR images to Super Resolution (SR) images
4. The discriminator distinguishes the HR images (ground truth) and the SR images (fake)
5. Backpropagate loss to train discriminator and generator

 ## Generator

<p align="center"><image src="assets/generator.png"></image></p>

The generator takes a LR image, process it with a conv and a PReLU (trainable LReLU) layer, puts it through 16 [residual blocks](https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec) borrowed from SRResNet, upsamples by factor of 2 twice, and puts it through one last conv layer to produce the SR image. Unlike normal convolutional layers, going crazy with the number of residual blocks is not prone to overfitting the dataset because of their ability of [identity mapping](https://arxiv.org/abs/1603.05027). 

### Discriminator

<p align="center"><image src="assets/discriminator.png"></image></p>

The discriminator is similar to a normal image classifier, except its task is now more daunting due to having to classify two images with near identical content (if the generator is trained well enough). It puts HR/SR images first through a conv and a LReLU layer, process the image through 7 conv-BN-LReLU blocks, flatten the image, then use two dense layers with a LReLU in middle and a sigmoid function at the end for binary classification. 

### Intuition

At first I thought despite identity mapping, 16 Residual layers (generator) and 7 discriminator blocks (discriminator) is an overkill for small images with length of 256. However, it makes sense because the generator needs to learn the features of an image well enough to the pointing of producing a more detailed version of it. On the other hand, the discriminator has to classify two images with increasingly identical content. 

### Loss Function

<p align="center"><image src="assets/goal.png" height="50%" width="50%"></image></p>

This equation above describes the goal of SRGAN - to find the generator weights/parameters that minimize the perceptual loss function averaged over a number of images. On the right side of the equation inside the summation, the perceptual loss function takes two arguments - a generated SR image by putting an LR image into the generator function, and the ground truth HR image. 

<p align="center"><image src="assets/gan_loss.png" height="40%" width="40%"></image></p>

One of the major advantage DNN approach has over other numerical techniques for single image super resolution is having the perceptual loss function for backpropagation. Let's break it down. It adds the content loss and 0.001 of the adversial loss together and minimize them. 

<table align="center">
  <tr>
    <th>
        <p align="center"><image src="assets/perceptual_loss.png" height="105" width="1000"></image></p>
    </th>
    <th>
        <p align="center"><image src="assets/feature.png" height="225" width="1300"></image></p>
    </th>
  </tr>
</table>

Content loss refers to the loss of perceptual similarity between the SR and HR images. For many years people use MSE by default for this. However, minimizing MSE often produces blurry images, to computer the images might be similar, but human eyes extracts features from images instead of making pixel-wise calculations. Therefore, I used the VGG19 network for feature extraction, then took the MSE of the extracted features instead. 

<p align="center"><image src="assets/adv_loss.png" height="40%" width="40%"></image></p>

Adversarial loss uses the classification results to calculate the loss of the generator. The formula is close but not identical to binary cross entropy for better gradient behavior. Instead, I used binary cross entropy but tweaked the label value of SR images from 0 to a normal distribution around 0.1 to assist the discriminator's learning speed.

---

## Files

> Code
- SRGAN_coco.ipynb                - Google Colab implementation (coco dataset)
- SRGAN_coco_continue.ipynb       - Google Colab implementation (coco dataset restore model and continue training)
- SRGAN_face.ipynb       		      - Google Colab implementation (face dataset)
- SRGAN_face_continue.ipynb       - Google Colab implementation (face dataset restore model and continue training)
- SRGAN_test.py                   - Script for testing the trained models
- utils.py                        - keeps all of the preprocess and data operation functions

> Others
- README.md                       - self
- loss.txt                        - losses components of each epoch
- parameters.txt                  - a complete list of hyperparameters and other parameters I used
- assets                          - images for README.md
- model                           - .h5 files of the coco and face generators, discriminators not included due to being almost 300 MB
- raw_data                        - 1000 raw images from the COCO 2017 dataset
- data.zip                        - proprocessed data from utils.py functions
- output                          - Bunch of images with the epoch number beside them

Since `SRGAN.ipynb` was getting way too long, I moved all the data preprocess and management functions into `utils.py` to emphasize more on the neural network in the notebook. Also, I ended up separating the process of data preparation and training by download the preprocessed data and uploading them to `SRGAN.ipynb` so you can just run it alone without any other files. 

Google Colab provided me with Tesla K80 GPU. At 2.19 min/epoch on 800 training images for 500 epochs, the total training time was around 18.5 hours. I used batch size of 16 but I highly recommand at least increasing it to 32 if your hardware allows.

---

## Problems Faced

Note to self:
slow GPU and low GPU storage (12gb), decrease size but train with more images on coco, train images from a specific domain for a specific task like faces. Also decreasing size -> increase batch size so more efficiency in general. eyes not properly detected. too much detail in image. dataset being too wild. Decrease face size to increase batch size.

___

## Try it Yourself

**Dependencies**

Python 3.6, Tensorflow 1.12.0, Keras 2.2.4, numpy 1.15.0, matplotlib, Pillow, tqdm

**Train**

Open `SRGAN_coco.ipynb` file or `SRGAN_face.ipynb`, upload `coco.zip` or `celeb.zip`, select run all. If error is encountered, please notify me jacklu0831@gmail.com.

**Play with the Trained Model**

Run script `SRGAN_test.py`, make sure image input and image output directories and `coco_g_model2500.h5` or `face_g_model2500.h5` paths are specified. Have fun!

---

## Papers

- [SRGAN: Training Dataset Matters](https://arxiv.org/abs/1903.09922)
- [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)
