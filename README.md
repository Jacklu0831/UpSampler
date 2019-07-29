# SRGAN and Applications

**NOTE** - Below are some results produced in between epochs. I also to see if it helps with object detection and facial recognition, these applications will likely come later.

<p align="center">
  <b>&emsp;&emsp;&emsp;Input (LR)&emsp;&emsp;&emsp;&emsp;Output (SR)&emsp;&emsp;Ground Truth (HR)</b>
  <br>
  <image src="assets/result_236.png" height="50%" width="50%"></image>
  <image src="assets/result_245.png" height="50%" width="50%"></image>
  <image src="assets/result_255.png" height="50%" width="50%"></image>
  <image src="assets/result_261.png" height="50%" width="50%"></image>
  <image src="assets/result_266.png" height="50%" width="50%"></image>
</p>

Implemented a Photo-Realistic Single Image Super-Resolution Generative Adversial Network (Tensorflow, Keras) that maps (64, 64, 3) image to (256, 256, 3). Trained it on Google Colab and used the COCO 2017 dataset. The SRGAN network learns a mapping from the low-resolution patch through a series of convolutional, fully-connected, and transposed/upsampling convolutional layers into the high-resolution patch while keeping texture/perceptual details. Basically, I built and trained a deep neural network that asks for a video or image, then give me back a clearer version of it. 

Check parameters.txt for the (hyper)parameters I used for training. Google Colab provided me with Tesla T4 GPU. At 2.18 min/epoch for 500 epochs, the total training time is around 18 hours. I highly recommand increasing the batch size if you have access to stronger GPUs. If you notice that the SRGAN.ipynb has quite a lot of functional programming, it because I originally implemented this project in python scripts but moved everything to Colab for more convenience and visualizing images.

This is not my first dip in GAN. For my previous work on Celebrity Face Generator and CycleGAN, visit [this repo](https://github.com\Jacklu0831/GAN-Projects).

---

## Background + the Math

Invented by Ian GoodFellow in 2014, GAN showed amazing image generative abilities from road scenes to faces. However, generating images out of random noise is only a fraction of its capability. From switching images between domains (CycleGAN) to music generation (MidiNet), the breadth of tasks GAN could take on is still being rapidly discovered. Image super resolution can be defined as increasing the size of small images while keeping the drop in quality to minimum, or restoring high resolution images from rich details obtained from low resolution images. It has its applications in the fields of surveillance, forensics, medical imaging, satellite imaging, and consumer photography. 

**Neural Network Architecture**

<p align="center"><image src="assets/architecture.png"></image></p>

The architecture of SRGAN is quite simple
1. High resolution (HR) ground truth images are selected from the training set
2. Low resolution (LR) images corresponding to the HR images are created with bi-cubic downsampling 
3. The generator upsamples the LR images to Super Resolution (SR) images
4. The discriminator distinguishes the HR images (ground truth) and the SR images (fake)
5. Backpropagate loss to train discriminator and generator

**Generator**

<p align="center"><image src="assets/generator.png"></image></p>

The generator takes a LR image, process it with a conv and a PReLU (trainable LReLU) layer, puts it through 16 [residual blocks](https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec) borrowed from SRResNet, upsamples by factor of 2 twice, and puts it through one last conv layer to produce the SR image.  

**Discriminator**

<p align="center"><image src="assets/discriminator.png"></image></p>

The discriminator is similar to a normal image classifier, except its task is now more daunting due to having to classify two images with near identical content (if the generator is trained well enough). It puts HR/SR images first through a conv and a LReLU layer, process the image through 7 conv-BN-LReLU blocks, flatten the image, then use two dense layers with a LReLU in middle and a sigmoid function at the end for binary classification. 

**Intuition** 

At first I thought despite identity mapping, 16 Residual layers (generator) and 7 discriminator blocks (discriminator) is an overkill for small images with length of 256. However, it makes sense because the generator needs to learn the features of an image well enough to the pointing of producing a more detailed version of it. On the other hand, the discriminator has to classify two images with increasingly identical content. 

**Loss Function**

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
        <p align="center"><image src="assets/feature.png" height="225" width=1300"></image></p>
    </th>
  </tr>
</table>

Content loss refers to the loss of perceptual similarity between the SR and HR images. For many years people use MSE by default for this. However, minimizing MSE often produces blurry images, to computer the images might be similar, but human eyes extracts features from images instead of making pixel-wise calculations. Therefore, I used the VGG19 network for feature extraction, then took the MSE of the extracted features instead. 

<p align="center"><image src="assets/adv_loss.png" height="40%" width="40%"></image></p>

Adversarial loss uses the classification results to calculate the loss of the generator. The formula is close but not identical to binary cross entropy for better gradient behavior. Instead, I used binary cross entropy but tweaked the label value of SR images from 0 to a normal distribution around 0.1 to achieve the same effect.

---

## Files

- SRGAN.ipynb         - Google Colab (Jupyter Notebook) implementation
- README.md           - `self`
- loss.txt            - losses each epoch
- assets              - images for README.md
- model               - .h5 files of Generator and Discriminator
- raw_data            - 1000 raw images from the COCO 2017 dataset
- processed_data
  - high_res_images   - 1000 preprocessed high resolution images
  - low_res_images    - 1000 preprocessed low resolution images
- output              - Bunch of images with the epoch number beside them

---

## Try it Yourself

**Dependencies**

Python 3.6, Tensorflow 1.12.0, Keras 2.2.4, numpy 1.15.0, matplotlib, Pillow, tqdm
* If using Google Colab, all of the dependencies should be automatically satisfied except tqdm (for now at least).

**Run**

Simply open the SRGAN.ipynb file in Google Colab and run all. If error encountered, please notify me jacklu0831@gmail.com.

---

## Papers

- [SRGAN: Training Dataset Matters](https://arxiv.org/abs/1903.09922)
- [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

               
