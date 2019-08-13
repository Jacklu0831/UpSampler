# Super Resolution GAN

[table of contents that is soon to be made]

Built and trained a Photo-Realistic Single Image Super-Resolution Generative Adversial Network with Tensorflow + Keras. The aim is to map 64x64 colored images to 256x256 while keeping the perceptual details and texture. The motivation behind this project is out of self-interest: I really liked a wallpaper for my phone but it had low resolution, so I thought some kind of fully-convolutional encoder-decoder network would work and found out a lot of people have already tried this problem.  the main source of knowledge came from [this paper](https://arxiv.org/abs/1609.04802). Despite not having compatible hardware for how expensive it is to train the model, I was able to achieve great results by choosing smaller images, tweaking the model configuration, and lowering the batch size. 

## Brief Intro

#### Stage 1 - Building, Training, Failing
 
For the model architecture, I mainly constructed the model from the original paper and took some freedom with the number of residual blocks and the loss functions. A detailed explaination of the architecture components and how they come together is explained in the [Background Section](#Background-+-the-Math). 

Even Ian GoodFellow's friends thought GAN was not feasible for train, but SRGAN is even harder once you take a look at its [architecture](#Neural-Network-Architecture). I am incredibly grateful to Google for making their internal cloud computing engine [Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb#recent=true) free for all. However, even after decreasing the image size to free up storage and decreasing the training size to 2000 and testing size to 500 images, the provided T4 had to run at 4+ min/epoch for 2500 epochs, causing the total training time to be more than a week for each model. I highly recommend increasing the batch size and training size if you have access to stronger GPUs. For more details on the parameters I used, I made a pretty neat list of them in `parameters.txt`. I am quite fond of TF's format for model summary, so I put them into `model_summary.txt` to keep the notebook short.

#### Stage 2 - Performance Analysis

Two models with the same configuration were trained separately on the COCO dataset and the CelebA dataset, this was done to investigate how training SRGAN on domain specific dataset (faces) improve its performance on the domain of images it was trained on. The SRGAN network learns a mapping from the low-resolution patch through a series of convolutional, fully-connected, and transposed/upsampling convolutional layers into the high-resolution patch while keeping texture/perceptual details. Basically, I built and trained a deep neural network that asks for an image, then gives me back a clearer version of it. 

This is not really my first dip into GAN. For my previous work on making Celebrity Face Generator and seasonal CycleGAN, visit [this repo](https://github.com\Jacklu0831/GAN-Projects).

---

## Results

Below are a few test results (more in `results` folder) from COCO and CelebA datasets.

<pre> Low-Res Input      Super-Res Output        High-Res Ground Truth </pre>

#### COCO Results

<p align="center">
  <image src="assets/result_245.png" height="70%" width="70%"></image>
  <image src="assets/result_255.png" height="70%" width="70%"></image>
  <image src="assets/result_261.png" height="70%" width="70%"></image>
</p>

#### CelebA Results

<p align="center">
  <image src="assets/result_245.png" height="70%" width="70%"></image>
  <image src="assets/result_255.png" height="70%" width="70%"></image>
  <image src="assets/result_261.png" height="70%" width="70%"></image>
</p>

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

### Generator

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

One of the major advantage DNN approach has over other numerical techniques for single image super resolution is using the perceptual loss function for backpropagation. Let's break it down. It adds the content loss and 0.001 of the adversial loss together and minimize them. 

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

## My Experience

### Face Model vs. COCO Model

The idea of training the first model on a general dataset and the second model on a more restricted domain of images to observe if performance difference exist on the selected domain of images for the second model came to me while reading the original SRGAN paper. Then I came across [this paper](https://arxiv.org/abs/1903.09922) and decided to try out training one model on COCO and another on only human faces (CelebA) for the same number of epochs with the same model configuration. Below are some results.

[NEED UPDATE] 

### Problems Faced

Below are the collection of problems I encountered chronologically and my solutions for them. Take it either as some problem-solving tips or just a rant. 

**Problem 1: Learning Curve**

Being one of the newer applications of GAN when GAN is one of the newer neural architecture in the first place, resources on SRGAN was limited for me to implement my own. Thankfully, the [original paper](https://arxiv.org/abs/1609.04802) was very pleasant to read. It also contained a bundle of well-made images/diagrams I can't help but to use in this README. I carefully read the paper and several blogs to make sure I fully understand all functions before coding.

**Problem 2: Hardware**

I am incredibly grateful to Google for making their internal cloud computing technology [Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb#recent=true) free for all. However, the provided tesla T4 GPU was not quite powerful enough for training SRGAN. I lowered the originally intended LR and HR image sizes by a factor of 2 to allow for batch gradient descient. The training process failed multiple times due to the lack of computing power, GPU storage, and disconnection. However, these issues were resolved by decrease batch size, image sizes, manually uploading files from my local device, write outputs/models directly to my Google Drive every number of epochs, and having a separate file for continue training with saved models (see files section for more details). Persisting for 2500 epochs, I ended up learning so much more about using Google Colab and obtained great project results.

**Problem 3: Slow Training**

I carefully observed `face_loss.txt` and `coco_loss.txt` throughout the training process. The generator's perceptual loss steadily dereased throughout the 2500 epochs for both models. The good news is that it means the learning rate is not too big and the weights are indeed moving toward the Nash equilibrium. The bad news is that since the entire training took more than 2 weeks, it was difficult to know whether my hyperparameters were working, and each try means forfeiting a few days of training. [This blog](https://www.google.com/search?q=why+is+gan+hard+to+train&oq=why+is+gan+hard&aqs=chrome.0.69i59j69i60j69i57j0.1837j0j1&sourceid=chrome&ie=UTF-8) provides a nice explaination on why GAN is so hard to train compared to numerous other neural architectures.

**Problem 4: Struggles with Details**

I trained the first model on the COCO dataset and quickly noticed the issue of it performing atrociously with images with more details, which is because LR image not being able to capture the texture and perceptual details of its HR origin. Since human face is one of the most complex feature that can appear in a picture, I chose to train my second model completely on faces to observe how much I can push the performance on possibly the most complex features. Below is a side by Side comparison between the same model's performance on images with drastically different complexity.

[insert side by side comparison between details and non-details]

On the other hand, the model that is trained only face images were able to produce perceptually great faces by the 500th epoch. However, it struggled with the most detailed feature of human face, which are the eyes. Since the downsized images carry very less information for reconstructing the eyes of a person, it is mostly up to the generator for drawing on the eyes itself. Since eyes are actually very important for recognizing a face, I continuously trained the model and observed a gradually improvement in the generator's ability in reconstructing the eyes of people.

[insert face images]

---

## Files

> Code

<pre>
- SRGAN_coco.ipynb            - Google Colab implementation (coco dataset)
- SRGAN_coco_continue.ipynb   - Google Colab implementation (coco dataset restore model and continue training)
- SRGAN_face.ipynb            - Google Colab implementation (face dataset)
- SRGAN_face_continue.ipynb   - Google Colab implementation (face dataset restore model and continue training)
- SRGAN_test.py               - script for testing the trained models
- utils.py                    - some of image preprocess functions
</pre>

> Directories

<pre>
- assets                      - images for this README
- datasets                    - 2500 images from each of the COCO dataset and CelebA dataset
- final_models                - .h5 files of the coco and face generators, discriminators not included due to size (300+ MB)
- losses                      - files containing complete information on the training loss of each epoch
</pre>

> Others

<pre>
- README.md                   - self
- loss.txt                    - losses components of each epoch
- parameters.txt              - a complete list of hyperparameters and other parameters I used
- output                      - bunch of images with the epoch number beside them
</pre>

---

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
- [Why is GAN hard to train?](https://www.google.com/search?q=why+is+gan+hard+to+train&oq=why+is+gan+hard&aqs=chrome.0.69i59j69i60j69i57j0.1837j0j1&sourceid=chrome&ie=UTF-8)
