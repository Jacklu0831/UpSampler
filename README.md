# Super-Resolution-GAN

There was a lot to write about this two stage project + investigation. You can navigate to specific sections with TOC.

[table of contents that is soon to be made]

---

## Introduction

### Motivation
Super resolution imaging has a wide application from satellite imaging to photography. However, the truth is I wanted to use an image as my mobile phone's wallpaper but it had low resolution, so the idea of some kind of fully-convolutional encoder-decoder network came to me and of course, I found out a number of papers have already been published around this topic. With previous experience in implementing CycleGAN and DCGAN ([GANs repo](https://github.com/Jacklu0831/GAN-Projects)), I comprehended some relevant [resources](#Sources), then started this project. 

### Summary
I built and trained a Photo-Realistic Single Image Super-Resolution Generative Adversial Network with Tensorflow + Keras and investigated the importance of the training dataset (using FID as the evaluation metric). Being able to extract enough information from only 44x44 photos with complex geometry (human face) to realistic 176x176 portraits, the final model has surpassed human-level performance. 

<pre>    Low-Res               Bicubic        Super-Res (Model A)  Super-Res (Model B)     High-Res GT </pre>
<p align="center"><image src="assets/CelebA_results/4.jpg"></image></p>
<pre>       |                     |____________________|_____________________|                      | 
       |                                          |                                            | 
  Input Image                        Ways to Increase Resolution                         Ground Truth </pre>             

**Stage 1** of this project was to map 44x44 colored images to 176x176 (upsampling factor of 4) while keeping the perceptual details and textures. The main source of knowledge came from the [original SRGAN paper](https://arxiv.org/abs/1609.04802) and [an analysis on the importance of dataset](https://arxiv.org/abs/1903.09922). Despite not having compatible hardware for the computational demanding models, I achieved great [results](#Results) by using smaller images, tweaking model configuration, and lowering batch size. I and my model found out the hard way that it was impossible for 44x44 images to capture the complexity of human face but the model persisted and pumped out visually convincing results. 

**Stage 2** of this project was an investigation to answer my own question of "whether the variance of dataset matter". I trained two identical models (Model A and Model B) separately on the COCO dataset (high content variance) and the CelebA dataset (specific category of content), then used visual results and Fréchet Inception Distance to both qualitatively and quantitatively demonstrate the results of my investigation. 

---

## Background

### A Contemporary History Lesson
Invented by Ian GoodFellow in [2014](https://arxiv.org/abs/1406.2661), GAN showed amazing image generative abilities from road scenes to faces. However, generating images out of random noise is only a fraction of its capability. From switching images between domains (CycleGAN) to music generation (MidiNet), the breadth of tasks GAN could take on is still being rapidly discovered. 

<p align="center"><image src="assets/gan.png"></image></p>

Single image super resolution (SISR) can be defined as upscaling images while keeping the drop in quality to minimum, or restoring high resolution images from details obtained from low resolution images. Traditionally, this was done with mostly bicubic interpolation ([great video explaining bicubic spline](https://www.youtube.com/watch?v=poY_nGzEEWM)). However, interpolation has the tendency of making images [blurry and foggy](#Results). Utilizing deep learning techniques, the SRGAN learns a mapping from the low-resolution patch through a series of convolutional, fully-connected, and transposed/upsampling convolutional layers into the high-resolution patch while keeping texture/perceptual details and greatly [surpassed the performance of any traditional methods](#Results). It has applications in the fields of surveillance, forensics, medical imaging, satellite imaging, and consumer photography. The hereby implemented SRGAN's usage also include [image colorization and edge to photo mapping](https://arxiv.org/abs/1903.09922).

Below is my attempt to concisely explain everything about SRGAN with some images/diagrams from the [original publish](https://arxiv.org/abs/1609.04802). You will realize that the model borrowed a lot of up to date techniques from various published papers, such as using residual networks and PReLU (parameterized ReLU).

### Neural Network Architecture

<p align="center"><image src="assets/architecture.png"></image></p>

The high level architecture of SRGAN closely resembles the vanilla GAN architecture and is also about reaching the [Nash Equilibrium in a zero-sum game](https://www.cs.toronto.edu/~duvenaud/courses/csc2541/slides/gan-foundations.pdf) between the generator and the discriminator.

1. High resolution (HR) ground truth images get selected from the training set
2. Low resolution (LR) images are created with bicubic downsampling 
3. The generator upsamples the LR images to Super Resolution (SR) images to fool the discriminator
4. The discriminator distinguishes the HR images (ground truth) and the SR images (output)
5. Discriminator and generator trained with backpropagation

### Generator

<p align="center"><image src="assets/generator.png"></image></p>

The generator takes a LR image, process it with a conv and a PReLU (trainable LReLU) layer, puts it through 16 [residual blocks](https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec) borrowed from SRResNet, upsamples by factor of 2 twice, and puts it through one last conv layer to produce the SR image. Unlike normal convolutional layers, going crazy with the number of residual blocks is not prone to overfitting the dataset because of [identity mapping](https://arxiv.org/abs/1603.05027). 

### Discriminator

<p align="center"><image src="assets/discriminator.png"></image></p>

The discriminator is similar to a normal image classifier, except its task is now more daunting due to having to classify two images with near identical content (if the generator is trained well enough). It puts HR/SR images first through a conv and a LReLU layer, process the image through 7 Conv-BN-LReLU blocks, flatten the image, then use two dense layers with a LReLU in middle and a sigmoid function at the end for binary classification. 

### Intuition
At first I thought despite identity mapping, 16 residual blocks and 7 discriminator blocks was an overkill to produce a 176x176 image. However, it does make sense because the generator needs to learn the features of an image well enough to the point of producing a more detailed version of it. On the other hand, the discriminator has to classify two images with increasingly identical content. On top of all these, quoting Ian GoodFellow ([source podcast](https://www.youtube.com/watch?v=Z6rxFNMGdn0)):

> The way of building a generative model for GANs is we have a two-player game in the game theoretic sense and as the players of this game compete, one of them becomes able to generate realistic data. 

Adversarial networks have to be constantly struggling against each other in this zero-sum game of GAN. The process of converging to the Nash Equilibrium can be extremely slow, that is one more reason for the [large number of trainable parameters](https://github.com\Jacklu0831/Super-Resolution-GAN/model_summary.txt).

### Loss Function

<p align="center"><image src="assets/goal.png" height="50%" width="50%"></image></p>

The equation above describes the goal of SRGAN - to find the generator weights/parameters that minimize the perceptual loss function averaged over a number of images. Inside the summation on the right side of the equation, the perceptual loss function takes two arguments - a generated SR image by putting an LR image into the generator function, and the ground truth HR image. 

<p align="center"><image src="assets/gan_loss.png" height="40%" width="40%"></image></p>

One of the major advantage DNN approach has over other numerical techniques for single image super resolution is using the perceptual loss function for backpropagation. It adds the content loss and 0.1% of the adversial loss together then minimize them. Let's break it down.

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

Content loss refers to the loss of perceptual similarity between the SR and HR images. For many years people use MSE by default instead of this. However, minimizing MSE often produces blurry images as it is only based on pixel-to-pixel similarity, to computer the images might be similar, but human eyes extracts features from images instead of making pixel-wise calculations. Therefore, the original paper used the VGG19 network for feature extraction, then took the MSE of the extracted features instead. 

<p align="center"><image src="assets/adv_loss.png" height="40%" width="40%"></image></p>

Adversarial loss uses the classification results to calculate the loss of the generator. The formula provided by the paper is an augmented version of BCE loss for better gradient behavior. Instead, I chose to stick with BCE loss but tweaked the label value of SR images from 0 to a normal distribution around 0.1 to assist the discriminator's learning speed.

---

## Procedures and Challenges
This section contains an overview of what I did, the problems I faced, and my solutions for overcoming them or at least mitigate them. I'll be referencing the [background section](#Background) quite a bit.

### Stage 1 - Preprocessing
Preprocessing images from the COCO and CelebA datasets contain identical steps. I randomly selected images from each dataset, **cropped the center out** of raw images to serve as the high resolution data (ground truth), **downsized** them with Pillow's built in **bicubic downsampling** method to serve as the low resolution data (input), and **normalized** them before feeding them into the model. The code for these are in the beginning of the notebooks and in `utils.py`.

### Stage 2 - Building
Being one of the newer applications of GAN when GAN is one of the newer neural architecture in the first place, the number of papers and blogs on it were limited. Thankfully, the [original paper](https://arxiv.org/abs/1609.04802) was very informative and did not contain any steep learning curves. For the model architecture, I mainly constructed the model from the original paper and experimented with the number of residual blocks and the loss functions (I ended up using BCE and tweaking the label value). 

Refer to the [background section](#Background) for some detailed explanation of the architecture components and how they come together. For details on the parameters I used, I made a pretty neat list of them in `parameters.txt`. I also am quite fond of Keras' format for model summary especially compared to PyTorch's, I put the summaries in `model_summary.txt` to keep the notebooks short. 

### Stage 3 - Training

**Hardware**

Even the creator of GAN, Ian Goodfellow himself would have thought of it as an infeasible idea due to having simultaneously training two networks if he was not drunk ([podcast](https://www.youtube.com/watch?v=Z6rxFNMGdn0), around 27 mins). Unfortunately, my experience showed that SRGAN is even worse because of its [heavy and complex model configuration](#Neural-Network-Architecture). 

The actual training process failed multiple times due to the lack of computing power, GPU storage, and disconnection. These issues were resolved by:

> Decreasing batch size, decreasing image sizes, uploading files from my local device, write outputs and models directly to Drive every few epochs, and having [a continue training file](#File-Tree) with saved models (see files section for more details).

This is one of the most problematic project and I ended up learning a lot more about using Cloud Computing in general and obtained great results. After decreasing the image size and the training size to 2500 images (train + test), the provided T4 GPU ran at 4+ min/epoch and the total training time was more than a week for each model (2500 epochs).

**Loss Analysis**

<p align="center">Loss of Model A (COCO dataset)</p>
<p align="center"><image src="assets/COCO_loss.jpg"></image></p>
<pre>     _________________________________________________________/           \__________________________
    /                                 Expand epoch 1500 to 1750 (sorry about the font size)          \
</pre>
<p align="center"><image src="assets/COCO_loss_zoomed.jpg"></image></p>

<p align="center">Loss of Model B (CelebA dataset)</p>
<p align="center"><image src="assets/CelebA_loss.jpg"></image></p>
<pre>     _________________________________________________________/           \__________________________
    /                                                 Expand epoch 1500 to 1750                      \
</pre>
<p align="center"><image src="assets/CelebA_loss_zoomed.jpg"></image></p>

The script for parsing `face_loss.txt` and `coco_loss.txt` is inside `utils.py`. I carefully observed the losses to ensure that neither the generator nor the discriminator is dominating the game. The generator loss steadily dereased throughout the 2500 epochs for both models trained on COCO and CelebA. Indicating that gradient descent is not overshooting and the the model weights are moving toward the Nash equilibrium. However, tuning hyperparameter was hard since each try forfeits days of training progress. [This blog](https://www.google.com/search?q=why+is+gan+hard+to+train&oq=why+is+gan+hard&aqs=chrome.0.69i59j69i60j69i57j0.1837j0j1&sourceid=chrome&ie=UTF-8) explains why GAN is harder to train than most other neural architectures.

### Stage 4 - Performance Analysis

**Model A (Trained on COCO)**

I trained the first model on the COCO dataset and quickly noticed the issue of it performing atrociously with details in images due to 44x44 image not capturing enough texture and perceptual details (fur, patterns, etc). Since human face is the most complex feature that can appear in a picture, Model A's performance on it is often absolutely atrocious. Since I already wanted to investigate whether how the variance of dataset affect a model's performance, I chose to train my second model completely on faces with the CelebA dataset to observe just how much I can push the generator to extract the complex feature of human face packed inside a 44x44 image. Below is a side by side comparison between the same model's performance on images with different complexity.

[insert side by side comparison between details and non-details]

**Model B (Trained on CelebA)**

The model that is trained only face images were able to produce perceptually great faces by the 500th epoch. However, it struggled with the most detailed and a very important feature of human face - eyes. Since the downsized images compress eyes into only few black pixels, reconstructing the eyes of people was impossible even for humans. Gradually, the generator learned what eyes look like and "drew" them onto the black pixels. Since eyes are actually very important for recognizing a face, I continuously trained the model and observed a gradual improvement in the generator's ability in reconstructing/creating the eyes of people. Additionally, teeth gaps, heavy makeup, and creases have also been lost in when the HR images were downsized.

[insert face images]

The struggles with details can be traced back to not having more powerful hardware for processing bigger images and bicubic interpolation not being the optimal downsampling method for retaining perceptual information. 

- Performance Comparison between Bicubic, Model A, and Model B

> If I train model A with a variety of objects and model B with only one category/type of images (dataset with narrower domain), say cats. Would B perform better than A on cat images or is SRGAN only about recognizing small textures and edges as detailed as possible? 

I asked this question on Quora and received no response :( and only later found out about [this paper](https://arxiv.org/abs/1903.09922), so I clarified my own question through experimentation. In the paper just mentioned, the researchers trained models on different categories of images (face, dining room, tower) to demonstrate that each model performs best on the category of images they were trained on with FID as the evaluation metric. However, my question was when model A gets trained on images with high variance (COCO) and model B gets trained with images in a narrower domain (CelebA), is there a performance difference between A and B evaluated on the images from the narrower domain. 

Therefore, I trained two models with the same configuration separately on the COCO dataset and the CelebA dataset for the same number of epochs, this was done to investigate how training SRGAN on a narrow domain of images (faces) improve its performance on the domain of images that it was trained on. I also used FID to evaluate my models.

[insert stuff about FID]
[insert coco vs face]

Final note: Despite that Model B was able to perform much better on the CelebA testing set than Model A was able to perform on anything specific. Model A was able to generalize into a much wider range of image contents and perform better on all of them. In addition, the FID measures showed that it was able to perform much better than bicubic upsampling even on the face images. 

---

## Results
Below are some test results from both COCO and CelebA datasets. A few were included in previous sections and more can be found in the `results` dir. Go to [this section](#Try-it-Yourself) to try your own images.

### COCO
<pre>    Low-Res               Bicubic        Super-Res (Model A)  Super-Res (Model B)     High-Res GT </pre>

<p align="center">
  <image src="assets/CelebA_results/1.jpg"></image>
  <image src="assets/CelebA_results/2.jpg"></image>
  <image src="assets/CelebA_results/3.jpg"></image>
</p>

### CelebA

<p align="center">
  <image src="assets/COCO_results/1.jpg"></image>
  <image src="assets/COCO_results/2.jpg"></image>
  <image src="assets/COCO_results/3.jpg"></image>
</p>

---

## File Tree

### Files

<pre>
- SRGAN_coco.ipynb            - Colab implementation (COCO)
- SRGAN_coco_continue.ipynb   - Colab implementation (COCO restore model and continue training)
- SRGAN_face.ipynb            - Colab implementation (CelebA)
- SRGAN_face_continue.ipynb   - Colab implementation (CelebA restore model and continue training)
- SRGAN_test.py               - script for testing the trained models with various types of outputs
- FID.py                      - script for calculating the Fréchet Inception Distance between 2 image distributions
- utils.py                    - miscellaneous functions for plotting performance, managing/process data, parsing loss files...
- README.md                   - self
</pre>

### Directories

<pre>
- assets                      - images for this README
- input                       - train + test images from each of the COCO dataset and CelebA dataset
- output                      - some randomly chosen results of the 2500 epoch generators on the test dataset
    - CelebA                  - 20 CelebA results with all 5 image types
    - COCO                    - 20 COCO results with all 5 image types
- model                       - .h5 files of the 2 generators, discriminators not included (300+ MB)
- loss                        - files containing complete information on the training loss of each epoch with plots
- info                        - information about model configuration and parameters/hyperparameters
</pre>

---

## Try it Yourself

### Dependencies

<pre>
Necessary (not version specific)      Unnecessary
- Notebook/Colab (virtual env)        - tqdm
- Python 3.7                          - OpenCV (utils.py)
- Tensorflow 1.14.0		      - Pillow
- Keras 2.2.4
- numpy 1.15.0
- matplotlib
</pre>

### Train

Open `SRGAN_coco.ipynb` or `SRGAN_face.ipynb`, upload `COCO.zip` or `CelebA.zip`, make sure path names are correct and `shift + enter` away. If you encounter any confusion, feel free to shoot me an email.

### Try Your Own Images

Use the script `SRGAN_test.py`. Make sure the input and output directories and the generator (`coco_g_model_2500.h5` or `face_g_model_2500.h5`) paths are correctly specified. There are quite a few types of outputs you can customize, read the top of the script file to know the ID of the output type you wish for.

---

## Sources

### Papers

[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)

[SRGAN: Training Dataset Matters](https://arxiv.org/abs/1903.09922)

[General Adversarial Networks](https://arxiv.org/abs/1406.2661)

[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

[Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)

### Miscellaneous

[Recent Conversation between Ian Goodfellow with Lex Fridman](https://www.youtube.com/watch?v=Z6rxFNMGdn0)

[Explanation of how simple bicubic interpolation is](https://www.youtube.com/watch?v=poY_nGzEEWM)

[Frétchet Inception Distance](https://nealjean.com/ml/frechet-inception-distance/)

[Why is GAN hard to train?](https://medium.com/@jonathan_hui/gan-why-it-is-so-hard-to-train-generative-advisory-networks-819a86b3750b)

[Blog on SRGAN](https://towardsdatascience.com/srgan-a-tensorflow-implementation-49b959267c60)

[Another blog on SRGAN](https://medium.com/@jonathan_hui/gan-super-resolution-gan-srgan-b471da7270ec)

[Another another blog on SRGAN](https://github.com/deepak112/Keras-SRGAN)

[University of Toronto Slide on GAN](https://www.cs.toronto.edu/~duvenaud/courses/csc2541/slides/gan-foundations.pdf)
