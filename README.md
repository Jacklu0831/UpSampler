# Super Resolution GAN

There is quite a lot to write about this project, you can use this table of contents to navigate.

[table of contents that is soon to be made]

## Intro

Motivation:\
> Super resolution has a wide application from satellite imaging to photography. However, the truth is I wanted to use an image for my phone's wallpaper but it had low resolution, so the idea of some kind of fully-convolutional encoder-decoder network came to me and of course, I found out a number of papers have already been published around this topic. With previous experience in implementing a CycleGAN and DCGAN ([first GANs](https://github.com\Jacklu0831/GAN-Projects)), I comprehended some relevant [resources](#Sources) then started this. 

In summary, I built and trained a Photo-Realistic Single Image Super-Resolution Generative Adversial Network with Tensorflow + Keras. The aim is to map 44x44 colored images to 176x176 (factor of 4) while keeping the perceptual details and texture. The main source of knowledge came from [the original SRGAN paper](https://arxiv.org/abs/1609.04802) and [this analysis on the importance of dataset](https://arxiv.org/abs/1903.09922). Despite not having compatible hardware for the computational demanding models, I was able to achieve great [results](#Results) by choosing smaller images, tweaking the model configuration, and lowering the batch size. 

<p align="center"><image src=""></image></p>

---

## Background + the Math

Invented by Ian GoodFellow in [2014](https://arxiv.org/abs/1406.2661), GAN showed amazing image generative abilities from road scenes to faces. However, generating images out of random noise is only a fraction of its capability. From switching images between domains (CycleGAN) to music generation (MidiNet), the breadth of tasks GAN could take on is still being rapidly discovered. 

Image super resolution can be defined as increasing the size of small images while keeping the drop in quality to minimum, or restoring high resolution images from rich details obtained from low resolution images. Simply put, the SRGAN network learns a mapping from the low-resolution patch through a series of convolutional, fully-connected, and transposed/upsampling convolutional layers into the high-resolution patch while keeping texture/perceptual details. It has applications in the fields of surveillance, forensics, medical imaging, satellite imaging, and consumer photography. 

Below is my attempt to concisely explain SRGAN from [this paper](https://arxiv.org/abs/1609.04802). I collected some well-made images/diagrams from the paper and blogs for the visuals. 

### Neural Network Architecture

<p align="center"><image src="assets/architecture.png"></image></p>

The high level architecture of SRGAN is quite simple, it closely resembles the vanilla GAN network and is also about reaching the [Nash Equilibrium in the zero-sum game])https://www.cs.toronto.edu/~duvenaud/courses/csc2541/slides/gan-foundations.pdf) between the generator and the discriminator.

1. High resolution (HR) ground truth images are selected from the training set
2. Low resolution (LR) images corresponding to the HR images are created with bi-cubic downsampling 
3. The generator upsamples the LR images to Super Resolution (SR) images to fool the discriminator
4. The discriminator distinguishes the HR images (ground truth) and the SR images (output) to judge the generator
5. Train discriminator and generator with backpropagation

### Generator

<p align="center"><image src="assets/generator.png"></image></p>

The generator takes a LR image, process it with a conv and a PReLU (trainable LReLU) layer, puts it through 16 [residual blocks](https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec) borrowed from SRResNet, upsamples by factor of 2 twice, and puts it through one last conv layer to produce the SR image. Unlike normal convolutional layers, going crazy with the number of residual blocks is not prone to overfitting the dataset because of their ability of [identity mapping](https://arxiv.org/abs/1603.05027). 

### Discriminator

<p align="center"><image src="assets/discriminator.png"></image></p>

The discriminator is similar to a normal image classifier, except its task is now more daunting due to having to classify two images with near identical content (if the generator is trained well enough). It puts HR/SR images first through a conv and a LReLU layer, process the image through 7 conv-BN-LReLU blocks, flatten the image, then use two dense layers with a LReLU in middle and a sigmoid function at the end for binary classification. 

### Intuition

At first I thought despite identity mapping, 16 Residual layers (generator) and 7 discriminator blocks (discriminator) was an overkill for small images with size of 176x176. However, it does make sense because the generator needs to learn the features of an image well enough to the point of producing a more detailed version of it. On the other hand, the discriminator has to classify two images with increasingly identical content. On top of all these, quoting Ian GoodFellow ([source podcast](https://www.youtube.com/watch?v=Z6rxFNMGdn0)):

> The way of building a generative model for GANs is we have a two-player game in the game theoretic sense and as the players of this game compete, one of them becomes able to generate realistic data. 

Adversarial networks have to both be constantly struggling against each other in this zero-sum game of GAN. The process of converging to the Nash Equilibrium can be extremely slow, that is one more reason why the generator needed 16 residual layers to deeply learn about the details of images.

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

## Procedures and Challenges

This section contains an overview of what I did, the problems I faced, and the solutions for overcoming them.

### Stage 1 - Building

Being one of the newer applications of GAN when GAN is one of the newer neural architecture in the first place, resources on SRGAN was limited. Thankfully, the [original paper](https://arxiv.org/abs/1609.04802) was very informative and was a pleasant read. For the model architecture, I mainly constructed the model from the original paper and experimented with the number of residual blocks and the loss functions (used BCE with tweaks). 

Redirect to [Background Section](#Background-+-the-Math) for detailed explanation of the architecture components and how they come together with visuals. For details on the parameters I used, I made a pretty neat list of them in `parameters.txt`. I also am quite fond of TF's format for model summary, so I put them in `model_summary.txt` to keep the Jupyter/Colab notebooks short. 

### Stage 2 - Training

> Ian GoodFellow himself would have thought GAN was infeasible partly due to the dual-network training expense if he was not drunk ([podcast](https://www.youtube.com/watch?v=Z6rxFNMGdn0) at around 27 mins), but my experience shows that SRGAN is even worse because of its heavy [architecture](#Neural-Network-Architecture). 

The actual training process failed multiple times due to the lack of computing power, GPU storage, and disconnection. These issues were resolved by decrease batch size, image sizes, manually uploading files from my local device, write outputs/models directly to my Google Drive every number of epochs, and having a separate file for continue training with saved models (see files section for more details). Although this is not the first time using Google Colab, this is one of the most problematic project and I ended up learning a lot more about using Cloud Computing in general and obtained great project results. 

As for the time taken, even after decreasing the image size to free up storage and decreasing the dataset size to 2500 images (train + test), the provided T4 GPU had to run at 4+ min/epoch for 2500 epochs. The total training time was more than a week for each model and the project spanned a month. 

I carefully observed `face_loss.txt` and `coco_loss.txt` throughout the training process to make sure that both the generator and the discriminator to make sure that no one is dominating this zero-sum game. The generator's perceptual loss steadily dereased throughout the 2500 epochs for both models on COCO and CelebA. The good news is that it means the learning rate is not too big and the the model weights are indeed moving toward the Nash equilibrium. The bad news is that since the GPU is not very fast, it was difficult to know whether my hyperparameters were working, and each try means forfeitin up to days of training. [This blog](https://www.google.com/search?q=why+is+gan+hard+to+train&oq=why+is+gan+hard&aqs=chrome.0.69i59j69i60j69i57j0.1837j0j1&sourceid=chrome&ie=UTF-8) provides a nice explaination on why GAN is so hard to train compared to numerous other neural architectures.

### Stage 3 - Performance Analysis

I trained the first model on the COCO dataset and quickly noticed the issue of it performing atrociously with images with more details, which is because LR image not being able to capture the texture and perceptual details of its HR origin. Since human face is one of the most complex feature that can appear in a picture, I chose to train my second model completely on faces to observe how much I can push the performance on possibly the most complex features. Below is a side by Side comparison between the same model's performance on images with drastically different complexity.

[insert side by side comparison between details and non-details]

On the other hand, the model that is trained only face images were able to produce perceptually great faces by the 500th epoch. However, it struggled with the most detailed feature of human face, which are the eyes. Since the downsized images carry very less information for reconstructing the eyes of a person, it is mostly up to the generator for drawing on the eyes itself. Since eyes are actually very important for recognizing a face, I continuously trained the model and observed a gradually improvement in the generator's ability in reconstructing the eyes of people.

[insert face images]




> If I train model A with a variety of objects and model B with only one category/type of images (dataset with narrower domain), say cats. Would B perform better than A on cat images or is SRGAN only about recognizing small textures and edges as detailed as possible? 

I asked this question on Quora and received no response :( and only later found out about [this paper](https://arxiv.org/abs/1903.09922), so I decided to clarify my own question through experimentation. I trained two models with the same configuration separately on the COCO dataset and the CelebA dataset for the same number of epochs, this was done to investigate how training SRGAN on a narrow domain of images (faces) improve its performance on the domain of images that it was trained on. 

[insert stuff about FID]
[insert coco vs face]

---

## Results

Below are a few test results from COCO and CelebA datasets. More can be found in the `results` dir.

### COCO Results

<pre>          Low-Res Input      Super-Res Output        High-Res Ground Truth </pre>

<p align="center">
  <image src="assets/result_245.png" height="70%" width="70%"></image>
  <image src="assets/result_255.png" height="70%" width="70%"></image>
  <image src="assets/result_261.png" height="70%" width="70%"></image>
</p>

### CelebA Results

<pre>          Low-Res Input      Super-Res Output        High-Res Ground Truth </pre>

<p align="center">
  <image src="assets/result_245.png" height="70%" width="70%"></image>
  <image src="assets/result_255.png" height="70%" width="70%"></image>
  <image src="assets/result_261.png" height="70%" width="70%"></image>
</p>

---

## Files

#### Code

<pre>
- SRGAN_coco.ipynb            - Google Colab implementation (coco dataset)
- SRGAN_coco_continue.ipynb   - Google Colab implementation (coco dataset restore model and continue training)
- SRGAN_face.ipynb            - Google Colab implementation (face dataset)
- SRGAN_face_continue.ipynb   - Google Colab implementation (face dataset restore model and continue training)
- SRGAN_test.py               - script for testing the trained models
- utils.py                    - some of image preprocess functions
</pre>

#### Directories

<pre>
- assets                      - images for this README
- datasets                    - 2500 images from each of the COCO dataset and CelebA dataset
- final_models                - .h5 files of the coco and face generators, discriminators not included due to size (300+ MB)
- losses                      - files containing complete information on the training loss of each epoch
</pre>

#### Others

<pre>
- README.md                   - self
- loss.txt                    - losses components of each epoch
- parameters.txt              - a complete list of hyperparameters and other parameters I used
- output                      - bunch of images with the epoch number beside them
</pre>

---

## Try it Yourself

#### Dependencies

Python 3.7, Tensorflow 1.14.0, Keras 2.2.4, numpy 1.15.0, matplotlib, Pillow, tqdm, OpenCV (utils)

#### Train

Open `SRGAN_coco.ipynb` file or `SRGAN_face.ipynb`, upload `coco.zip` or `celeb.zip`, make sure path names are correct and `shift + enter` away. If you encounter any confusion, feel free to [contact me](jacklu0831@gmail.com) (email).

#### Try Your Own Images

Run the script `SRGAN_test.py`. Make sure input and output directories and generator (`coco_g_model2500.h5` or `face_g_model2500.h5`) paths are correctly specified. 

---

## Sources

#### Papers

- [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)
- [SRGAN: Training Dataset Matters](https://arxiv.org/abs/1903.09922)
- [General Adversarial Networks](https://arxiv.org/abs/1406.2661)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)

#### Miscellaneous

- [Recent Conversation between Ian Goodfellow with Lex Fridman](https://www.youtube.com/watch?v=Z6rxFNMGdn0)
- [Why is GAN hard to train?](https://www.google.com/search?q=why+is+gan+hard+to+train&oq=why+is+gan+hard&aqs=chrome.0.69i59j69i60j69i57j0.1837j0j1&sourceid=chrome&ie=UTF-8)
- [UofT Slide on GAN]((https://www.cs.toronto.edu/~duvenaud/courses/csc2541/slides/gan-foundations.pdf))
