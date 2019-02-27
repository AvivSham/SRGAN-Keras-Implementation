# SRGAN-Keras_Implementation
Implementing SRGAN - an Generative Adversarial Network model to produce high resolution photos.
In this repository we have reproduced the SRGAN Paper - Which can be used on low resolution images to make them high resolution images. 
The link to the paper can be found here: [SRGAN](https://arxiv.org/pdf/1609.04802.pdf)

### Model Architechture
The model is assembled from two components Discriminator and Generator.
Discriminator - Responsible to distinguish between generated photos and real photos.
Generator - Generate high resolution images from low resolution images.
![GAN architecture](https://lilianweng.github.io/lil-log/assets/images/GAN.png) 
**CHANGE THE PHOTO**

#### Discriminator
components list:
* 7 Convolution blocks Each block with the same number of filters
* PReLU with ( &alpha; = 0.2 ) is used as activation layer
* 2 PixelShuffler layers for upsampling - PixelShuffler is feature map upscaling
* Skip connections are used to achieve faster convergence 

#### Generator
components list:
* 16 Residual blocks Each block with increasing number of filters
* LeakyReLU with ( &alpha; = 0.2 ) is used as activation layer
* 2 Dense layers

![SRGAN](https://github.com/tensorlayer/srgan/raw/master/img/model.jpeg)

## Data download link
`!wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip`

## How To Use
still to come

## Some Results
still to come
