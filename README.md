# SRGAN-Keras-Implementation
Implementing SRGAN - an Generative Adversarial Network model to produce high resolution photos.
In this repository we have reproduced the SRGAN Paper - Which can be used on low resolution images to make them high resolution images. 
The link to the paper can be found here: [SRGAN](https://arxiv.org/pdf/1609.04802.pdf)

### Model Architechture
The model is assembled from two components Discriminator and Generator.
Discriminator - Responsible to distinguish between generated photos and real photos.
Generator - Generate high resolution images from low resolution images.

![GAN architecture](https://lilianweng.github.io/lil-log/assets/images/GAN.png) 


#### Generator
components list:
* 7 Convolution blocks Each block with the same number of filters
* PReLU with ( &alpha; = 0.2 ) is used as activation layer
* 2 PixelShuffler layers for upsampling - PixelShuffler is feature map upscaling
* Skip connections are used to achieve faster convergence 

#### Discriminator
components list:
* 16 Residual blocks Each block with increasing number of filters
* LeakyReLU with ( &alpha; = 0.2 ) is used as activation layer
* 2 Dense layers

![SRGAN](https://github.com/tensorlayer/srgan/raw/master/img/model.jpeg)


## Data download link
`!wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip`

## How To Use
You can run this in two ways:
* Using terminal
* Running the Notebook

If you decided the first choice follow the next steps:
0. you first need to download the data from this [link](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip)
1. run this line from the terminal: `python3 init.py --mode train --dir-path <path to your images folder>`
2. use `--help` to see all the available commands: `python3 init.py --help`

#### Further work
* Create loader which doesn't hold the images in memory.
* Add a link to pre-trained model.

## Results
#### After 350 Epochs

![result](https://github.com/AvivSham/SRGAN-Keras-Implementation/blob/master/image.png)
