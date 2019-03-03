from models.SRGAN import SRGAN
from utils import *
def train(FLAGS):
  
  dir_path = FLAGS.dir_path
  upscale_factor = FLAGS.upscale_factor
  lr_height = FLAGS.lr_height
  lr_width = FLAGS.lr_width
  channels = FLAGS.number_channels
  generator_lr = FLAGS.generator_lr
  discriminator_lr = FLAGS.discriminator_lr
  gan_lr = FLAGS.gan_lr
  batch_size = FLAGS.batch_size
  save_interval = FLAGS.save_interval
  epochs = FLAGS.epochs
  
  high_reso, low_reso = prepare_data(dir_path)
  srgan_model = SRGAN(high_reso,low_reso, lr_height = lr_height, lr_width = lr_width, channels = channels,
                      upscale_factor = upscale_factor, generator_lr = generator_lr,
                      discriminator_lr = discriminator_lr, gan_lr = gan_lr)
  srgan_model.train(epochs, save_interval = save_interval, batch_size = batch_size)
