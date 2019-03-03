from models.SRGAN import SRGAN
from utils import *
def test(FLAGS):
  
  dir_path = FLAGS.dir_path
  load_weights_path = FLAGS.load_weights_path
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
  n_images = FLAGS.n_images
  
  high_reso, low_reso = prepare_data(dir_path)
  srgan_model = SRGAN(lr_height = lr_height, lr_width = lr_width, channels = channels,
                      upscale_factor = upscale_factor, generator_lr = generator_lr,
                      discriminator_lr = discriminator_lr, gan_lr = gan_lr)
  model_srgan.load_weights(load_weights_path)
  plot_predict(low_reso,high_reso,srgan_model,n_images)
