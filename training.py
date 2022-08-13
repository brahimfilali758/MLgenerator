from models import build_descriminator, build_generator
from data_prep import load_data
import tensorflow as tf
from models import build_descriminator, build_generator
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import numpy as np
from os.path import join


BATCH_SIZE = 16
CHANNELS = 3
SEED_SIZE = 100
resolu = 3
WIDTH = 32*resolu
LENGTH = 32*resolu
EPOCHS = 10000


DATA_PATH = "data"
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

data = load_data()
training_data = tf.data.Dataset.from_tensor_slices(data).batch(BATCH_SIZE)


generator = build_generator(SEED_SIZE,CHANNELS)
print(generator.summary())

noise = tf.random.normal([1, SEED_SIZE])
generated_image = generator(noise, training=False)
print(generated_image.shape)
# plt.imshow(generated_image[0,:,:,0])
# plt.show()	

image_shape = (WIDTH,LENGTH,CHANNELS)
discriminator = build_descriminator(image_shape)
decision = discriminator(generated_image)
print(decision)



cross_entropy = tf.keras.losses.BinaryCrossentropy()

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1.5e-4,0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1.5e-4,0.5)

@tf.function
def train_step(images):
  seed = tf.random.normal([BATCH_SIZE, SEED_SIZE])

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(seed, training=True)

    real_output = discriminator(images, training=True)
    fake_output = discriminator(generated_images, training=True)

    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)
    

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(
        gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(
        gradients_of_discriminator, 
        discriminator.trainable_variables))
  return gen_loss,disc_loss

def train(dataset, epochs):
  fixed_seed = np.random.normal(0, 1, (1, 
                                       SEED_SIZE))
  start = time.time()

  for epoch in range(epochs):
    epoch_start = time.time()

    gen_loss_list = []
    disc_loss_list = []

    for image_batch in dataset:
      t = train_step(image_batch)
      gen_loss_list.append(t[0])
      disc_loss_list.append(t[1])

    g_loss = sum(gen_loss_list) / len(gen_loss_list)
    d_loss = sum(disc_loss_list) / len(disc_loss_list)

    epoch_elapsed = time.time()-epoch_start
    print (f'Epoch {epoch+1}, gen loss={g_loss},disc loss={d_loss},'\
           f' {hms_string(epoch_elapsed)}')
    # save_images(epoch,fixed_seed)
    if (epoch%20) == 0 :
	  generator.save(join(DATA_PATH,"pokemon_generator.h5"))
  elapsed = time.time()-start
  print (f'Training time: {hms_string(elapsed)}')



train(training_data, EPOCHS)

generator.save(join(DATA_PATH,"pokemon_generator.h5"))







