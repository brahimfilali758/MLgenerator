import tensorflow as tf
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
generator = tf.keras.models.load_model(join("data","pokemon_generator.h5"))
for i in range(10):
	fixed_seed = np.random.normal(0, 1, (1, 100))
	generated_image = generator(fixed_seed)
	print(tf.reduce_min(generated_image))
	print(tf.reduce_max(generated_image))
	generated_image = (generated_image + 1.) * 127.5
	print(tf.reduce_min(generated_image))
	print(tf.reduce_max(generated_image))
	plt.figure()
	plt.imshow(generated_image[0,:,:,:].astype(np.uint8))
plt.show()
