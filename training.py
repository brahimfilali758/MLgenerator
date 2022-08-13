from models import build_descriminator, build_generator
from data_prep import load_dataa
import tensorflow as tf

BATCH_SIZE = 16
data = data_prep.load_dataa()
training_data = tf.data.Dataset.from_tensor_slices(data).batch(BATCH_SIZE)

print(training_data.shape)





