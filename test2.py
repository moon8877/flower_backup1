import numpy as np
np.random.seed(1)
import random
random.seed(2)
import tensorflow as tf
tf.compat.v1.set_random_seed(3)
tf.random.set_seed(4)
# tf.compat.v1.disable_eager_execution()

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import GlorotUniform


ipt = Input((4,))
out = Dense(4, kernel_initializer=GlorotUniform(seed=0))(ipt)
model = Model(ipt, out)
model.compile('adam', 'mse')

x = y = np.random.randn(32, 4)
model.train_on_batch(x, y)
print(model.get_weights())

grads = get_gradients(model, x, y, model.trainable_weights)
print(grads)
