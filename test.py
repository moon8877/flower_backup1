import keras
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
# Dummy input
test = np.random.rand(1, 32, 32, 1)

x = tf.placeholder(tf.float32, shape=(None, 32, 32, 1))

model = keras.layers.Conv2D(16, 5, padding = 'same', activation='elu') (test)
model = keras.layers.Flatten() (model)
model = keras.layers.Dense(128, activation='relu') (model)
predictions = keras.layers.Dense(1) (model)

opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
gradient_step = opt.compute_gradients(predictions, tf.trainable_variables())
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    gradients = sess.run(gradient_step)
    print(gradients)
    print(np.array(gradients).shape)
