#!/usr/bin/env python
import tensorflow as tf
import read_inputs
import random
import numpy as np

BATCH_SIZE = 50
NUMBER_CLASSES = 10
IMAGE_WIDTH, IMAGE_HEIGHT = 28, 28
COLOR_CHANNELS = 1
NUM_INPUT = IMAGE_WIDTH * IMAGE_HEIGHT

# read data from file
data_input = read_inputs.load_data_mnist('MNIST_data/mnist.pkl.gz')
data = data_input[0]

# data layout changes since output should be an array of 10 with probabilities
real_output = np.zeros((np.shape(data[0][1])[0], NUMBER_CLASSES), dtype=np.float)
for i in range(np.shape(data[0][1])[0]):
    real_output[i][data[0][1][i]] = 1.0

# data layout changes since output should be an array of 10 with probabilities
real_check = np.zeros((np.shape(data[2][1])[0], NUMBER_CLASSES), dtype=np.float)
for i in range(np.shape(data[2][1])[0]):
    real_check[i][data[2][1][i]] = 1.0

# set up the computation. Definition of the variables.
x = tf.placeholder(tf.float32, [None, NUM_INPUT])
W = tf.Variable(tf.zeros([NUM_INPUT, NUMBER_CLASSES]))
y_ = tf.placeholder(tf.float32, [None, NUMBER_CLASSES])

keep_prob = tf.placeholder(tf.float32)


# declare weights and biases
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# convolution and pooling
def conv2d(x, W, b):
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape x to a 4d tensor, with the second and third dimensions corresponding to image width and height.
    # 28x28 = 784
    # The final dimension corresponding to the number of color channels.
    x = tf.reshape(x, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS])

    # Convolution Layer
    conv1 = conv2d(x, weights['conv1'], biases['conv1'])
    conv1 = max_pool_2x2(conv1)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['conv2'], biases['conv2'])
    conv2 = max_pool_2x2(conv2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['dense'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['dense']), biases['dense'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


layer1_filters = 32
layer2_filters = 64

weights = {
  "conv1": weight_variable([5, 5, 1, layer1_filters]),
  "conv2": weight_variable([5, 5, layer1_filters, layer2_filters]),
  "dense": weight_variable([7 * 7 * layer2_filters, 1024]),
  "out": weight_variable([1024, NUMBER_CLASSES])
}

biases = {
  "conv1": bias_variable([layer1_filters]),
  "conv2": bias_variable([layer2_filters]),
  "dense": bias_variable([1024]),
  "out": bias_variable([NUMBER_CLASSES])
}


logits = conv_net(x, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # TRAIN
    print("TRAINING")

    shuffled = list(range(1000))
    random.shuffle(shuffled)
    for i in shuffled:

        # until 1000 96,35%
        batch_ini = BATCH_SIZE * i
        batch_end = BATCH_SIZE * i + BATCH_SIZE

        batch_xs = data[0][0][batch_ini:batch_end]
        batch_ys = real_output[batch_ini:batch_end]

        if i % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch_xs, y_: batch_ys, keep_prob: 1.0})
            print('step %d, training accuracy %g Batch [%d,%d]' % (i, train_accuracy, batch_ini, batch_end))

        train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

    # TEST
    print("TESTING")

    train_accuracy = accuracy.eval(feed_dict={x: data[2][0], y_: real_check, keep_prob: 1.0})
    print('test accuracy %g' % train_accuracy)
