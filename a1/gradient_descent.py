#!/usr/bin/env python
import tensorflow as tf


# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer

learning_rates = [0.01, 0.001, 0.0001, 0.1, 0.5]
learning_rate_i = 0
for learning_rate in learning_rates:

    optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    # training data
    x_train = [1, 2, 3, 4]
    y_train = [0, -1, -2, -3]
    # training loop
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init) # reset values to wrong

    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

    losses = []
    for i in range(1000):
      sess.run(train, {x: x_train, y: y_train})
      curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
      print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
      losses.append(curr_loss)

    with open('losses' + str(learning_rate_i) + '.txt', 'w') as f:
        for item in losses:
            f.write("%s\n" % item)

    learning_rate_i = learning_rate_i + 1