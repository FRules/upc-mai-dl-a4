#!/usr/bin/env python
import tensorflow as tf
import read_inputs
import numpy as N


optimizers = [tf.train.GradientDescentOptimizer(0.01), tf.train.GradientDescentOptimizer(0.001), tf.train.GradientDescentOptimizer(0.0001),
              tf.train.GradientDescentOptimizer(0.1), tf.train.GradientDescentOptimizer(0.5),
              tf.train.AdagradOptimizer(0.01), tf.train.AdagradOptimizer(0.001), tf.train.AdagradOptimizer(0.0001),
              tf.train.AdagradOptimizer(0.1), tf.train.AdagradOptimizer(0.5),
              tf.train.AdadeltaOptimizer(0.01), tf.train.AdadeltaOptimizer(0.001), tf.train.AdadeltaOptimizer(0.0001),
              tf.train.AdadeltaOptimizer(0.1), tf.train.AdadeltaOptimizer(0.5),
              tf.train.AdamOptimizer(0.01), tf.train.AdamOptimizer(0.001), tf.train.AdamOptimizer(0.0001),
              tf.train.AdamOptimizer(0.1), tf.train.AdamOptimizer(0.5)]

losses = {"experiment_1": [], "experiment_2": [], "experiment_3": [], "experiment_4": [], "experiment_5": [],
          "experiment_6": [], "experiment_7": [], "experiment_8": [], "experiment_9": [], "experiment_10": [],
          "experiment_11": [], "experiment_12": [], "experiment_13": [], "experiment_14": [], "experiment_15": [],
          "experiment_16": [], "experiment_17": [], "experiment_18": [], "experiment_19": [], "experiment_20": [] }

accuracies = {"experiment_1": 0, "experiment_2": 0, "experiment_3": 0, "experiment_4": 0, "experiment_5": 0,
              "experiment_6": 0, "experiment_7": 0, "experiment_8": 0, "experiment_9": 0, "experiment_10": 0,
              "experiment_11": 0, "experiment_12": 0, "experiment_13": 0, "experiment_14": 0, "experiment_15": 0,
              "experiment_16": 0, "experiment_17": 0, "experiment_18": 0, "experiment_19": 0, "experiment_20": 0 }

experiment_index = 1
for optimizer in optimizers:
    #read data from file
    data_input = read_inputs.load_data_mnist('MNIST_data/mnist.pkl.gz')
    #FYI data = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    data = data_input[0]
    #print ( N.shape(data[0][0])[0] )
    #print ( N.shape(data[0][1])[0] )

    #data layout changes since output should an array of 10 with probabilities
    real_output = N.zeros( (N.shape(data[0][1])[0] , 10), dtype=N.float )
    for i in range ( N.shape(data[0][1])[0] ):
      real_output[i][data[0][1][i]] = 1.0

    #data layout changes since output should an array of 10 with probabilities
    real_check = N.zeros( (N.shape(data[2][1])[0] , 10), dtype=N.float )
    for i in range ( N.shape(data[2][1])[0] ):
      real_check[i][data[2][1][i]] = 1.0



    #set up the computation. Definition of the variables.
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    train_step = optimizer.minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    #TRAINING PHASE
    print("TRAINING")

    for i in range(500):
      batch_xs = data[0][0][100*i:100*i+100]
      batch_ys = real_output[100*i:100*i+100]
      _, cross_entropy_value = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
      losses["experiment_" + str(experiment_index)].append(cross_entropy_value)

    #CHECKING THE ERROR
    print("ERROR CHECK")
    print("LOSS:", cross_entropy_value)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracies["experiment_" + str(experiment_index)] = sess.run(accuracy, feed_dict={x: data[2][0], y_: real_check})
    print(sess.run(accuracy, feed_dict={x: data[2][0], y_: real_check}))

    with open('losses' + str(experiment_index) + '.txt', 'w') as f:
        for item in losses["experiment_" + str(experiment_index)]:
            f.write("%s\n" % item)

    experiment_index = experiment_index + 1

print(accuracies)
