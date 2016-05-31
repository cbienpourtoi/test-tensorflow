# encoding: UTF-8
# Copyright 2016 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import mnist_data
import tensorflowvisu
tf.set_random_seed(0)
import math
# neural network with 1 layer of 10 softmax neurons
#
# · · · · · · · · · ·       (input data, flattened pixels)       X [batch, 784]        # 784 = 28 * 28
# \x/x\x/x\x/x\x/x\x/    -- fully connected layer (softmax)      W [784, 10]     b[10]
#   · · · · · · · ·                                              Y [batch, 10]

# The model is:
#
# Y = softmax( X * W + b)
#              X: matrix for 100 grayscale images of 28x28 pixels, flattened (there are 100 images in a mini-batch)
#              W: weight matrix with 784 lines and 10 columns
#              b: bias vector with 10 dimensions
#              +: add with broadcasting: adds the vector to each line of the matrix (numpy)
#              softmax(matrix) applies softmax on each line
#              softmax(line) applies an exp to each value then divides by the norm of the resulting line
#              Y: output matrix with 100 lines and 10 columns

# Download images and labels
mnist = mnist_data.read_data_sets("data")

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])


# dropouts :
pkeep = tf.placeholder(tf.float32)


# weights W[784, 10]   784=28*28
W1 = tf.Variable(tf.truncated_normal([784, 200], stddev=0.1))
# biases b[10]
b1 = tf.Variable(tf.zeros([200]))


# weights W[784, 10]   784=28*28
W2 = tf.Variable(tf.truncated_normal([200, 100], stddev=0.1))
# biases b[10]
b2 = tf.Variable(tf.zeros([100]))


# weights W[784, 10]   784=28*28
W3 = tf.Variable(tf.truncated_normal([100, 60], stddev=0.1))
# biases b[10]
b3 = tf.Variable(tf.zeros([60]))


# weights W[784, 10]   784=28*28
W4 = tf.Variable(tf.truncated_normal([60, 30], stddev=0.1))
# biases b[10]
b4 = tf.Variable(tf.zeros([30]))

# weights W[784, 10]   784=28*28
W5 = tf.Variable(tf.truncated_normal([30, 10], stddev=0.1))
# biases b[10]
b5 = tf.Variable(tf.zeros([10]))



# flatten the images into a single line of pixels
# -1 in the shape definition means "the only possible dimension that will preserve the number of elements"
XX = tf.reshape(X, [-1, 784])

# The model
Y1 = tf.nn.relu(tf.matmul(XX, W1) + b1)
Y1b = tf.nn.dropout(Y1, pkeep)

# The model
Y2 = tf.nn.relu(tf.matmul(Y1b, W2) + b2)
Y2b = tf.nn.dropout(Y2, pkeep)

# The model
Y3 = tf.nn.relu(tf.matmul(Y2b, W3) + b3)
Y3b = tf.nn.dropout(Y3, pkeep)

# The model
Y4 = tf.nn.relu(tf.matmul(Y3b, W4) + b4)
Y4b = tf.nn.dropout(Y4, pkeep)

# The model
Ylogits = tf.matmul(Y4b, W5) + b5
Y5 = tf.nn.softmax(Ylogits)

# loss function: cross-entropy = - sum( Y_i * log(Yi) )
#                           Y: the computed output vector
#                           Y_: the desired output vector

# cross-entropy
# log takes the log of each element, * multiplies the tensors element by element
# reduce_mean will add all the components in the tensor
# so here we end up with the total cross-entropy for all images in the batch
#cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y5)) * 1000.0  # normalized for batches of 100 images,
                                                          # *10 because  "mean" included an unwanted division by 10

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(Ylogits, Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100


# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y5, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# variable learning rate
lr = tf.placeholder(tf.float32)

# training, learning rate = 0.005
train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)

# matplotlib visualisation
allweights = tf.reshape(W5, [-1])
allbiases = tf.reshape(b5, [-1])
I = tensorflowvisu.tf_format_mnist_images(X, Y5, Y_)  # assembles 10x10 images by default
It = tensorflowvisu.tf_format_mnist_images(X, Y5, Y_, 1000, lines=25)  # 1000 images on 25 lines
datavis = tensorflowvisu.MnistDataVis()

# init
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data):


    # learning rate decay
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000 # 0.003-0.0001-2000=>0.9826 done in 5000 iterations
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)


    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)

    # compute training values for visualisation
    if update_train_data:
        a, c, im, w, b = sess.run([accuracy, cross_entropy, I, allweights, allbiases], feed_dict={X: batch_X, Y_: batch_Y, pkeep: 1.0})
        datavis.append_training_curves_data(i, a, c)
        datavis.append_data_histograms(i, w, b)
        datavis.update_image1(im)
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c))

    # compute test values for visualisation
    if update_test_data:
        a, c, im = sess.run([accuracy, cross_entropy, It], feed_dict={X: mnist.test.images, Y_: mnist.test.labels, pkeep: 1.0})
        datavis.append_test_curves_data(i, a, c)
        datavis.update_image2(im)
        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))

    # the backpropagation training step
    sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y, lr:learning_rate, pkeep: 0.75})


datavis.animate(training_step, iterations=8000+1, train_data_update_freq=40, test_data_update_freq=50, more_tests_at_start=True)

# to save the animation as a movie, add save_movie=True as an argument to datavis.animate
# to disable the visualisation use the following line instead of the datavis.animate line
# for i in range(2000+1): training_step(i, i % 50 == 0, i % 10 == 0)

print("max test accuracy: " + str(datavis.get_max_test_accuracy()))

# final max test accuracy = 0.9268 (10K iterations). Accuracy should peak above 0.92 in the first 2000 iterations.
