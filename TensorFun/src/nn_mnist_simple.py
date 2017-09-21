'''
Created on Aug 31, 2017

@author: Amin
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import sys

from tensorflow.examples.tutorials.mnist import input_data



if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    print ('here in nn_mnist_simple')
    
#     print (np.shape(mnist.test.images))
#     print (np.shape(mnist.test.labels))
#     sys.exit()
    '''
    print (type(mnist))
    print(np.shape(mnist))
    print (np.shape(mnist.train.images))
    print (np.shape(mnist.test.images))
    X = mnist.train.images[0]
    Y = mnist.train.labels[0]
    print (np.shape(X))
    X = X.reshape((28,28))
    print (np.shape(X))
    print (Y)
    plt.imshow(X,cmap='gray')
    plt.show()
    '''
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    #y = tf.nn.softmax(logits=tf.matmul(x, W)+b)         #this will have hypothesis for n data points
    y = tf.matmul(x, W)+b
    y_ = tf.placeholder(tf.float32, [None, 10])         #place holder for true distribution
    #pred_error = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices = [1]))   # red ind 1 gives N sum
    pred_error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(pred_error)
    
    
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    for _ in range(1000):
        print ('training...step', _)
        #batch_xs, batch_ys = mnist.train.next_batch(100)
        print (sess.run([train_step,pred_error], feed_dict={x: mnist.train.images, y_: mnist.train.labels}))

    correct_pred = tf.equal((tf.argmax(y, axis=1)),tf.argmax(y_,axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    print (sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
    print (sess.run(W[400]))
    sess.close()
    print (y.shape)
    
    
    
    
    
    