'''
Created on Sep 3, 2017

@author: Amin
'''
import tensorflow as tf
from scipy.io import loadmat
import numpy as np
import sys


x = loadmat('../mnist_data/ex4data1.mat')
w = loadmat('../mnist_data/ex4weights.mat')
X = x['X']

# one hot conversion
y_temp = x['y']
y_temp = np.reshape(y_temp, (len(y_temp),))
y = np.zeros((len(y_temp),10))
y[np.arange(len(y_temp)), y_temp-1] = 1.

'''
Theta1 = w['Theta1']
Theta1 = Theta1[:,-1]
print (np.shape(Theta1))
print (type(Theta1))
sys.exit()
'''


'''
y = tf.one_hot(y, depth= 10)
with tf.Session() as sess:
    sess.run(y)
    print (np.shape(y))
'''

input_size = 400
hidden1_size = 25
output_size = 10
num_iters = 5000
learning_rate = 0.2
reg_param_lambda = 1

x = tf.placeholder(tf.float32, [None, input_size], name='data')

'''
# this set of params require 5000 iteration to obtain 87%
W1 = tf.Variable(tf.truncated_normal([hidden1_size, input_size],mean=0, stddev=1.0/np.sqrt(hidden1_size)), dtype=tf.float32, name='weights_1st_layer')
b1 = tf.Variable(tf.truncated_normal([hidden1_size],mean=0, stddev=1.0/np.sqrt(hidden1_size)),dtype=tf.float32, name='bias_layer_1')
W2 = tf.Variable(tf.truncated_normal([output_size, hidden1_size],mean=0, stddev=1.0/np.sqrt(output_size)), dtype=tf.float32, name='weights_2nd_layer')
b2 = tf.Variable(tf.truncated_normal([output_size], mean=0, stddev=1.0/np.sqrt(output_size)), dtype=tf.float32, name='bias_layer_2')
'''

# working with loaded parameters already trained weights
W1 = tf.Variable(tf.cast(w['Theta1'][:,1:],tf.float32))
b1 = tf.Variable(tf.cast(w['Theta1'][:,0], tf.float32))
W2 = tf.Variable(tf.cast(w['Theta2'][:,1:],tf.float32))
b2 = tf.Variable(tf.cast(w['Theta2'][:,0],tf.float32))

hidden_op = tf.sigmoid((tf.add(tf.matmul(x, W1, transpose_b=True), b1)))
output_op = (tf.matmul(hidden_op, W2, transpose_b=True) + b2)
 

y_ = tf.placeholder(tf.float32, [None, 10], name='actual_labels')


cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=y_, logits=output_op))
cross_entropy =cross_entropy + (reg_param_lambda/10000.0)*(tf.nn.l2_loss(W1)+tf.nn.l2_loss(W2))                                         

# custom cost problematic gives nan
'''
cross_entropy = tf.multiply(tf.log(output_op), y_) + tf.multiply(tf.subtract(tf.ones((5000,10), tf.float32), output_op), tf.subtract(tf.ones((5000,10),tf.float32), y_)) 
cross_entropy = tf.reduce_sum(tf.reduce_sum(cross_entropy, axis=1))
cross_entropy = -(cross_entropy/len(X))
'''

'''
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print (sess.run([W1[20], W2[8]]))
    print (sess.run(y_,feed_dict={y_:y}))
    print (sess.run(tf.subtract(tf.ones(tf.shape(y_)), y_),feed_dict={y_:y}))
    print (sess.run(output_op,feed_dict={x : X, y_ : y}))
'''


train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(num_iters):
    print ('training..', _)
    #print ('first cost', sess.run())
    #print ('first regg',sess.run((reg_param_lambda/5000.0)*(tf.nn.l2_loss(W1)+tf.nn.l2_loss(W2))))
    print (sess.run([train_step, cross_entropy], feed_dict={x : X, y_ : y}))

#pred = tf.nn.softmax(output_op)
corr_pred = tf.equal(tf.argmax(output_op, axis=1), tf.argmax(y_, axis=1))
acc = tf.reduce_mean(tf.cast(corr_pred, tf.float32))
print (sess.run(acc, feed_dict={x:X, y_:y}))
#print (sess.run(tf.argmax(tf.nn.softmax(output_op), axis=1), feed_dict={x:X}))
sess.close()










