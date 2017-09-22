'''
Created on Aug 31, 2017

@author: Amin
'''

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


if __name__ == '__main__':
    print ('testing')
    '''
    x = tf.placeholder(tf.float16, (3,2))
    x1 = tf.placeholder(tf.float16, (3,3))
    y1 = tf.reduce_mean(x) 
    y2 = tf.reduce_mean(x, 0) 
    y3 = tf.reduce_mean(x, 1)
    data_arr = [[1,2],[3,4],[5,6]]
    data_arr2 = [[1,2],[3,4],[5,6],[7,8]]
    with tf.Session() as sess:
        print (sess.run(y1, feed_dict = {x : data_arr}))
        print (sess.run(y2, feed_dict = {x : data_arr}))
        print (sess.run(y3, feed_dict = {x : data_arr})) 
        # dimenstion issue
        print (sess.run(x*x1, feed_dict = {x: data_arr, x1 : data_arr2}))
    '''
    '''
    tens  = [1,2,3,4,5]
    otens = tf.one_hot(tens,depth=15)
    with tf.Session() as sess: 
        print (sess.run(otens))
    '''
    
    #tens  = [1,2,3,4,4]
    
    #one_hot = np.zeros((len(tens),10))
    #one_hot[np.arange(len(tens)),tens] = 1.0
    #one_hot[tens] = 1.0
    
    #print (one_hot)
    
    '''
    logits = tf.random_uniform([10,5], minval=0.0, maxval=1.0)
    labels = tf.one_hot([i%5 for i in range(10)], depth=5)
    with tf.Session() as sess:
        print (sess.run([logits, labels]))
        print (sess.run(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits))))
        loss = (tf.multiply(tf.log(logits),labels) + tf.multiply(tf.log(tf.subtract(tf.ones([10,5]),logits)),tf.subtract(tf.ones([10,5]),labels))) 
        loss = tf.reduce_sum(tf.reduce_sum(loss, axis=1))
        loss = -loss/10
        print (sess.run(loss))
    '''    
    '''
    logits = tf.random_uniform([4,3,1], minval=0.0, maxval=1.0)
    
    with tf.Session() as sess:
        print (sess.run(logits))
        print (sess.run(tf.squeeze(logits)))
        
    '''
    
    '''
    logits = tf.random_uniform([2,5,4], minval=0, maxval=10, dtype=tf.float32)
    logits2 = tf.random_uniform([5,2,4], minval=0, maxval=10, dtype=tf.float32)
    with tf.Session() as sess:
        vals = sess.run(logits)
        print (type(vals))
        print (vals)
        print (sess.run(tf.reshape(vals,[-1, 4])))
        
        print (sess.run(tf.nn.softmax_cross_entropy_with_logits(labels=logits, logits=logits2)))
    '''
    
    #print (list(zip([1,2,4,3],['a','b','c','d','e','f'])))
    
    
    #split squeeze test
    x = tf.random_uniform([4,5,2], minval=0, maxval=10, dtype=tf.float32)
    #y = [tf.squeeze(i) for i in tf.split(x, num_or_size_splits=5, axis=1)]
    y = tf.random_uniform([12,2], minval=0, maxval=10, dtype=tf.float32)
    
        
    with tf.Session() as sess:
        #print (sess.run(tf.split(x,num_or_size_splits=5,axis=1)))
        #print (sess.run(y))
        #print (sess.run(tf.unstack(x, axis=1)))
        #print (sess.run([tf.shape(tf.unstack(x, axis=1)), tf.shape(x)]))
        vals = sess.run(y)
        print (vals)
        print (sess.run(tf.reshape(vals, [3,4,2])))
    
    
    