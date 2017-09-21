'''
Created on Sep 11, 2017

@author: Amin
'''


import tensorflow as tf
from scipy.io import loadmat
import numpy as np
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class RNN_config(object):
    num_steps = 5
    batch_size = 200
    num_classes = 2
    state_size = 4
    learning_rate = 0.1

    def __init__(self, num_steps=5, state_size=4):
        self.num_steps = num_steps
        self.state_size = state_size

def setup_graph(graph, config):
    with graph.as_default():

        """
        Placeholders
        """

        x = tf.placeholder(tf.int32, [config.batch_size, config.num_steps], name='input_placeholder')
        y = tf.placeholder(tf.int32, [config.batch_size, config.num_steps], name='labels_placeholder')
#         x = tf.placeholder(tf.int32, [config.batch_size, None], name='input_placeholder')
#         y = tf.placeholder(tf.int32, [config.batch_size, None], name='labels_placeholder')
        default_init_state = tf.zeros([config.batch_size, config.state_size])
        init_state = tf.placeholder_with_default(default_init_state, [config.batch_size, config.state_size], name='state_placeholder')

        """
        rnn_inputs and y_as_list
        """

        

        # Turn our y placeholder into a list of one-hot tensors
        y_one_hot = tf.one_hot(y, config.num_classes)
        #y_as_list = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(1, config.num_steps, y_one_hot)]
        #y_as_list = tf.unstack(y_one_hot, axis=1)                # needed for static rnn op
        y_as_list = tf.reshape(y_one_hot, [-1, config.num_classes]) # for fitting to dynamic rnn op

        """
        Definition of rnn_cell
        This is very similar to the __call__ method on Tensorflow's BasicRNNCell. See:
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell.py
        """
        '''
        with tf.variable_scope('rnn_cell'):
            W = tf.get_variable('W', [config.num_classes + config.state_size, config.state_size])
            b = tf.get_variable('b', [config.state_size], initializer=tf.constant_initializer(0.0))

        def rnn_cell(rnn_input, state):
            with tf.variable_scope('rnn_cell', reuse=True):
                W = tf.get_variable('W', [config.num_classes + config.state_size, config.state_size])
                b = tf.get_variable('b', [config.state_size], initializer=tf.constant_initializer(0.0))
            return tf.tanh(tf.matmul(tf.concat([rnn_input, state], 1), W) + b)
        '''
        """
        Adding rnn_cells to graph
        This is a simplified version of the "rnn" function from Tensorflow's api. See:
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn.py
        """
        '''
        state = init_state
        rnn_outputs = []
        for rnn_input in rnn_inputs:
            state = rnn_cell(rnn_input, state)
            rnn_outputs.append(state)
        final_state = rnn_outputs[-1]
        '''
        
        cell = tf.contrib.rnn.BasicRNNCell(config.state_size)
        
        # Turn our x placeholder into a list of one-hot tensors
        # 1. tf.split creates a list of config.num_steps tensors, each with shape [batch_size X 1 X 2]
        # 2. tf.squeeze gets rid of the middle dimension from each
        # 3. Thus, rnn_inputs is a list of config.num_steps tensors with shape [batch_size, 2]
        '''
        # for static rnn
        x_one_hot = tf.one_hot(x, config.num_classes)
        rnn_inputs = tf.unstack(x_one_hot, axis =1)     # seperating inputs for different cell 
        #rnn_inputs = [tf.squeeze(i,squeeze_dims=[1]) for i in tf.split(1, config.num_steps, x_one_hot)]
        rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, initial_state=init_state)
        '''
        
        
        # for dynamic rnn
        x_one_hot = tf.one_hot(x, config.num_classes)
        rnn_inputs = x_one_hot          # a tensor of size [batch_size, num_steps, num_class] 
        #rnn_inputs = [tf.squeeze(i,squeeze_dims=[1]) for i in tf.split(1, config.num_steps, x_one_hot)]
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
        
#         with tf.Session() as sess:
#             print(sess.run(tf.shape(rnn_outputs)))
        """
        Predictions, loss, training step
        Losses and total_loss are simlar to the "sequence_loss_by_example" and "sequence_loss"
        functions, respectively, from Tensorflow's api. See:
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/seq2seq.py
        """

        #logits and predictions
        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [config.state_size, config.num_classes])
            b = tf.get_variable('b', [config.num_classes], initializer=tf.constant_initializer(0.0))
            
        '''
        # for static rnn, for dyn rnn needs to update
        logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
        '''
            
        
        # for dyn rnn
        #logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
        
        logits = tf.matmul(tf.reshape(rnn_outputs, [-1, config.state_size]), W) + b 
        
        
        
        
        #predictions = [tf.nn.softmax(logit) for logit in logits]
        predictions = tf.nn.softmax(logits)

        #losses and train_step
        #losses = [tf.nn.softmax_cross_entropy_with_logits(logit,label) for logit, label in zip(logits, y_as_list)]
        
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_as_list)
        total_loss = tf.reduce_mean(losses)
        train_step = tf.train.AdagradOptimizer(config.learning_rate).minimize(total_loss)

        return losses, total_loss, final_state, train_step, x, y, init_state