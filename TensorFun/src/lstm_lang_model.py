'''
Created on Sep 14, 2017
This program creates a LSTM for language modeling.
Basically it takes text as input and gets trained.
After that, it can generate texts like texts it was trained on.
Most codes taken from https://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html
@author: Amin
'''


from scipy.io import loadmat
import numpy as np
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import urllib
import ptbiterator
import time


file_url = 'https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/data/tiny-shakespeare.txt'
file_name = 'tinyshakespeare.txt'
if not os.path.exists(file_name):
    urllib.request.urlretrieve(file_url, file_name)

with open(file_name,'r') as f:
    raw_data = f.read()
    print("Data length:", len(raw_data))

vocab = set(raw_data)
vocab_size = len(vocab)

idx_to_vocab = dict(enumerate(vocab))    
vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))

data = [vocab_to_idx[c] for c in raw_data]
del raw_data


def gen_epochs(n, num_steps, batch_size):
    for i in range(n):
        yield ptbiterator.ptb_iterator(data, batch_size, num_steps)


def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()
    
    
def train_network(g, num_epochs, num_steps = 200, batch_size = 32, verbose = True, save=False):
    tf.set_random_seed(2345)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps, batch_size)):
            training_loss = 0
            steps = 0
            training_state = None
            for X, Y in epoch:
                steps += 1

                feed_dict={g['x']: X, g['y']: Y}
                if training_state is not None:
                    feed_dict[g['init_state']] = training_state
                training_loss_, training_state, _ = sess.run([g['total_loss'],
                                                      g['final_state'],
                                                      g['train_step']],
                                                             feed_dict)
                training_loss += training_loss_
            if verbose:
                print("Average training loss for Epoch", idx, ":", training_loss/steps)
            training_losses.append(training_loss/steps)

        if isinstance(save, str):
            g['saver'].save(sess, save)

    return training_losses


def build_basic_rnn_graph_with_list(
    state_size = 100,
    num_classes = vocab_size,
    batch_size = 32,
    num_steps = 200,
    learning_rate = 1e-4):

    reset_graph()

    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

    x_one_hot = tf.one_hot(x, num_classes)
#     with tf.Session() as sess:
#         print (sess.run(tf.shape(x_one_hot)))
#    rnn_inputs = [tf.squeeze(i,squeeze_dims=[1]) for i in tf.split(1, num_steps, x_one_hot)]
#    updated version
#    rnn_inputs = [tf.squeeze(i,squeeze_dims=[1]) for i in tf.split( x_one_hot, num_steps,1)]
#    or we can use, but this will be problem when batch_size of x in None, in that case upper version
    rnn_inputs = tf.unstack(x_one_hot, axis=1)

#     with tf.Session() as sess:
#         print (sess.run([tf.shape(rnn_inputs), tf.shape(x_one_hot)]))
    
    cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
    init_state = cell.zero_state(batch_size, tf.float32)
    rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, initial_state=init_state)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    
    #logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
    #This was as list, need to change so that as tensor
    with tf.Session() as sess:
        print('rnn_output shape', sess.run(tf.shape(rnn_outputs)))

    logits = tf.reshape(tf.matmul(tf.reshape(rnn_outputs,[-1, state_size]), W) + b, [batch_size, num_steps, num_classes]) 
    #logits = tf.matmul(rnn_outputs, W) + b
    
    #y_as_list = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(y, num_steps, 1)]
    # or we can do with unstack
    #y_one_hot = tf.one_hot(y, num_classes)
    #y_as_list = tf.unstack(y, axis=1)
    y_as_list = y

    with tf.Session() as sess:
        print ('y_as_list shape', sess.run(tf.shape(y_as_list)))
        print ('logits shape', sess.run(tf.shape(logits)))
    #loss_weights = [tf.ones([batch_size]) for i in range(num_steps)]
    loss_weights = tf.ones([num_steps, batch_size])
    #losses = tf.nn.seq2seq.sequence_loss_by_example(logits, y_as_list, loss_weights)
    #losses = tf.contrib.seq2seq.sequence_loss_by_example(logits, y_as_list, loss_weights)
    # changed version
    losses = tf.contrib.seq2seq.sequence_loss(logits, y_as_list, loss_weights)
    total_loss = tf.reduce_mean(losses)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(
        x = x,
        y = y,
        init_state = init_state,
        final_state = final_state,
        total_loss = total_loss,
        train_step = train_step
    )

def main(_):
    print ('inside main')
  
    t = time.time()
    build_basic_rnn_graph_with_list()
    print("It took", time.time() - t, "seconds to build the graph.")
    
if __name__ == '__main__':
    tf.app.run()
