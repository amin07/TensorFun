'''
Created on Sep 14, 2017
This program creates a LSTM for language modeling.
Basically it takes text as input and gets trained.
After that, it can generate texts like texts it was trained on.
Most codes taken from https://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html
@author: Amin
'''

import tensorflow as tf
from scipy.io import loadmat
import numpy as np
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import urllib
import ptbiterator


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


def main(_):
    print ('inside main')
  
    
if __name__ == '__main__':
    tf.app.run()
