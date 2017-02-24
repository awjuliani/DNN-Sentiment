import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import math

import re
import itertools
from collections import Counter

import random
import collections
import word2vec

class TextRNN(object):

    def __init__(self, learning_rate,n_input,n_steps,n_hidden,n_classes,embedding_size,vocab_size,useW2V,final_embeddings):
        
        
        
        self.x = tf.placeholder("int32", [None, n_steps], name='x_input')
        self.istate = tf.placeholder("float", [None, 2*n_hidden], name='initial_state') #state & cell => 2x n_hidden
        self.y = tf.placeholder("float", [None, n_classes], name='y_True')
        self.keep_prob = tf.placeholder('float32', name="keep_prob")
        
        if (useW2V == False):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
        else:
            self.W = tf.Variable(final_embeddings,name="W")
        self.embedded_chars = tf.nn.embedding_lookup(self.W, self.x)
        self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Define weights
        self.weights = {
            'hidden': tf.Variable(tf.random_normal([embedding_size, n_hidden])), # Hidden layer weights
            'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
        }
        self.biases = {
            'hidden': tf.Variable(tf.random_normal([n_hidden])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }
        
        self.pred = RNN(self.embedded_chars, self.istate, self.weights, self.biases,embedding_size,self.keep_prob,n_hidden,n_steps)
        self.fixedPred = tf.argmax
        # Define loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y)) # Softmax loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost) # Adam Optimizer

        # Evaluate model
        self.correct_pred = tf.equal(tf.argmax(self.pred,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        
def RNN(_X, _istate, _weights, _biases,embedding_size,keep_prob,n_hidden,n_steps):

        # input shape: (batch_size, n_steps, n_input)
        _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
        # Reshape to prepare input to hidden activation
        _X = tf.reshape(_X, [-1, embedding_size]) # (n_steps*batch_size, n_input)
        # Linear activation
        _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']

        # Define a lstm cell with tensorflow
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple = False)
        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        # _X = tf.split(0, n_steps, _X) # n_steps * (batch_size, n_hidden)
        _X = tf.split(_X, n_steps, 0) # n_steps * (batch_size, n_hidden)

        # Get lstm cell output
        # outputs, states = rnn.rnn(lstm_cell, _X, initial_state=_istate)
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, _X, initial_state=_istate)
        out_drop = tf.nn.dropout(outputs[-1],keep_prob)
        # Linear activation
        # Get inner loop last output
        return tf.nn.xw_plus_b(out_drop, _weights['out'],_biases['out'],name='prediction')
        
