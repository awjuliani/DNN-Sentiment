import tensorflow as tf
#from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
import math

import re
import itertools
from collections import Counter

import random
import nltk
import collections
import word2vec
from text_rnn import TextRNN

# Load training data
x_train, y_train, vocabulary, vocabulary_inv = data_helpers.load_data()
#y_test = np.argmax(y_test, axis=1)
vocab_size = len(vocabulary)
print("Vocabulary size: {:d}".format(vocab_size))
print("Test set size {:d}".format(len(y_train)))

# Generate word embeddings
data = x_train.flatten()
iterations = 1000
data = data[data!=468]
w2v = word2vec.word2vec(data,vocabulary,vocabulary_inv,vocab_size,iterations)
final_embeddings = w2v.runWord2Vec()

x_train = np.fliplr(x_train)
fullTrain = np.concatenate((x_train,y_train),axis=1)
shuffledTrain = np.random.permutation(fullTrain)
sTrain, sTest = np.vsplit(shuffledTrain,[9000])
x_train,y_train = np.hsplit(sTrain,[-2])
x_test,y_test = np.hsplit(sTest,[-2])

# Parameters
learning_rate = 0.01
training_iters = 1000
batch_size = 64
display_step = 10
keep = 0.5

# Network Parameters
n_input = 1 
n_steps = 56 # Sentence length
n_hidden = 256 # hidden layer num of features
n_classes = 2 
embedding_size = 128

# Load model
myModel = TextRNN(learning_rate,n_input,n_steps,n_hidden,n_classes,embedding_size,vocab_size,True,final_embeddings)


# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    
    #Writing Directory information
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "rnn_runs", timestamp))
    print("Writing to {}\n".format(out_dir))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.all_variables())
    
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    fullTrain = np.concatenate((x_train,y_train),axis=1)
    while step < training_iters:
        perms = np.random.permutation(fullTrain)
        for i in range(perms.shape[0]/batch_size):
            batch = perms[i *batch_size:(i+1) * batch_size,:]
            batch_xs,batch_ys = np.hsplit(batch,[-2])
            # Reshape data 
            #batch_xs = batch_xs.reshape((batch_size, n_steps))
            # Fit training using batch data
            sess.run(myModel.optimizer, feed_dict={myModel.x: batch_xs, myModel.y: batch_ys, myModel.keep_prob: keep,
                                       myModel.istate: np.zeros((batch_size, 2*n_hidden))})
            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(myModel.accuracy, feed_dict={myModel.x: batch_xs, myModel.y: batch_ys, myModel.keep_prob: 1.0,
                                                    myModel.istate: np.zeros((batch_size, 2*n_hidden))})
                # Calculate batch loss
                loss = sess.run(myModel.cost, feed_dict={myModel.x: batch_xs, myModel.y: batch_ys, myModel.keep_prob: 1.0,
                                                 myModel.istate: np.zeros((batch_size, 2*n_hidden))})
                print "Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                      ", Training Accuracy= " + "{:.5f}".format(acc)
            step += 1
    path = saver.save(sess, checkpoint_prefix, global_step=step)
    print("Saved model checkpoint to {}\n".format(path))
    print "Optimization Finished!"
    test_len = len(y_test)
    print "Testing Accuracy:", sess.run(myModel.accuracy, feed_dict={myModel.x: x_test, myModel.y: y_test, myModel.keep_prob: 1.0,
                                                             myModel.istate: np.zeros((test_len, 2*n_hidden))})








