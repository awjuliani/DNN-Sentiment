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
import nltk
import collections
import word2vec
import sys


def cleanForView(string):
    stringA = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    return stringA.strip().lower()
    
def getSentimentCNN(fileToLoad, modelDir):
    checkpoint_dir = "./rnn_runs/"+modelDir+"/checkpoints/"
    batch_size = 64
    x_test, y_test, vocabulary, vocabulary_inv,trainS = data_helpers.load_data_for_books("./data/"+fileToLoad+".txt")
    y_test = np.argmax(y_test, axis=1)
    print("Vocabulary size: {:d}".format(len(vocabulary)))
    print("Test set size {:d}".format(len(y_test)))
    
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            scores = graph.get_operation_by_name("output/scores").outputs[0]
            # Generate batches for one epoch
            batches = data_helpers.batch_iter(x_test, batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []
            all_scores = []
            for x_test_batch in batches:
                batch_scores = sess.run(scores, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                batch_predictions = np.argmax(batch_scores,axis=1)
                #batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])
                all_scores = np.concatenate([all_scores,batch_scores[:,1] - batch_scores[:,0]])
                
    mbs = float(len(all_predictions[all_predictions == 1]))/len(all_predictions)
    mss = np.mean(all_scores)
    print "Mean Binary Sentiment",mbs
    print "Mean Smooth Sentiment",mss
    return all_predictions,all_scores
    
def getSentimentRNN(fileToLoad,modelDir):
    checkpoint_dir = "./rnn_runs/"+modelDir+"/checkpoints/"
    batch_size = 64
    n_hidden = 256
    
    x_test, y_test, vocabulary, vocabulary_inv,trainS = data_helpers.load_data_for_books("./data/"+fileToLoad+".txt")
    y_test = np.argmax(y_test, axis=1)
    print("Vocabulary size: {:d}".format(len(vocabulary)))
    print("Test set size {:d}".format(len(y_test)))
    x_test = np.fliplr(x_test)
    
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            print("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("x_input").outputs[0]
            predictions = graph.get_operation_by_name("prediction").outputs[0]
            istate = graph.get_operation_by_name('initial_state').outputs[0]
            keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
            # Generate batches for one epoch
            batches = data_helpers.batch_iter(x_test, batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []
            all_scores = []
            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, istate: np.zeros((len(x_test_batch), 2*n_hidden)), keep_prob: 1.0})
                binaryPred = np.argmax(batch_predictions,axis=1)
                all_predictions = np.concatenate([all_predictions, binaryPred])
                all_scores = np.concatenate([all_scores, batch_predictions[:,1] - batch_predictions[:,0]])
                
        mbs = float(len(all_predictions[all_predictions == 1]))/len(all_predictions)
        mss = np.mean(all_scores)
        print "Mean Binary Sentiment",mbs
        print "Mean Smooth Sentiment",mss
        return all_predictions,all_scores
        

        
def saveSentiment(fileToSave,all_predictions,all_scores):
    text = ''.join(open("./data/"+fileToSave+".txt").readlines()).decode('utf8')

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    book = tokenizer.tokenize(text)
    book = [cleanForView(sent) for sent in book]
    toOut = zip(book,all_predictions,all_scores)
    
    import unicodecsv as csv
    myfile = open(fileToSave+'.csv', 'wb')
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(["Text","Binary_Sentiment","Cont_Sentiment"])
    for row in toOut:
        wr.writerow(row)
    print "Saved",fileToSave+'.csv'
    
arguments = sys.argv
book = arguments[1]
nntype = arguments[2]
modelDir = arguments[3]

if nntype == "CNN":
    all_predictions,all_scores = getSentimentCNN(book,modelDir)
    saveSentiment(book,all_predictions,all_scores)
elif nntype == "RNN":
    all_predictions,all_scores = getSentimentRNN(book,modelDir)
    saveSentiment(book,all_predictions,all_scores)
else:
    print "Please choose a neural network type: CNN or RNN"
