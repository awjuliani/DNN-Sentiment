import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import nltk
from nltk.tokenize import RegexpTokenizer
import string
from nltk.corpus import stopwords

class word2vec(object):
    
    def generate_batch(self, batch_size, num_skips, skip_window):
        #global self.data_index
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1 # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
        for i in range(batch_size // num_skips):
            target = skip_window  # target label at the center of the buffer
            targets_to_avoid = [ skip_window ]
            for j in range(num_skips):
              while target in targets_to_avoid:
                target = random.randint(0, span - 1)
              targets_to_avoid.append(target)
              batch[i * num_skips + j] = buffer[skip_window]
              labels[i * num_skips + j, 0] = buffer[target]
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
        return batch, labels
    
    def __init__(self,data, vocabulary, vocabulary_inv,vocab_size, iterations):
        print "Loaded"
        
        self.data_index = 0
        self.data = data
        self.dictionary = vocabulary
        self.reverse_dictionary = vocabulary_inv
        self.iterations = iterations
        self.vocabulary_size = vocab_size
    
    def runWord2Vec(self):
        batch, labels = self.generate_batch(batch_size=8, num_skips=2, skip_window=1)
        for i in range(8):
           print(batch[i], '->', labels[i, 0])
           print(self.reverse_dictionary[batch[i]], '->', self.reverse_dictionary[labels[i, 0]])
      
        # Step 4: Build and train a skip-gram model.

        batch_size = 128
        embedding_size = 128  # Dimension of the embedding vector.
        skip_window = 1       # How many words to consider left and right.
        num_skips = 2         # How many times to reuse an input to generate a label.

        # We pick a random validation set to sample nearest neighbors. Here we limit the
        # validation samples to the words that have a low numeric ID, which by
        # construction are also the most frequent.
        valid_size = 16     # Random set of words to evaluate similarity on.
        valid_window = 100  # Only pick dev samples in the head of the distribution.
        valid_examples = np.array(random.sample(np.arange(valid_window), valid_size))
        num_sampled = 64    # Number of negative examples to sample.

        graph = tf.Graph()

        with graph.as_default():

          # Input self.data.
          train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
          train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
          valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

          # Ops and variables pinned to the CPU because of missing GPU implementation
          with tf.device('/cpu:0'):
            # Look up embeddings for inputs.
            embeddings = tf.Variable(
                tf.random_uniform([self.vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            # Construct the variables for the NCE loss
            nce_weights = tf.Variable(
                tf.truncated_normal([self.vocabulary_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

          # Compute the average NCE loss for the batch.
          # tf.nce_loss automatically draws a new sample of the negative labels each
          # time we evaluate the loss. 
          loss = tf.reduce_mean(
             tf.nn.nce_loss(nce_weights, nce_biases, train_labels, embed,
                            num_sampled, self.vocabulary_size))


          # Construct the SGD optimizer using a learning rate of 1.0.
          optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

          # Compute the cosine similarity between minibatch examples and all embeddings.
          norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
          normalized_embeddings = embeddings / norm
          valid_embeddings = tf.nn.embedding_lookup(
              normalized_embeddings, valid_dataset)
          similarity = tf.matmul(
              valid_embeddings, normalized_embeddings, transpose_b=True)


        # Step 5: Begin training.
        num_steps = self.iterations

        with tf.Session(graph=graph) as session:
          # We must initialize all variables before we use them.
          tf.initialize_all_variables().run()
          print("Initialized")

          average_loss = 0
          for step in xrange(num_steps):
            batch_inputs, batch_labels = self.generate_batch(
                batch_size, num_skips, skip_window)
            feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
              if step > 0:
                average_loss /= 2000
              # The average loss is an estimate of the loss over the last 2000 batches.
              print("Average loss at step ", step, ": ", average_loss)
              average_loss = 0

            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
              sim = similarity.eval()
              for i in xrange(valid_size):
                valid_word = self.reverse_dictionary[valid_examples[i]]
                top_k = 8 # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log_str = "Nearest to %s:" % valid_word
                for k in xrange(top_k):
                  close_word = self.reverse_dictionary[nearest[k]]
                  log_str = "%s %s," % (log_str, close_word)
                print(log_str)
          final_embeddings = normalized_embeddings.eval()

        return final_embeddings
