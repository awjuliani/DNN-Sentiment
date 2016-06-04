# DNN-Sentiment
Convolutional and recurrent deep neural networks for text sentiment analysis.

Convolutional model based on [cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf).

Recurrent model based on [example by aymericdamien](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py).

Both models written in Tensorflow, and have been designed to allow for analyzing of sentinment of large text documents,
such as works of literature once trained. Both can use word2vec to obtain pre-trained word embeddings to improve performance.

To train each model, run train_cnn.py and train_rnn.py.

### In order to obtain sentiment of a text file, run:

`python generateSentiment.py textfile model checkpoint`

`textfile` is the document to analyze sentiment from. Don't include .txt extension. For example, when processing 'kafka.txt,' use 'kafka'

`model` is either "RNN" or "CNN"

`checkpoint` is the latest model checkpoint, contained in the runs or rnn_runs folder
