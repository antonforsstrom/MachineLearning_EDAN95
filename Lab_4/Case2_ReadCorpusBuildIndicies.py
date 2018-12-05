"""
RNN to perform NER on the conll 2003 english dataset.

Modified example from Pierre Nugues script:
https://github.com/pnugues/edan95/blob/master/programs/4.3-rnn-pos-tagger.ipynb
"""

from conll_dictorizer import CoNLLDictorizer, Token
from keras import models, layers, callbacks
import numpy as np
from keras.layers import LSTM, Bidirectional, SimpleRNN, Dense
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from keras.callbacks import History


# Some parameters to be used
optimizer = 'rmsprop'
batch_size = 128
epochs = 2
embedding_dim = 100
max_sequence_length = 150
# lstm_units = 512


# Loading embeddings from GloVe and returning a dictionary
def load(file):
    """
    Return the embeddings in the from of a dictionary
    :param file:
    :return dictionary:
    """
    file = file
    embeddings = {}
    glove = open(file)
    for line in glove:
        values = line.strip().split()
        word = values[0]
        vector = np.array(values[1:], dtype='float32')
        embeddings[word] = vector
    glove.close()
    emb_dict = embeddings
    embedded_words = sorted(list(embeddings_dict.keys()))
    return emb_dict, embedded_words


embedding_file = '/Users/Anton/Documents/LTH/EDAN95/Datasets/glove.6B/glove.6B.100d.txt'
embeddings_dict = load(embedding_file)


# Loading the corpus
def load_conll2003_en():
    base_dir = '/Users/Anton/Documents/LTH/EDAN95/Datasets/NER-data'
    train_file = base_dir + '/eng.train'
    valid_file = base_dir + '/eng.valid'
    test_file = base_dir + '/eng.test'
    # How do we know these column names?
    column_names = ['form', 'ppos', 'pchunk', 'ner']

    train_sentences = open(train_file).read().strip()
    valid_sentences = open(valid_file).read().strip()
    test_sentences = open(test_file).read().strip()

    return train_sentences, valid_sentences, test_sentences, column_names


# Converting the corpus into a dictionary
if __name__ == '__main__':
    train_sentences, valid_sentences, test_sentences, column_names = load_conll2003_en()

    # Why use '+' as sep?
    conll_dict = CoNLLDictorizer(column_names)
    train_dict = conll_dict.transform(train_sentences)
    valid_dict = conll_dict.transform(valid_sentences)
    test_dict = conll_dict.transform(test_sentences)
    print('First sentence, train:', train_dict[0])