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

# As Pierre's example program 4.3

# Some parameters
OPTIMIZER = 'rmsprop'
BATCH_SIZE = 128
EPOCHS = 10
EMBEDDING_DIM = 100  # How is embedding dimension decided?
MAX_SEQUENCE_LENGTH = 150
LSTM_UNITS = 512


# Makes the GloVe embeddings into a dictionary
def load(file):
    file = file
    embeddings = {}
    glove = open(file)
    for line in glove:
        values = line.strip().split()
        word = values[0]
        vector = np.array(values[1:], dtype='float32')
        vector /= np.sqrt(np.dot(vector, vector))
        embeddings[word] = vector
    glove.close()
    embeddings_dict = embeddings
    embedded_words = sorted(list(embeddings_dict.keys()))
    return embeddings_dict, embedded_words


embedding_file = '/Users/Anton/Documents/LTH/EDAN95/Datasets/glove.6B/glove.6B.100d.txt'
embeddings_dict, embedded_words = load(embedding_file)

# Loading the corpus
BASE_DIR = '/Users/Anton/Documents/LTH/EDAN95/Datasets/NER-data'


# Loads conll data and makes into sentences
def load_conll2003_en():
    train_file = BASE_DIR + '/eng.train'
    dev_file = BASE_DIR + '/eng.valid'
    test_file = BASE_DIR + '/eng.test'
    column_names = ['form', 'ppos', 'pchunk', 'ner']
    train_sentences = open(train_file).read().strip()
    dev_sentences = open(dev_file).read().strip()
    test_sentences = open(test_file).read().strip()
    return train_sentences, dev_sentences, test_sentences, column_names


# Converting the corpus into a dictionary
if __name__ == '__main__':
    train_sentences, dev_sentences, test_sentences, column_names = load_conll2003_en()

    conll_dict = CoNLLDictorizer(column_names, col_sep=' +')
    train_dict = conll_dict.transform(train_sentences)
    dev_dict = conll_dict.transform(dev_sentences)  # added for dev and test as well
    test_dict = conll_dict.transform(test_sentences)
    print('First sentence, train:', train_dict[0])


# Function to build the two-way sequences
# Two vectors: x and Y
# Instead of extracting_dictorizer
def build_sequences(corpus_dict, key_x='form', key_y='ner', tolower=True):
    """
    Creates sequences from a list of dictionaries
    :param corpus_dict:
    :param key_x:
    :param key_y:
    :return:
    """
    X = []
    Y = []
    for sentence in corpus_dict:
        x = []
        y = []
        for word in sentence:
            x += [word[key_x]]
            y += [word[key_y]]
        if tolower:
            x = list(map(str.lower, x))
        X += [x]
        Y += [y]
    return X, Y


X_train_cat, Y_train_cat = build_sequences(train_dict, key_x='form', key_y='ner')
print('First sentence, words', X_train_cat[2])
print('First sentence, NER', Y_train_cat[1])

# Extracting the unique words and NER (does it work to just change POS into NER?)
vocabulary_words = sorted(list(set([word for sentence in X_train_cat for word in sentence])))
ner = sorted(list(set([ner for sentence in Y_train_cat for ner in sentence])))
print(ner)
print(len(ner))
NB_CLASSES = len(ner)
ner

# We create the dictionary
# We add two words for the padding symbol and unknown words
embeddings_words = embeddings_dict.keys()
print('Words in GloVe:', len(embeddings_dict.keys()))
vocabulary_words = sorted(list(set(vocabulary_words +
                                   list(embeddings_words))))
cnt_uniq = len(vocabulary_words) + 2
print('# unique words in the vocabulary: embeddings and corpus:',
      cnt_uniq)


# Function to convert the words or NER
def to_index(X, idx):
    """
    Convert the word lists (or NER lists) to indexes
    :param X: List of word (or NER) lists
    :param idx: word to number dictionary
    :return:
    """
    X_idx = []
    for x in X:
        # We map the unknown words to one
        x_idx = list(map(lambda x: idx.get(x, 1), x))
        X_idx += [x_idx]
    return X_idx


# We create the indexes
# We start at one to make provision for the padding symbol 0
# in RNN and LSTMs and 1 for the unknown words
rev_word_idx = dict(enumerate(vocabulary_words, start=2))
rev_ner_idx = dict(enumerate(ner, start=2))
word_idx = {v: k for k, v in rev_word_idx.items()}
ner_idx = {v: k for k, v in rev_ner_idx.items()}
# print('word index:', list(word_idx.items())[:10])
# print('NER index:', list(ner_idx.items())[:10])

# We create the parallel sequences of indexes
print('X', X_train_cat[1])
print('Y', Y_train_cat[1])
X_idx = to_index(X_train_cat, word_idx)
Y_idx = to_index(Y_train_cat, ner_idx)
print('First sentences, word indices', X_idx[:3])
print('First sentences, NER indices', Y_idx[:3])

# We pad the sentences
X = pad_sequences(X_idx)
Y = pad_sequences(Y_idx)

print(X[:3])
print(Y[:3])

# The number of NER classes and 0 (padding symbol)
Y_train = to_categorical(Y, num_classes=len(ner) + 2)
print(Y_train[0][0])
print(Y_train[0][-1])

# We create an embedding matrix
rdstate = np.random.RandomState(1234567)
embedding_matrix = rdstate.uniform(-0.05, 0.05,
                                   (len(vocabulary_words) + 2,
                                    EMBEDDING_DIM))

for word in vocabulary_words:
    if word in embeddings_dict:
        # If the words are in the embeddings, we fill them with a value
        embedding_matrix[word_idx[word]] = embeddings_dict[word]

# print('Shape of embedding matrix:', embedding_matrix.shape)
# print('Embedding of table', embedding_matrix[word_idx['table']])
# print('Embedding of the padding symbol, idx 0, random numbers',
#      embedding_matrix[0])

if models.load_model('ModelForNER_w_LSTM') == False:
    # We build the model
    model = models.Sequential()
    model.add(layers.Embedding(len(vocabulary_words) + 2,
                               EMBEDDING_DIM,
                               mask_zero=True,
                               input_length=None))
    model.layers[0].set_weights([embedding_matrix])
    # The default is True
    model.layers[
        0].trainable = False  # Should be false according to Chollet p.191 - Change from true as in Pierre's ex.
    # model.add(SimpleRNN(100, return_sequences=True))
    model.add(layers.Dropout(0.5))
    model.add(Bidirectional(SimpleRNN(100, return_sequences=True)))
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(Dense(NB_CLASSES + 2, activation='softmax'))  # Multiple categories and not binary as ex. pos./neg. review

    # We fit the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    model.summary()
    history = History()
    history = model.fit(X, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # We save the model
    model.save('ModelForNER_w_LSTM')

    # We plot and show accuracy and loss
    acc = history.history['acc']
    # val_acc = history.history['val_acc']  # Dev instead of val?
    loss = history.history['loss']
    # val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    # plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training loss')
    plt.legend()
    plt.show()


else:
    model = models.load_model('ModelForNER_w_LSTM')


# Evaluate the model
# Formatting the test set
# In X_dict, we replace the words with their index
X_test_cat, Y_test_cat = build_sequences(test_dict)
# We create the parallel sequences of indexes
X_test_idx = to_index(X_test_cat, word_idx)
Y_test_idx = to_index(Y_test_cat, ner_idx)

# print('X[0] test idx', X_test_idx[0])
# print('Y[0] test idx', Y_test_idx[0])

X_test_padded = pad_sequences(X_test_idx)
Y_test_padded = pad_sequences(Y_test_idx)
# print('X[0] test idx passed', X_test_padded[0])
# print('Y[0] test idx padded', Y_test_padded[0])
# One extra symbol for 0 (padding)
Y_test_padded_vectorized = to_categorical(Y_test_padded,
                                          num_classes=len(ner) + 2)
# print('Y[0] test idx padded vectorized', Y_test_padded_vectorized[0])
# print(X_test_padded.shape)
# print(Y_test_padded_vectorized.shape)


# Evaluation using the evaluate-method
# Evaluates with the padding symbol
test_loss, test_acc = model.evaluate(X_test_padded,
                                     Y_test_padded_vectorized)
print('Loss:', test_loss)
print('Accuracy:', test_acc)

# Evaluating on all the test corpus
# print('X_test', X_test_cat[0])
# print('X_test_padded', X_test_padded[0])
corpus_ner_predictions = model.predict(X_test_padded)
# print('Y_test', Y_test_cat[0])
# print('Y_test_padded', Y_test_padded[0])
# print('predictions', corpus_ner_predictions[0])

# Remove the padding
ner_pred_num = []
for sent_nbr, sent_ner_predictions in enumerate(corpus_ner_predictions):
    ner_pred_num += [sent_ner_predictions[-len(X_test_cat[sent_nbr]):]]
# print(ner_pred_num[:2])

# Convert NER indices to symbols
ner_pred = []
for sentence in ner_pred_num:
    ner_pred_idx = list(map(np.argmax, sentence))
    ner_pred_cat = list(map(rev_ner_idx.get, ner_pred_idx))
    ner_pred += [ner_pred_cat]

# print(ner_pred[:2])
# print(Y_test_cat[:2])

# Saving a file with the output
f = open("output_LSTM_2", "w+")
for id_s, sentence in enumerate(X_test_cat):
    for id_w, word in enumerate(sentence):
        f.write(X_test_cat[id_s][id_w] + " " + Y_test_cat[id_s][id_w] + " " + ner_pred[id_s][id_w] + "\n")
    f.write("\n")
f.close()