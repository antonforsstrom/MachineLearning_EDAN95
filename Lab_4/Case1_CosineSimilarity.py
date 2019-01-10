"""
Import the Glove 6B embeddings and store them in a dictionary,
where the keys will be the words and the values, the embeddings.
"""


import numpy as np
import sklearn.metrics as skm


def load(file):
    """
    Return the embeddings in the from of a dictionary
    :param file:
    :return:
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
    return emb_dict


embedding_file = '/Users/Anton/Documents/LTH/EDAN95/Datasets/glove.6B/glove.6B.100d.txt'
embeddings_dict = load(embedding_file)

# Allow user to input which word to find similar words for
print('Find closest words to: ')
word = str(input())
print('Processing in progress...')

# Initiate dictionary to hold five most similar words
similarityKeys = {0: 'a', 0.0001: 'b', 0.0002: 'c', 0.0003: 'd', 0.0004: 'd'}

for key in embeddings_dict.keys():
    current = skm.pairwise.cosine_similarity(np.array(embeddings_dict[word]).reshape(1, -1),
                                             np.array(embeddings_dict[key]).reshape(1, -1))
    if current > min([float(number) for number in list(similarityKeys.keys())]) and str(key) != word:
        similarityKeys.pop(min(similarityKeys.keys()))
        similarityKeys[current[0][0]] = key

print('Most similar words and similarity value:')

keyList = similarityKeys.keys()
keyList = sorted(keyList, reverse=True)
for key in keyList:
    print("%s: %s" % (similarityKeys[key], key))


