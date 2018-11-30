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
    # embedded_words = sorted(list(embeddings_dict.keys()))
    return emb_dict


embedding_file = '/Users/Anton/Documents/LTH/EDAN95/Datasets/glove.6B/glove.6B.100d.txt'
embeddings_dict = load(embedding_file)

print(embeddings_dict['table'])

similarity = 0
similarityKey = 'table'

for key in embeddings_dict.keys():
    current = skm.pairwise.cosine_similarity(embeddings_dict['table'], embeddings_dict[key])
    if current > similarity:
        similarity = current
        similarityKey = key


print(similarityKey)


# co_sim = cosine_similarity(embeddings_dict['table'],embeddings_dict)

# co_sim_max = np.argmax (co_sim, axis=1)

# print (max = np.amax(embeddings_dict['table']))

