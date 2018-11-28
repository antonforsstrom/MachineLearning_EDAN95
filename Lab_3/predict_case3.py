"""
Predict labels using the network trained on top of pretrained Inception V3 network.
Author: Anton Forsström
"""

import os
import numpy as np
from keras import models
import sklearn.metrics


base = '/Users/Anton/Documents/LTH/EDAN95/Datasets/flowers_split'
test_dir = os.path.join(base, 'test')

# Load saved trained model and features
model = models.load_model('flowers_recognition_case3.h5')

features = np.load('features.npy.npz')

# Evaluate model
loss, acc = model.evaluate(features['test_feat'], features['test_labels'])

# Predict test
Y_predicted = model.predict(features['test_feat'])

y_pred = np.argmax(Y_predicted, axis=1)
y_true = np.argmax(features['test_labels'], axis=1)

cm = sklearn.metrics.confusion_matrix(
    y_true,
    y_pred)

print('Confusion matrix:\n', cm)

print('Loss:', round(loss, 2))

print('Accuracy:', round(acc, 2))
