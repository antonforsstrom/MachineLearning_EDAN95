"""
Evaluation of CNN to classify 5 species of flowers using Keras
Author: Anton Forsström
"""

import os
import pickle
import numpy as np
from keras import models
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

base = '/Users/Anton/Documents/LTH'
dataset = os.path.join(base, 'EDAN95/Datasets/flowers_split')
test_dir = os.path.join(dataset, 'validation')

# Import saved trained model
model = models.load_model('flowers_recognition_case2.h5')

# Create generator for test data
test_datagen = ImageDataGenerator(1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=1,
    class_mode='categorical',
    shuffle=False)

# Evaluate model against test data
print('Metrics being evaluated:', model.metrics_names)
eval_hist = model.evaluate_generator(
   test_generator,
   steps=865)

print('Evaluation loss:', eval_hist[0])
print('Evaluation accuracy:', eval_hist[1])

# Predict and create Confusion Matrix and Classification Report
# Y_pred = model.predict_generator(test_generator, 865)
# print('Y_pred:', Y_pred)
# y_pred = np.argmax(Y_pred, axis=1)
# print('y_pred:', y_pred)
#
# print('Classes:', test_generator.classes)
# print('Confusion Matrix')
# print(confusion_matrix(test_generator.classes, y_pred))
#
# target_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
# print('Classification Report')
# print(classification_report(test_generator.classes, y_pred, target_names=target_names))

# Pickle in history from training model
filename = 'history_data_case2'
infile = open(filename, 'rb')
history = pickle.load(infile)
infile.close()

# Test if pickling works as expected and then plot
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation_loss')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
