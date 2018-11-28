"""
Training CNN by extracting features from pretrained conv base Inception V3.
Convolutional base trained on Imagenet dataset.
Author: Anton Forsström
"""

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionV3
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt
import pickle

# Load the convolutional base from InceptionV3
conv_base = InceptionV3(weights='imagenet',
                        include_top=False,
                        input_shape=(150, 150, 3))

# conv_base.summary()

# Reference directories of training, validation and test data
base = '/Users/Anton/Documents/LTH/EDAN95/Datasets/flowers_split'

train_dir = os.path.join(base, 'train')
validation_dir = os.path.join(base, 'validation')
test_dir = os.path.join(base, 'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 10


def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 3, 3, 2048))
    labels = np.zeros(shape=(sample_count, 5))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels


train_features, train_labels = extract_features(train_dir, 2592)
validation_features, validation_labels = extract_features(validation_dir, 865)
test_features, test_labels = extract_features(test_dir, 866)

# Reshape and flatten features
train_features = np.reshape(train_features, (2592, 3 * 3 * 2048))
validation_features = np.reshape(validation_features, (865, 3 * 3 * 2048))
test_features = np.reshape(test_features, (866, 3 * 3 * 2048))

# Save the features
np.savez('features.npy',
         train_feat=train_features,
         val_feat=validation_features,
         test_feat=test_features,
         train_labels=train_labels,
         val_labels=validation_labels,
         test_labels=test_labels)

# With the extracted features we can train our own classifier
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=3 * 3 * 2048))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(5, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='categorical_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, train_labels,
                    epochs=5,
                    batch_size=10,
                    validation_data=(validation_features, validation_labels))

# Save the model
model.save('flowers_recognition_case3.h5')

# Pickle history
filename = 'history_data_case3'
outfile = open(filename, 'wb')
pickle.dump(history, outfile)
outfile.close()

# Plot training and validation stats
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
