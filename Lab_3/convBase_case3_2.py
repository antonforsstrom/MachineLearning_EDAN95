"""
Training CNN by adding densely connected layers on top of
pretrained conv base Inception V3.
Convolutional base trained on Imagenet dataset.
Author: Anton Forsström
"""

import os
from keras import models
from keras import layers
from keras.applications import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import pickle
import matplotlib.pyplot as plt

# Load the convolutional base from InceptionV3
conv_base = InceptionV3(weights='imagenet',
                        include_top=False,
                        input_shape=(150, 150, 3))

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))

model.summary()

print('# Trainable weights before freezing:', len(model.trainable_weights))

# Freeze updating of conv base
conv_base.trainable = False

print('# Trainable weights after freezing:', len(model.trainable_weights))

# Reference directories of training, validation and test data
base = '/Users/Anton/Documents/LTH/EDAN95/Datasets/flowers_split'

train_dir = os.path.join(base, 'train')
validation_dir = os.path.join(base, 'validation')
test_dir = os.path.join(base, 'test')

# Augment training data
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# Do not augment test data
test_datagen = ImageDataGenerator(rescale=1./255)

# Set up image generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical')

# Compile and train model
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.RMSprop(lr=2e-5),
    metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=130,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=43)

# Save the model
model.save('flowers_recognition_case3_2.h5')

# Save history
filename = 'history_data_case3_2'
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
