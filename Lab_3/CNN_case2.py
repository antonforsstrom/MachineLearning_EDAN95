"""
Training of CNN to classify 5 species of flowers using Keras and Data augmentation
to reduce overfit.
Author: Anton Forsström
Inspiration and code examples taken from listing 5.5 and 5.11 in Chollet deep learning book
"""

import os
import pickle
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
# from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Reference directories of training and validation data
base = '/Users/Anton/Documents/LTH'
original_dataset_dir = os.path.join(base, 'EDAN95/Datasets/flowers')
dataset = os.path.join(base, 'EDAN95/Datasets/flowers_split')

train_dir = os.path.join(dataset, 'train')
validation_dir = os.path.join(dataset, 'validation')

# Create sequential CNN model
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.2)) # To potentially reduce overfitting
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))

# Print summary of model layers and parameters
model.summary()

# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255)

# Set up generators for training and validation data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150), # Resize all images to 150 x 150
    batch_size=21,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=21,
    class_mode='categorical')

# Quit training when validation loss is no longer reduced
# early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Train model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=130,
    epochs=80, # Adjust
    validation_data=validation_generator,
    validation_steps=43,
    # callbacks=[early_stopping]
    )

# Save the trained model
model.save('flowers_recognition_case2.h5')

# Pickle history
filename = 'history_data_case2'
outfile = open(filename, 'wb')
pickle.dump(history, outfile)
outfile.close()

# Evaluate performance over epochs
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
