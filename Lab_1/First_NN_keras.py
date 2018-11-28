from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Create a network of two dense layers
# The second layer outputs probabilities of each digit
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

# Set optimizer, loss function and metrics to measure
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Normalize input data and convert into float32 array
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# Encode labels categorically
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Train the model by fitting it to the training data
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# Test how well network performs on test data
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

