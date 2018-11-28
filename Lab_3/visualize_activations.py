"""
Modified code snippet from Chollet: Deep learning with Python
Listings 5.26 - 5.30
"""

from keras.models import load_model
from keras.preprocessing import image
from keras import models
import numpy as np
import matplotlib.pyplot as plt

# Load model and show summary of layers
model = load_model('flowers_recognition_case2.h5')
model.summary()

# Preprocess single image that is unknown to the network
img_path = '/Users/Anton/Documents/LTH/EDAN95/Datasets/flowers_split/test/dandelion/138166590_47c6cb9dd0.jpg'

img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255

print(img_tensor.shape)

# Display the picture
plt.imshow(img_tensor[0])

# Instantiating model from input tensor
# Extract outputs of the top eight layers
layer_outputs = [layer.output for layer in model.layers[:8]]

# Create model that will return these outputs (one per layer), given model input
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# Running model in predict mode
# Returns list of five Numpy arrays, one per layer activation
activations = activation_model.predict(img_tensor)

# First layer activation is a feature map with 32 channels
first_layer_activation = activations[0]
print(first_layer_activation.shape)

# Plot fifth channel of the activation of the first layer of the model
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
plt.title('First layer activation, forth channel')

# Plot eight channel of the activation of the first layer of the model
plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')
plt.title('First layer activation, seventh channel')

plt.show()