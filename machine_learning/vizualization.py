import matplotlib.pyplot as plt
import numpy as np
import keras, os
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers import MaxPooling2D, ZeroPadding2D
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras import backend as K
from keras.utils import np_utils
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

model = keras.models.load_model(os.getcwd() + '/my_model.h5')
filename = os.listdir(os.getcwd() + '/train2/7/')[10]
print("Filename", filename)
# choose any image to want by specifying the index
img_to_visualize = np.array([img_to_array(Image.open(os.getcwd() + '/train2/7/' + filename))])
print(img_to_visualize)

print(model.layers)
def layer_to_visualize(layer, filename):
    inputs = [K.learning_phase()] + model.inputs

    _convout1_f = K.function(inputs, [layer.output])
    def convout1_f(X):
        # The [0] is to disable the training phase flag
        return _convout1_f([0] + [X])

    convolutions = convout1_f(img_to_visualize)
    convolutions = np.squeeze(convolutions)

    print ('Shape of conv:', convolutions.shape)
    print('Len of shape', len(convolutions.shape))

    n = convolutions.shape[0]
    n = int(np.ceil(np.sqrt(n)))

    # Visualization of each filter of the layer
    fig = plt.figure(figsize=(12,8))
    if len(convolutions.shape) > 1:
        for i in range(len(convolutions)):
            ax = fig.add_subplot(n,n,i+1)
            ax.imshow(convolutions[i], cmap='gray')
        fig.savefig(filename)

    else:
        plt.bar(range(len(convolutions)), convolutions)
        plt.savefig(filename)

# Specify the layer to want to visualize
count = 0
for layer in model.layers:
    layer_to_visualize(layer, str(count) + '.png')
    count += 1


