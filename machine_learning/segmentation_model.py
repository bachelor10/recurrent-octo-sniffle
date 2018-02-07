from PIL import Image
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from xml_parse import model_data_generator
from io import BytesIO
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

img = []
bounding_boxes = []

for image, bounding_boxes in model_data_generator():
    #print("ImageShape", np.asarray(image).shape)
    #print("Image", np.asarray(image)/255)
    img.append(np.asarray(image)/255)
    #print("BoundingShape", np.asarray(bounding_boxes).shape)
    #print("BoundingBox", np.asarray(bounding_boxes)/255)
    bounding_boxes.append(np.asarray(bounding_boxes)/255)

img = np.asarray(img)
bounding_boxes = np.asarray([bounding_boxes])

img = np.reshape(img, (1, 32, 64, 1))

print(img)


def generate_dataset():
    print("Generating dataset")
    train_generator = ImageDataGenerator(
        rescale=1. / 255,
    )
    print("Generating validation")

    validation_generator = ImageDataGenerator(
        rescale=1. / 255,
    )
    print("Getting train data")

    train_generator = train_generator.flow_from_directory(
        'segmentation_train',
        target_size=(32, 64),
        batch_size=64,
        color_mode='grayscale',
        class_mode='categorical',
    )
    for image in train_generator:
        print(image)
        break
    print("Getting validation data")

    validation_generator = validation_generator.flow_from_directory(
        'segmentation_validation',
        target_size=(26, 26),
        batch_size=64,
        color_mode='grayscale',
        class_mode='categorical')

    return train_generator, validation_generator


print("Creating model")
filter_size = 3
pool_size = 2


model = Sequential([
    Conv2D(32, kernel_size=(6, 6), input_shape=(32, 64, 1), dim_ordering='tf', activation='relu'),
    MaxPooling2D(pool_size=(pool_size, pool_size)),
    Conv2D(64, filter_size, filter_size, dim_ordering='tf', activation='relu'),
    MaxPooling2D(pool_size=(pool_size, pool_size)),
    Conv2D(128, filter_size, filter_size, dim_ordering='tf', activation='relu'),
    # #         MaxPooling2D(pool_size=(pool_size, pool_size)),
    Conv2D(128, filter_size, filter_size, dim_ordering='tf', activation='relu'),
    # #         MaxPooling2D(pool_size=(pool_size, pool_size)),
    Flatten(),
    Dropout(0.4),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(20)
])

print(model.summary)

print("Compiling model")

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])



print("Fitting model")
model.fit(
    img,
    bounding_boxes,
    epochs=1
)

print("Done!")
model.save('segmentation_model.h5')

from keras.utils import plot_model

plot_model(model, to_file='model.png', show_shapes=True)