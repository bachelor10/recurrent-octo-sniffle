from PIL import Image
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from io import BytesIO
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from xml_parse import model_data_generator

img = []
bounding_list = []


for (image, bounding_boxes) in model_data_generator():
    full_box = np.zeros((10, 4))
    img.append(np.asarray(image))

    for i, box in enumerate(bounding_boxes):
        if i == 10: break
        full_box[i] = box

    bounding_list.append(full_box.flatten())
    #img.append(np.asarray(image) / 255)
    #bounding_list.append(x/255 for x in bounding_boxes) # TODO this is the problem, scale down coordinates with 255, same as image data.


bounding_list = np.asarray(bounding_list).astype('float32')
img = np.asarray(img).astype('float32')



# img = np.reshape(img, (1, 32, 64, 1))


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
        # print("Image in train_generator", image)
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
print("SHape", img.shape, bounding_list.shape)

model = Sequential([
    Conv2D(32, kernel_size=(6, 6), input_shape=(32, 64, 2), dim_ordering='tf', activation='relu'),
    MaxPooling2D(pool_size=(3, 3)),
    Conv2D(64, filter_size, filter_size, dim_ordering='tf', activation='relu'),
    MaxPooling2D(pool_size=(pool_size, pool_size)),
    Conv2D(128, filter_size, filter_size, dim_ordering='tf', activation='relu'),
    # #         MaxPooling2D(pool_size=(pool_size, pool_size)),
    #Conv2D(128, filter_size, filter_size, dim_ordering='tf', activation='relu'),
    # #         MaxPooling2D(pool_size=(pool_size, pool_size)),
    Flatten(),
    Dropout(0.4),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(40)
])

# print(model.summary())
# print(model.inputs)

print("Compiling model")

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])



print("Fitting model")
print("Input", img.shape, bounding_list.shape)
model.fit(
    img,
    bounding_list,
    epochs=500
)

print("Done!")
model.save('segmentation_model.h5')

from keras.utils import plot_model

#plot_model(model, to_file='model.png', show_shapes=True)
