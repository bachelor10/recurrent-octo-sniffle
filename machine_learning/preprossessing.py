import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Dense, Flatten, Dropout
from io import BytesIO
import os
from matplotlib import pyplot as plt



from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def generate_dataset():
    print("Generating dataset")
    train_generator = ImageDataGenerator(
        rescale = 1./255
    )
    print("Generating validation")

    validation_generator = ImageDataGenerator(
        rescale = 1./255
    )
    print("Getting train data")

    train_generator = train_generator.flow_from_directory(
        'train2',
        target_size=(26,26),
        batch_size=64,
        color_mode='grayscale',
        class_mode='categorical',
    )

    """count = 0

    for image, y in train_generator:
        print("Shape", image[0].shape)
        img = (image[0] * 255)
        img = Image.fromarray(img.astype('uint8'))
        img.save('preview/' + str(count) + ".png")
        if count > 100: break
        count += 1"""
 
    print("Getting validation data")

    """img = load_img('data/train/cats/cat.0.jpg')  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `pre view/` directory
    i = 0
    """

    validation_generator = validation_generator.flow_from_directory(
        'validation2',
        target_size=(26, 26),
        batch_size=64,
        color_mode='grayscale',
        class_mode='categorical')

    print("Validation classes", validation_generator.classes)
    print("Validation class_indices", validation_generator.class_indices)
    print("Classes", os.listdir(os.getcwd() + '/train2'))
    return train_generator, validation_generator


train_generator, validation_generator = generate_dataset()

print("Creating model")
"""
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(26, 26, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(13, activation='softmax'))
"""
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(26,26,1), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))
model.add(Flatten())

# Fully connected layer
model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(39))
model.add(Activation('softmax'))

print("Compiling model")

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
print("Fitting model")

model.fit_generator(
        train_generator,
        epochs=10,
        validation_data=validation_generator
)

print("Done!")
model.save('my_model.h5')

#from keras.utils import plot_model

#plot_model(model, to_file='model.png', show_shapes=True)

# creates a HDF5 file 'my_model.h5'
#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])




#img = load_img('../bitmap_data/1516694853600.png')  # this is a PIL image

#transform_image(img)



