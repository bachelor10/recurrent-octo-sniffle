import keras
from PIL import Image
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model

model = keras.models.load_model('my_model.h5')
plot_model(model, to_file='model.png', show_shapes=True)
"""
filepath = 'validation/rightarrow/rightarrow_3f42a332-13d8-41f3-ad5d-1877faefbddc.bmp'
img = Image.open(filepath)
i = img.convert('RGB')

def generate_dataset():
    print("Generating dataset")
    train_generator = ImageDataGenerator()
    print("Generating validation")

    validation_generator = ImageDataGenerator()
    print("Getting train data")

    train_generator = train_generator.flow_from_directory(
        'train',
        target_size=(26, 26),
        batch_size=32,
        class_mode='categorical')

    print("Getting validation data")

    validation_generator = validation_generator.flow_from_directory(
        'validation',
        target_size=(26, 26),
        batch_size=32,
        class_mode='categorical')

    return train_generator, validation_generator

train_generator, validation_generator = generate_dataset()

res = model.evaluate_generator(validation_generator)

print("CNN Erro: %.2f%%" % (100-res[1]*100))"""