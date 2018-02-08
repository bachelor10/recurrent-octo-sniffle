import os
import keras
from machine_learning.xml_parse import model_data_generator, Equation

import numpy as np
from PIL import Image, ImageDraw

model_path = os.getcwd() + '/machine_learning/segmentation_model.h5'

model = keras.models.load_model(model_path)


prediction_img = None

img = []
bounding_list = []

for (image, bounding_boxes) in model_data_generator(limit=1):
    full_box = np.zeros((10, 4))
    prediction_img = image
    img.append(np.asarray(image))

    for i, box in enumerate(bounding_boxes):
        if i == 10: break
        full_box[i] = box

    bounding_list.append(full_box.flatten())

image = Image.new('LA', (Equation.IMG_WIDTH, Equation.IMG_HEIGHT), "white")

draw = ImageDraw.Draw(image)

for box in bounding_list:
    draw.rectangle(((box[0] - box[2] / 2, box[1] - box[3] / 2), (box[0] + box[2] / 2, box[1] + box[3] / 2)), outline="green")

predicted_boxes = model.predict(img)

predicted_boxes = np.reshape(predicted_boxes[0], (10, 4))

for box in predicted_boxes:
    draw.rectangle(((box[0] - box[2] / 2, box[1] - box[3] / 2), (box[0] + box[2] / 2, box[1] + box[3] / 2)), outline="red")

image.save("yup.png")

