import os
import keras
from xml_parse import model_data_generator, Equation

import numpy as np
from PIL import Image, ImageDraw

model_path = os.getcwd() + '/segmentation_model.h5'

model = keras.models.load_model(model_path)


prediction_img = None

img = []
bounding_list = []

for (image, bounding_boxes) in model_data_generator(limit=0):
    full_box = np.zeros((10, 4))
    prediction_img = image
    prediction_img.save('bilde.png')

    img.append(np.asarray(image))

    for i, box in enumerate(bounding_boxes):
        if i == 10: break
        full_box[i] = box

    bounding_list.append(full_box)

#bounding_list = np.asarray(bounding_list).astype('float32')
np_img = np.asarray(img).astype('float32')
prediction_img.convert('RGB')

draw = ImageDraw.Draw(prediction_img)

#for box in bounding_list:
    
    #draw.rectangle(((box[0] - box[2] / 2, box[1] - box[3] / 2), (box[0] + box[2] / 2, box[1] + box[3] / 2)), outline="green")

predicted_boxes = model.predict(np_img / 255)

print(np_img.shape)
predicted_boxes = np.reshape(predicted_boxes[0], (10, 4))
print(predicted_boxes.shape)


for box in predicted_boxes:
    print(box)
    draw.rectangle(((box[0] - box[2] / 2, box[1] - box[3] / 2), (box[0] + box[2] / 2, box[1] + box[3] / 2)), outline="red")

prediction_img.save("yup.png")
