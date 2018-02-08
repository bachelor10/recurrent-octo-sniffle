import os
import keras
from xml_parse import model_data_generator


model_path = os.getcwd() + '/machine_learning/segmentation_model.h5'

model = keras.models.load_model(model_path)



img = []
bounding_list = []

for (image, bounding_boxes) in model_data_generator():
    full_box = np.zeros((10, 4))
    img.append(np.asarray(image))

    for i, box in enumerate(bounding_boxes):
        if i == 10: break
        full_box[i] = box

    bounding_list.append(full_box.flatten())

bounding_list = np.asarray(bounding_list).astype('float32')
img = np.asarray(img).astype('float32')


