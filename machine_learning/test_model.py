import keras
from PIL import Image
import numpy as np
model = keras.models.load_model('my_model.h5')

filepath = 'validation/rightarrow/rightarrow_3f42a332-13d8-41f3-ad5d-1877faefbddc.bmp'
img = Image.open(filepath)
i = img.convert('RGB')


print(i.size)
in_data = np.asarray([np.asarray(i)])

print(in_data.shape)

res = model.predict(in_data)

print("RES", res)