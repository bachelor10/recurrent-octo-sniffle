from PIL import Image
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv1D

from io import BytesIO
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from xml_parse import model_data_generator

img = []
bounding_list = []


for (image, bounding_boxes) in model_data_generator(limit=10):
    full_box = np.zeros((10, 4))
    img.append(np.asarray(image))

    for i, box in enumerate(bounding_boxes):
        if i == 10: break
        full_box[i] = box
    # add flatten for fix
    bounding_list.append(full_box[0])
    #img.append(np.asarray(image) / 255)
    #bounding_list.append(x/255 for x in bounding_boxes) # TODO this is the problem, scale down coordinates with 255, same as image data.


bounding_list = np.asarray(bounding_list).astype('float32')
img = np.asarray(img).astype('float32')/255



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

''' model = Sequential([
    Conv2D(32, kernel_size=(6, 6), input_shape=(64, 128, 2), dim_ordering='tf', activation='relu'),
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
    Dense(4)
])
'''

model2 = Sequential()
model2.add(Conv1D(64, 3, input_shape=(64, 128)))
print(model2.output_shape)
model2.add(Dense(256, activation='relu'))
print(model2.output_shape)
model2.add(Dropout(0.2))
print(model2.output_shape)
model2.add(Dense(4))
print(model2.output_shape)



print("Compiling model")


model2.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adadelta()
)
# print(model.summary())
# print(model.inputs)


''' model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy']) '''

def IOU(bbox1, bbox2):
    '''Calculate overlap between two bounding boxes [x, y, w, h] as the area of intersection over the area of unity'''
    x1, y1, w1, h1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    x2, y2, w2, h2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]

    w_I = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_I = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_I <= 0 or h_I <= 0:  # no overlap
        return 0
    I = w_I * h_I

    U = w1 * h1 + w2 * h2 - I

    return I / U

def distance(bbox1, bbox2):
    return np.sqrt(np.sum(np.square(bbox1[:2] - bbox2[:2])))

num_epochs = 10
ious_epoch = np.zeros((len(img), num_epochs))
dists_epoch = np.zeros((len(img), num_epochs))
mses_epoch = np.zeros((len(img), num_epochs))

for epoch in range(num_epochs):
    print('Epoch:', epoch)
    print(bounding_list)
    model2.fit(img, bounding_list, epochs=1, verbose=1, batch_size=1)    
    
    pred_y = model.predict(img)

    for i, (pred_bboxes, exp_bboxes) in enumerate(zip(pred_y, bounding_list)):
            
        mse = np.mean(np.square(pred_bboxes - exp_bboxes))
        
        iou = IOU(pred_bboxes[:4], exp_bboxes[:4]) + IOU(pred_bboxes[4:], exp_bboxes[4:])
        
        dist = distance(pred_bboxes[:4], exp_bboxes[:4]) + distance(pred_bboxes[4:], exp_bboxes[4:])
        
        '''
        if mse_flipped < mse:  # you can also use iou or dist here
            flipped_train_y[i] = flipped_exp_bboxes
            flipped[i, epoch] = 1
            mses_epoch[i, epoch] = mse_flipped / 2.
            ious_epoch[i, epoch] = iou_flipped / 2.
            dists_epoch[i, epoch] = dist_flipped / 2.
        else:
            mses_epoch[i, epoch] = mse / 2.
            ious_epoch[i, epoch] = iou / 2.
            dists_epoch[i, epoch] = dist / 2. 
        '''
            
    print('Mean IOU: {}'.format(np.mean(ious_epoch[:, epoch])))
    print('Mean dist: {}'.format(np.mean(dists_epoch[:, epoch])))
    print('Mean mse: {}'.format(np.mean(mses_epoch[:, epoch])))



''' print("Fitting model")
print("Input", img.shape, bounding_list.shape)
model.fit(
    img,
    bounding_list,
    epochs=50,
    verbose=1
) 
'''

print("Done!")
model2.save('segmentation_model.h5')

from keras.utils import plot_model

#plot_model(model, to_file='model.png', show_shapes=True)
