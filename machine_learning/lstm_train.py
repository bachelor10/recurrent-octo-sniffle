

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Dense, Flatten, Dropout, LSTM, Embedding
from io import BytesIO
import os
from matplotlib import pyplot as plt
from class_model import Predictor
from xml_parse import continous_symbol_generator
import numpy as np


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def generate_train_data(limit=1000):
    prediction_data = []
    truth_data = []
    for image, truth in continous_symbol_generator(limit=limit):
        try:
            Predictor.CLASS_INDICES[truth]
            prediction_data.append(img_to_array(image))
            one_hot = np.zeros(39)
            one_hot[Predictor.CLASS_INDICES[truth]] = 1
            truth_data.append(one_hot)
        except:
            pass
    
    return prediction_data, truth_data

def predict_classes(imgs):
    MODEL_PATH = os.getcwd() + '/my_model.h5'
    #CLASSES = ["+", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "="]#os.listdir(os.getcwd() + '/machine_learning' + '/train')    
    #MODEL_PATH = os.getcwd() + '/my_model.h5'

    model = keras.models.load_model(MODEL_PATH)

    return model.predict_proba(imgs)


def store_train_data(limit=1000):
    p_data, t_data = generate_train_data(limit=limit)
    print("P_DATA shape", np.asarray(p_data).shape)
    predicted = predict_classes(np.asarray(p_data))


    np.savetxt("predictions.csv", predicted, delimiter=",")
    np.savetxt("truths.csv", t_data, delimiter=",")
    return predicted, t_data
    """
    for i, p in enumerate(predicted):
        class_index = np.argmax(p)
        truth_index = np.argmax(t_data[i])
        for key, value in Predictor.CLASS_INDICES.items():
            if value == class_index:
                print("Prediction_arr", p)
                print("Predicted", key)
            if value == truth_index:
                print("Truth", key)
    """


def create_model():
    model = Sequential()
    model.add(Embedding(30, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(39, activation="softmax"))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

    return model

def run_model():
    trainX, trainY = store_train_data(limit=2500)
    print("Shape before", trainX.shape)
    #trainX = np.asarray([trainX])
    print("X shape", trainX.shape)
    print("Type", type(trainX))
    m = create_model()
    #trainX = trainX.reshape(1, 303, 39)
    #data = data.reshape(1, 10, 2)
    #trainY = np.asarray(trainY).reshape(1,303, 39)
    #print(data.shape)
    print(trainX.shape)

    m.fit(np.array(trainX), np.array(trainY), epochs=15, verbose=1, shuffle=False)

    m.save('lstm_model.h5')

def find_truth(prediction):
    pred_index = np.argmax(prediction)
    for key, val in Predictor.CLASS_INDICES.items():
        if val == pred_index:
            return key
#run_model()

 
lstm_mod = keras.models.load_model(os.getcwd() + '/lstm_model.h5')
pred_res, truth = store_train_data(limit=5)
actual_truth = [find_truth(pred) for pred in truth]
cnn_truths = [find_truth(pred) for pred in pred_res]
for pred in pred_res:
    print(pred)

p = np.array(pred_res)

for a in p: print(a)
predictions = lstm_mod.predict_proba(p)
lstm_truths = [find_truth(pred) for pred in predictions]

print("Actual truth", actual_truth)
print("CNN truths", cnn_truths)
print("LSTM truths", lstm_truths)
