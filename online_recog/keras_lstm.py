

import keras
from keras.models import Sequential
from keras.layers import Conv1D, Conv2D, GRU, Concatenate, Bidirectional, MaxPooling1D, MaxPooling2D, Merge, BatchNormalization, Activation, Input, TimeDistributed, Dense, Flatten, Dropout, LSTM, Embedding
from io import BytesIO
import os
from matplotlib import pyplot as plt
from preprocessing import generate_dataset
import numpy as np

plt.style.use('ggplot')


CLASS_INDICES = {'3': 7, 'y': 36, 'lt': 26,'\lt': 26, 'gamma': 22, '\\gamma': 22, 'beta': 20, '\\beta': 20, ')': 1, '0': 4, '1': 5, 'sqrt': 33, '\sqrt': 33, 'lambda': 25, '\\lambda': 25, '7': 11, 'z': 37, '6': 10, 'Delta': 15,'\\Delta': 15, '-': 3, 'neq': 28,'\\neq': 28, '=': 14, '8': 12, 'G': 16, 'sigma': 32,'\\sigma': 32, 'f': 21, 'rightarrow': 31,'\\rightarrow': 31, 'phi': 29,'\phi': 29, 'infty': 24,'\infty': 24, 'x': 35, '[': 17, '9': 13, 'gt': 23, '\gt': 23, 'theta': 34,'\\theta': 34, 'pi': 30, '\pi': 30, '4': 8, '5': 9, '2': 6, 'mu': 27, '\mu': 27, '(': 0, ']': 18, 'alpha': 19, '\\alpha': 19, '+': 2}

def generate_train_data(limit=10000):
    prediction_data = []
    pred_images = []
    truth_data = []

    segments, images, truths = generate_dataset(limit, include=CLASS_INDICES)
    count = 0
    for segment, image, truth in zip(segments, images, truths):

        prediction_data.append(segment)
        pred_images.append(np.reshape(np.array(image)/255, (26, 26, 1)))
        one_hot = np.zeros(38)
        one_hot[CLASS_INDICES[truth]] = 1
        truth_data.append(one_hot)

        count += 1
    
    return [np.array(prediction_data), np.array(pred_images)], np.array(truth_data)

def predict_classes(imgs):
    MODEL_PATH = os.getcwd() + '/my_model.h5'
    #CLASSES = ["+", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "="]#os.listdir(os.getcwd() + '/machine_learning' + '/train')    
    #MODEL_PATH = os.getcwd() + '/my_model.h5'

    model = keras.models.load_model(MODEL_PATH)

    return model.predict_proba(imgs)



# Original LSTM model
def create_LSTM_model():
    model = Sequential()
    model.add(Conv1D(48, 5, activation="relu", padding="valid", input_shape=(40, 3)))
    model.add(MaxPooling1D())
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Conv1D(48, 5, activation="relu", padding="valid"))
    model.add(MaxPooling1D())
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Bidirectional(GRU(128, return_sequences=True, activation="relu")))
    model.add(Dropout(0.2))
    model.add(Bidirectional(GRU(128, activation="relu")))
    model.add(Dropout(0.2))
    #model.add(Dense(38, activation="softmax"))
    return model

   #model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])


# Example CNN model
def create_CNN_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(26,26,1), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.25))

    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(512, activation='tanh'))
    model.add(Dropout(0.5))
    #model.add(Dense(38, activation="softmax"))
    return model

#Compined CNN and LSTM
def create_combined_model():
    LSTM_model = create_LSTM_model()
    CNN_model = create_CNN_model()

    concatenated = Concatenate()([LSTM_model.output, CNN_model.output])
    model = (Dense(128, activation="relu"))(concatenated)
    model = (Dropout(0.3))(model)
    model = (Dense(38, activation = 'softmax'))(model)

    return keras.models.Model([LSTM_model.input, CNN_model.input], model)

def compile_model(model):

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer="adam",
                metrics=['accuracy'])

def run_model(trainX, trainY):

    m = create_combined_model()
    compile_model(m)
    print(m.summary())

    #trainX = trainX.reshape(1, 303, 39)
    #data = data.reshape(1, 10, 2)
    #trainY = np.asarray(trainY).reshape(1,303, 39)
    #print(data.shape)
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True, write_images=True)

    history = m.fit(trainX, trainY, epochs=10, verbose=1, shuffle=True, validation_split=0.1, callbacks=[tensorboard])

    m.save('lstm_model.h5')

    return history

def find_truth(prediction):
    pred_index = np.argmax(prediction)
    for key, val in CLASS_INDICES.items():
        if val == pred_index:
            return key



#trainX, trainY = generate_train_data(50000)


#np.save('./data/trainX_trace', trainX[0])
#np.save('./data/trainX_img', trainX[1])
#np.save('./data/trainY', trainY)

trainX_trace = np.load('./data/trainX_trace.npy')
trainX_img = np.load('./data/trainX_img.npy')
trainY = np.load('./data/trainY.npy')

    
#print(trainY[0])

history = run_model([trainX_trace, trainX_img], trainY)
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
"""
lstm_mod = keras.models.load_model(os.getcwd() + '/lstm_model.h5')
trainX, trainY = generate_train_data(100)
predictions = lstm_mod.predict_proba(trainX)
actual_truth = [find_truth(pred) for pred in trainY]
cnn_truths = [find_truth(pred) for pred in predictions]

print("Actual truth", actual_truth)
print("CNN truths", cnn_truths)

"""