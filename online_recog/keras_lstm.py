

import keras
from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers import Conv1D, Conv2D, GRU, Concatenate, Bidirectional, MaxPooling1D, MaxPooling2D, Merge, BatchNormalization, Activation, Input, TimeDistributed, Dense, Flatten, Dropout, LSTM, Embedding
from io import BytesIO
import os
from matplotlib import pyplot as plt
from preprocessing import generate_dataset, get_single_segment
import numpy as np
from sklearn.utils import shuffle


plt.style.use('ggplot')

#https://github.com/keras-team/keras/issues/2548
"""
    A callback to run validation on more than test and validation data (real data). 
    Stores the results from all epochs in a log directory.
"""
class StorageCallback(Callback):
    #Uncomment lines if you are running with real dataset aswell
    def __init__(self, real_dataX=None, real_dataY=None, name=""):
        #self.real_dataX = real_dataX
        #self.real_dataY = real_dataY

        #self.real_data_loss = []
        #self.real_data_acc = []

        self.validation_data_loss = []
        self.validation_data_acc = []
        
        self.train_data_loss = []
        self.train_data_acc = []

        self.filename = name

    def on_epoch_end(self, epoch, logs={}):
        #x, y = self.real_dataX, self.real_dataY
        #loss, acc = self.model.evaluate(x, y, verbose=0)
        #print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

        #self.real_data_loss.append(loss)
        #self.real_data_acc.append(acc)

        self.validation_data_loss.append(logs['val_loss'])
        self.validation_data_acc.append(logs['val_acc'])

        self.train_data_loss.append(logs['loss'])
        self.train_data_acc.append(logs['acc'])


        #p.save('./logs/' + self.filename + "_real_data_loss", np.array(self.real_data_loss))
        #np.save('./logs/' + self.filename + "_real_data_acc", np.array(self.real_data_acc))
        np.save('./logs/' + self.filename + "_validation_data_loss", np.array(self.validation_data_loss))
        np.save('./logs/' + self.filename + "_validation_data_acc", np.array(self.validation_data_acc))
        np.save('./logs/' + self.filename + "_train_data_loss", np.array(self.train_data_loss))
        np.save('./logs/' + self.filename + "_train_data_acc", np.array(self.train_data_acc))


        

# The classes to index mapping, used to determine what truth the predicted class maps to.
CLASS_INDICES = {'3': 7, 'y': 36, 'lt': 26,'\lt': 26, 'gamma': 22, '\\gamma': 22, 'beta': 20, '\\beta': 20, ')': 1, '0': 4, '1': 5, 'sqrt': 33, '\sqrt': 33, 'lambda': 25, '\\lambda': 25, '7': 11, 'z': 37, '6': 10, 'Delta': 15,'\\Delta': 15, '-': 3, 'neq': 28,'\\neq': 28, '=': 14, '8': 12, 'G': 16, 'sigma': 32,'\\sigma': 32, 'f': 21, 'rightarrow': 31,'\\rightarrow': 31, 'phi': 29,'\phi': 29, 'infty': 24,'\infty': 24, 'x': 35, '[': 17, '9': 13, 'gt': 23, '\gt': 23, 'theta': 34,'\\theta': 34, 'pi': 30, '\pi': 30, '4': 8, '5': 9, '2': 6, 'mu': 27, '\mu': 27, '(': 0, ']': 18, 'alpha': 19, '\\alpha': 19, '+': 2}

# Helper function to generate dataset and do the last preprocessing (truth to onehot and image to channel_last and divided by 255)
def generate_train_data(limit=10000):
    prediction_data = []
    pred_images = []
    truth_data = []

    segments, images, truths, original_traces = generate_dataset(limit, include=CLASS_INDICES, num_augumentations=2)
    count = 0
    for segment, image, truth in zip(segments, images, truths):

        prediction_data.append(segment)
        pred_images.append(np.reshape(np.array(image)/255, (26, 26, 1)))
        one_hot = np.zeros(38)
        one_hot[CLASS_INDICES[truth]] = 1
        truth_data.append(one_hot)
        
        """f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(np.array(image).reshape(26, 26))
        ax2.plot(segment[:, 0], segment[:, 1], '-o')
        ax2.invert_yaxis()
        ax2.set_xlim([-1, 1])
        ax2.set_ylim([1, -1])

        plt.show()"""


        count += 1
    
    return [np.array(prediction_data), np.array(pred_images)], np.array(truth_data), np.array(original_traces)

# Helper method for testing a model
def predict_classes(imgs):
    MODEL_PATH = os.getcwd() + '/my_model.h5' #Set the name of your model

    model = keras.models.load_model(MODEL_PATH)

    return model.predict_proba(imgs)



# The resulting RNN model after experimentation
def create_RNN_model(with_last_layer=False):
    model = Sequential()
    model.add(Conv1D(48, 5, activation="relu", padding="valid", input_shape=(40, 3)))
    model.add(MaxPooling1D())
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Conv1D(64, 5, activation="relu", padding="valid"))
    model.add(MaxPooling1D())
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Conv1D(96, 5, activation="relu", padding="valid"))
    model.add(MaxPooling1D())
    model.add(BatchNormalization())
    model.add(Dropout(0.3))


    model.add(Bidirectional(GRU(128, return_sequences=True, activation="relu", dropout=0.2, recurrent_dropout=0.2)))
    model.add(Bidirectional(GRU(128, activation="relu", recurrent_dropout=0.2, dropout=0.2)))
    if with_last_layer:
        model.add(Dense(38, activation="softmax"))
    
    return model


# The resulting CNN model after experimentation
def create_CNN_model(with_last_layer=False):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(26,26,1), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.25))

    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(512, activation='tanh'))
    model.add(Dropout(0.5))
    if with_last_layer:
        model.add(Dense(38, activation="softmax"))
    return model

#The resulting combined model from both a CNN and RNN
def create_combined_model():
    RNN_model = create_RNN_model()
    CNN_model = create_CNN_model()

    concatenated = Concatenate()([RNN_model.output, CNN_model.output])
    model = (Dense(128, activation="relu"))(concatenated)
    model = (Dropout(0.3))(model)
    model = (Dense(38, activation = 'softmax'))(model)

    return keras.models.Model([RNN_model.input, CNN_model.input], model)

# Combile the model with adam opimizer and categorical crossentropy
def compile_model(model):

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer="adam",
                metrics=['accuracy'])

# Do training
def run_model(trainX, trainY, realX, realY, model, name="", num_epochs=20):
    
    callbacks = []
    if realX and realY:
        callbacks.append(StorageCallback(realX, realY, name)) # Only use callback if you have access to real data

    history = model.fit(trainX, trainY, epochs=num_epochs, verbose=1, shuffle=True, validation_split=0.1, 
        callbacks=callbacks)


    return history

# Helper function to find a truth from a prediction
def find_truth(prediction):
    pred_index = np.argmax(prediction)
    for key, val in CLASS_INDICES.items():
        if val == pred_index:
            return key

# Run the different models
def run_combined_model(trainX, trainY, realX, realY):

    m = create_combined_model()
    compile_model(m)
    run_model(trainX, trainY, realX, realY, m, "combined_model")
    m.save('combined_model.h5')

def run_RNN_model(trainX, trainY, realX, realY):

    m = create_RNN_model(with_last_layer=True)
    compile_model(m)

    run_model(trainX[0], trainY, realX[0], realY, m, "RNN_model")
    m.save('RNN_model.h5')

def run_CNN_model(trainX, trainY, realX, realY):

    m = create_CNN_model(with_last_layer=True)
    compile_model(m)
    run_model(trainX[1], trainY, realX[1], realY, m, "CNN_model")
    m.save('CNN_model.h5')


# Generate a new dataset
def generate_and_save_dataset():

    trainX, trainY, original_traces = generate_train_data(50000)

    np.save('./data/trainX_trace', trainX[0])
    np.save('./data/trainX_img', trainX[1])
    np.save('./data/trainY', trainY)
    np.save('./data/original_traces', original_traces)

def load_dataset_and_run_model():
    # Load data and train model
    trainX_trace = np.load('./data/trainX_trace.npy')
    trainX_img = np.load('./data/trainX_img.npy')
    trainY = np.load('./data/trainY.npy')
    
    #Remove if you have access to real data
    realY = [None]
    realX = [None, None]
    
    #Uncomment if you have access to real data
    #realX = [np.load('./data/real_test_data/trainX_trace.npy'), np.load('./data/real_test_data/trainX_img.npy')]
    #realY = np.load('./data/real_test_data/trainY.npy')

    # Real data needs a bit of preprocessing.
    #realX[0] = realX[0].reshape(len(realX[0]), 40, 3)
    #realX[1] = realX[1].reshape(len(realX[0]), 26, 26, 1)

    # Shuffle data (gave same results, therefore commented out)
    #trainX_trace, trainX_img, trainY = shuffle(trainX_trace, trainX_img, trainY, random_state=0)

    # Run the specified model
    run_RNN_model([trainX_trace, trainX_img], trainY, realX, realY)
   
#Uncomment to create a new dataset from InkML
#generate_and_save_dataset()

#Uncomment to run training on either self generated or downloaded training data.
#load_dataset_and_run_model()

