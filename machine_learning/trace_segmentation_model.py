from trace_preprocessing import get_training_pairs
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv1D

# import training data
training_set, validation_set = get_training_pairs()

input_amount = len(training_set[0])

# create model
model = Sequential()
model.add(Dense(512, input_dim=input_amount, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
print(model.input_shape)
print(model.summary())
# compile model

model.compile(optimizer=keras.optimizers.Adadelta(),
              loss=keras.losses.mean_squared_error,
              metrics=['accuracy'])

# fit model



model.fit(training_set, validation_set, epochs=10, verbose=1, shuffle=True)


# save model