import keras
import os

model_path = os.getcwd() + '/machine_learning/my_model.h5'

class Predictor:
    def __init__(self):
        self.model = keras.models.load_model(model_path)

    def predict(self, ):