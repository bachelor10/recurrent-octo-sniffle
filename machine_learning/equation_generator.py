from PIL import Image
import os, random
from xml_parse import full_truths_generator
class EquationGenerator:

    def __init__(self):
        self.directory = os.getcwd() + '/train/'

    def get_symbol(self, symbol):

        symbol_directory = self.directory + symbol

        file_name = random.choice(os.listdir(symbol_directory))

        return Image.load(file_name)


for equation in full_truths_generator():
    pass#print(truth)
