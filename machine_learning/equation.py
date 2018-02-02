import os, math, sys, random
from PIL import Image, ImageDraw
from xml_parse import dirs


class Equation:
    IMG_HEIGHT = 40
    IMG_WIDTH = 120

    def __init__(self, segments):
        self.segments = segments
        self.glob_max_x = 0
        self.glob_min_x = math.inf
        self.glob_max_y = 0
        self.glob_min_y = math.inf

    def save(self):
        for segment in self.segments:
            x, y, h, w = segment.compute_bounding_box()

            image, truth, segment_id = segment.generate_bitmap()

            if image is None: continue

            directory = os.getcwd() + "/" + dirs[1]

            if random.random() > 0.8:
                directory = os.getcwd() + "/" + dirs[0]

            if truth.isalpha():
                if truth.isupper():
                    truth = truth + "_"

            if truth == '|':
                truth = '_|'

            subdir = directory + "/" + truth

            filename = subdir + '/' + truth + "_" + segment_id + ".bmp"

            if not os.path.exists(subdir):
                os.makedirs(subdir)

            image.save(filename)

    def compute_global_boundaries(self):

        for segment in self.segments:
            max_x, max_y, min_x, min_y = segment.calculate_bounding_box()

            self.glob_max_x = max(self.glob_max_x, max_x)

            self.glob_max_y = max(self.glob_max_y, max_y)
            self.glob_min_x = min(self.glob_min_x, min_x)
            self.glob_min_y = min(self.glob_min_y, min_y)
            # print("GLOBS!!!", self.glob_max_x, self.glob_max_y, self.glob_min_x, self.glob_min_y, "\n")

    def create_image_and_scale(self):

        image = Image.new('L', (Equation.IMG_WIDTH, Equation.IMG_HEIGHT), "white")

        draw = ImageDraw.Draw(image)

        for segment in self.segments:
            segment.calculate_bounding_box()

            segment.normalize(Equation.IMG_HEIGHT, Equation.IMG_WIDTH, self.glob_max_x, self.glob_max_y,
                              self.glob_min_x, self.glob_min_y)

            segment.draw_symbol(draw)

        image.save('eksempel.bmp')