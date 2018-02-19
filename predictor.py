import keras
import math
import numpy as np
from PIL import Image, ImageDraw
from itertools import cycle
from io import BytesIO
import os
from machine_learning import xml_parse
import intersect

classes = [""]#os.listdir(os.getcwd() + '/machine_learning' + '/train')

model_path = os.getcwd() + '/my_model.h5'

class Predictor:
    def __init__(self):
        pass
        #self.model = keras.models.load_model(model_path)

    def create_tracegroups(self, trace_pairs):
        tracegroups = []       
        tracegroups.append(set(list(trace_pairs)[0]))

        for i, pair in enumerate(trace_pairs):
            
            for s in tracegroups:
                if pair[0] in s:
                    s.add(pair[1])
                    break
                else:
                    new_set = set()
                    new_set.add(pair[0])
                    new_set.add(pair[1])
                    tracegroups.append(new_set)
        
        return tracegroups

    def predict2(self, image):
        return self.model.predict(image, steps=1, batch_size=None, verbose=1)


    def predict(self, traces):

        print("Taces", traces)

        # Create tracegroups

        print("Creating tracegroups")

        overlap_pairs = set()

        for i, trace in enumerate(traces[:-1]):
            for j, trace2 in enumerate(traces[i+1:]):
                for coord1 in trace:
                    for coord2 in trace2:
                        if math.hypot(coord2[0] - coord1[0], coord2[1] - coord1[1]) < 10:
                            overlap_pairs.add((i, i+j+1))

                # Check lines between endpoints
                overlap = intersect.intersect(trace[0], trace[-1], trace2[0], trace2[-1])
                print("End to end overlap",overlap)
                if(overlap):
                    overlap_pairs.add((i, i+j+1))

                

        print("overlap_pairs", overlap_pairs)



        if len(overlap_pairs) > 0:
            tracegroups = self.create_tracegroups(overlap_pairs)
        else:
            tracegroups = []

        # Add single traces to a tracegroup
        for i, trace in enumerate(traces):
            found = False
            for group in tracegroups:
                if i in group:
                    found = True
            if not found:
                tracegroups.append(set([i]))
        
        sorted_tracegroups = sorted(tracegroups, key=lambda m:next(iter(m)))

        print(sorted_tracegroups)
        '''
                line1_cycle = cycle(trace)
                next(line1_cycle)

                line2_cycle = cycle(trace2)
                next(line2_cycle)

                for k, coords in enumerate(trace[:-1]):
                    coord_A = coords
                    coord_B = next(line1_cycle)
                    coord_C = trace2[k]
                    coord_D = next(line2_cycle)

                    #print(coord_A, coord_B, coord_C, coord_D)
                    
                    if intersect.intersect(coord_A, coord_B, coord_C, coord_D):
                        #print("i", i)
                        print("Intersect", coord_A, coord_B, coord_C, coord_D)
        '''

        predictions = []

        for group in sorted_tracegroups:
            # lots of copying, TODO optimalize
            res = [traces[i] for i in list(group)]
            res_processed = self.pre_process(res)

            prediction = self.model.predict(res_processed, steps=1, batch_size=None, verbose=1)

            best_pred = (0, 0)

            for i, p in enumerate(prediction[0]):
                print("Predicted: ", classes[i], "as", p)

                if p > best_pred[1]:
                    best_pred = (i, p)
                    predictions.append(best_pred)


        ''' res = self.pre_process(traces)

        prediction = self.model.predict_classes(res, batch_size=1, verbose=1)

        print("Prediction", prediction)

        for i, p in enumerate(prediction[0]):
            print("Predicted: ", classes[i], "as", p)

            if p > best_pred[1]:
                best_pred = (i, p)
         '''

        to_return = []

        for p in predictions:
            to_return.append((classes[p[0]], p[1]))

        return to_return

        #return classes[best_pred[0]], best_pred[1]

    def pre_process(self, traces):
        resolution = 24
        image_resolution = 26

        image = Image.new('L', (image_resolution, image_resolution), "white")
        draw = ImageDraw.Draw(image)

        max_x = 0
        min_x = math.inf
        max_y = 0
        min_y = math.inf


        for trace in traces:
            y = np.array(trace).astype(np.float)
            x, y = y.T

            if max_x < x.max():
                max_x = x.max()

            if max_y < y.max():
                max_y = y.max()

            if min_x > x.min():
                min_x = x.min()

            if min_y > y.min():
                min_y = y.min()

        width = max_x - min_x
        height = max_y - min_y
        scale = width / height

        width_scale = 0
        height_scale = 0

        if scale > 1:
            # width > height
            height_scale = resolution / scale
        else:
            # width < height
            width_scale = resolution * scale

        for trace in traces:

            y = np.array(trace).astype(np.float)

            x, y = y.T

            if width_scale > 0:
                # add padding in x-direction
                new_y = xml_parse.scale_linear_bycolumn(y, high=resolution, low=0, ma=max_y, mi=min_y)
                side = (resolution - width_scale) / 2
                new_x = xml_parse.scale_linear_bycolumn(x, high=(resolution - side), low=(side), ma=max_x, mi=min_x)
            else:
                # add padding in y-direction
                new_x = xml_parse.scale_linear_bycolumn(x, high=resolution, low=0, ma=max_x,
                                              mi=min_x)  # , maximum=(max_x, max_y), minimum=(min_x, min_y))
                side = (resolution - height_scale) / 2
                new_y = xml_parse.scale_linear_bycolumn(y, high=(resolution - side), low=(side), ma=max_y,
                                              mi=min_y)  # , maximum=(max_x, max_y), minimum=(min_x, min_y))

            coordinates = list(zip(new_x, new_y))
            xy_cycle = cycle(coordinates)

            next(xy_cycle)

            for x_coord, y_coord in coordinates[:-1]:
                next_coord = next(xy_cycle)
                draw.line([x_coord, y_coord, next_coord[0], next_coord[1]], fill="black", width=1)


        i = image.convert('LA')

        arr = np.asarray(i)

        formatted = []
        for row in arr:
            new_row = []
            for col in row:
                new_row.append(col[0])

            formatted.append(new_row)
        #print(np.asarray([np.asarray(i)]))
        return np.asarray([np.asarray(formatted).reshape((26, 26, 1))])
