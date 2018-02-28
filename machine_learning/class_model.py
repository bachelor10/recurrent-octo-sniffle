import intersect
import math, uuid, os
import numpy as np
import keras
from itertools import cycle, combinations
from PIL import Image, ImageDraw

from time import time

class Boundingbox:
    def __init__(self, traces):
        
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
            
        self.mid_x = (max_x + min_x)/2
        self.mid_y = (max_y + min_y)/2
        self.max_x = max_x
        self.max_y = max_y
        self.min_x = min_x
        self.min_y = min_y
        self.width = max_x - min_x
        self.height = max_y - min_y


class Group:
    def __init__(self, mid_x):
        self.mid_x = mid_x


class Regular(Group):
    def __init__(self, id, mid_x):
        Group.__init__(self, mid_x)
        self.id = id
    
    @staticmethod
    def asLatex(truth):
        if truth == 'sqrt' or truth == 'alpha' or truth == 'beta' or truth == 'Delta' or truth == 'gamma' or truth == 'infty' or truth == 'lambda' or truth == 'pi' or truth == 'mu' or truth == 'phi' or truth == 'sigma' or truth == 'sum' or truth == 'times' or truth == 'rightarrow':
            return '\\' + truth
        elif truth == 'gt':
            return '>'
        elif truth == 'lt':
            return '<'
        
        else:
            return truth


class Fraction(Group):
    def __init__(self, numerator, denominator, mid_x):
        Group.__init__(self, mid_x)
        self.numerator = numerator
        self.denominator = denominator


class Power:
    def __init__(self, base=None, exponent=None):
        self.base = base
        self.exponent = exponent

    def is_power(base, exponent):
        pass


class Trace:
    def __init__(self, points):
        self.points = points
        self.boundingbox = Boundingbox(points)

    def add_points(amount):
        pass

    def remove_points(amount):
        pass

    def check_overlap(trace):
        pass


class Segment:
    def __init__(self, traces, id, truth=''):
        self.traces = traces
        self.boundingbox = Boundingbox(traces)
        self.id = id
        self.truth = truth

    def print_info(self):
        print("\nSegment info for", self.id)
        print("Truth:", self.truth)
        print("Amount of traces:", len(self.traces))
        print("Length of traces", [len(t) for t in self.traces])
        b = self.boundingbox
        print("Boundingbox (x, y, w, h):", b.mid_x, b.mid_y, b.width, b.height)
        print("max_x, min_x, max_y, min_y", b.max_x, b.min_x, b.max_y, b.min_y)


    def add_trace(trace):
        # might be useful for live feedback
        pass


class Segmentgroup:
    def __init__(self, segments):
        self.segments = segments

    
class Expression:
    def __init__(self, predictor):
        self.groups = []
        self.segments = dict()
        self.predictor = predictor
        self.processed = []

    def create_tracegroups(self, traces, trace_pairs):
        tracegroups = []
        
        for i, trace in enumerate(traces):

            flag = False
            for j, group in enumerate(tracegroups):

                common = []
                for p in trace_pairs:
                    if i in p:
                        common = common + list(p)
                common = list(set(common))

                if len(set(common).intersection(group)) > 0:
                     tracegroups[j] = list(set(common + group))
                     flag = True

            if not flag:
                new_group = [i]
                for pair in trace_pairs:
                    if i in pair:
                        new_group = new_group + list(pair)
                
                new_group = list(set(new_group))
                tracegroups.append(new_group)
            
        sorted_tracegroups = sorted(tracegroups, key=lambda m:next(iter(m)))

        return sorted_tracegroups

    def find_overlap_pairs(self, traces):
        overlap_pairs = set()

        for i, trace in enumerate(traces[:-1]):
            for j, trace2 in enumerate(traces[i+1:]):
                for coord1 in trace:
                    for coord2 in trace2:
                        if math.hypot(coord2[0] - coord1[0], coord2[1] - coord1[1]) < 10:
                            overlap_pairs.add((i, i+j+1))

                # Check lines between endpoints
                overlap = intersect.intersect(trace[0], trace[-1], trace2[0], trace2[-1])
                if(overlap):
                    overlap_pairs.add((i, i+j+1))
        
        return overlap_pairs


    def feed_traces(self, traces):
        overlap_pairs = self.find_overlap_pairs(traces)
        tracegroups = self.create_tracegroups(traces, overlap_pairs)
        self.create_segments(traces, tracegroups)


    def create_segments(self, traces, tracegroups):
        for i, group in enumerate(tracegroups):
            traces_for_segment = [traces[j] for j in list(group)]
            id = str(i)
            segment = Segment(traces_for_segment, id)
            self.segments[id] = segment


    def join_segments(self, id1, id2, truth=''):
        segment1 = self.segments.pop(id1, None)
        segment2 = self.segments.pop(id2, None)

        traces = segment1.traces + segment2.traces
        id = segment1.id

        new_segment = Segment(traces, id, truth)
        self.segments[id] = new_segment
        
    
    def find_segments_in_area(self, max_x, min_x, max_y, min_y, ignore=[]):
        # Searches through segments and look for middle points inside area
        segments_in_area = []

        for id, segment in self.segments.items():
            if min_x <= segment.boundingbox.mid_x <= max_x and min_y <= segment.boundingbox.mid_y <= max_y:
                if segment.id not in ignore:
                    segments_in_area.append(segment.id)

        return segments_in_area


    def search_horizontal(self):
        
        pass


    def recursive_search_for_id(self, id, group):
        
        if type(group) == Fraction:
            for g in group.numerator:
                check = self.recursive_search_for_id(id, g)
                if check:
                    return True
                
            for g in group.denominator:
                check = self.recursive_search_for_id(id, g)
                if check:
                    return True

        elif type(group) == Regular:
            if group.id == id:
                return True
            else:
                return False


    def sort_id_list_x(self, ids):
        return [seg.id for seg in sorted([self.segments[id] for id in ids], key=lambda x: x.boundingbox.mid_x, reverse=False)]
    

    def sort_ids_by_width(self, ids):
        return [seg.id for seg in sorted([self.segments[id] for id in ids], key=lambda x: x.boundingbox.width, reverse=True)]


    def sort_groups(self):
        self.groups.sort(key=lambda group: group.mid_x, reverse=False)
    

    def sort_groups_by_width(self, groups):
        groups.sort(key=lambda group: group.mid_x, reverse=True)


    def is_fraction(self, id, max_y, min_y):
        coords = self.segments[id].boundingbox

        ignore = self.processed + [id] 
        
        over = self.find_segments_in_area(coords.max_x+20, coords.min_x-20, coords.mid_y, min_y, ignore)
        under = self.find_segments_in_area(coords.max_x+20, coords.min_x-20, max_y, coords.mid_y, ignore)

        return len(over) > 0 and len(under) > 0, over, under


    def fraction_search(self, id, ids, max_y, min_y):
        is_frac, over, under = self.is_fraction(id, max_y, min_y)
        
        if is_frac:

            over = self.sort_id_list_x(over)
            under = self.sort_id_list_x(under)

            over_objects_reg = []
            over_objects_frac = []
            under_objects_reg = []
            under_objects_frac = []

            for over_id in over:
                if over_id in ids:
                    obj = self.fraction_search(over_id, ids, self.segments[id].boundingbox.mid_y - 1, self.segments[over_id].boundingbox.mid_y - 200)
                    
                    if type(obj) == Fraction:
                        over_objects_frac.append(obj)
                    elif type(obj) == Regular:
                        over_objects_reg.append(obj)

            # Check if any segments is doubled up
            # Only a case for
            for r in over_objects_reg:
                check = False
                for f in over_objects_frac:
                    #Check if r can be found in any fractions
                    if self.recursive_search_for_id(r.id, f):
                        check = True

                if check:
                    # Remove r from over_object_reg

                    over_objects_reg.remove(r)

                
            for over_id in over:
                if over_id not in self.processed:
                    mid_r = self.segments[over_id].boundingbox.mid_x
                    reg = Regular(over_id, mid_r)
                    over_objects_reg.append(reg)

            for under_id in under:
                if under_id in ids:
                    obj = self.fraction_search(under_id, ids, self.segments[under_id].boundingbox.mid_y + 200, self.segments[id].boundingbox.mid_y + 1)
                    if type(obj) == Fraction:
                        under_objects_frac.append(obj)
                    elif type(obj) == Regular:
                        under_objects_reg.append(obj)
            

            for r in under_objects_reg:
                check = False
                for f in under_objects_frac:
                    #Check if r can be found in any fractions
                    if self.recursive_search_for_id(r.id, f):
                        check = True
            
                if check:
                    # Remove r from over_object_reg
                    over_objects_reg.remove(r)

            for under_id in under:
                if under_id not in self.processed:
                    mid_r = self.segments[under_id].boundingbox.mid_x
                    reg = Regular(under_id, mid_r)
                    under_objects_reg.append(reg)

            self.processed = self.processed + over + under + [id]
            self.segments[id].truth = 'frac'


            over_objects = over_objects_reg + over_objects_frac
            under_objects = under_objects_reg + under_objects_frac

            over_objects.sort(key=lambda group: group.mid_x, reverse=False)
            under_objects.sort(key=lambda group: group.mid_x, reverse=False)

            # return a Fraction
            mid = self.segments[id].boundingbox.mid_x
            fraction = Fraction(over_objects, under_objects, mid)


            print('Found fraction, info:', id, [i.id for i in fraction.numerator if type(i) == Regular], [i.id for i in fraction.denominator if type(i) == Regular])

            print('Processed:', self.processed)

            return fraction
            
        else:
            # return a Regular 
            self.processed.append(id)
            regular = Regular(id, self.segments[id].boundingbox.mid_x)
            print('Found regular, info:', id)
            print('Processed:', self.processed)
            return regular

    
    def find_fractions(self, ids):
        for id in ids:
            if id not in self.processed:
                max_y = self.segments[id].boundingbox.mid_y + 200
                min_y = self.segments[id].boundingbox.mid_y - 200
                is_frac, over, under = self.is_fraction(id, max_y, min_y)

                if is_frac:
                    group = self.fraction_search(id, ids, self.segments[id].boundingbox.mid_y + 200, self.segments[id].boundingbox.mid_y - 200)
                    self.groups.append(group)


    '''
    def find_fractions(self, ids):
        new_ids = []

        for minus_id in ids:
            is_frac, over, under = self.is_fraction(minus_id)

            if is_frac:
                # Create new fraction
                over = self.sort_id_list_x(over)
                under = self.sort_id_list_x(under)
                self.segments[minus_id].truth = 'frac'

                mid_x = self.segments[minus_id].boundingbox.mid_x

                fraction = Fraction(over, under, mid_x)
                self.groups.append(fraction)

                # Set minus_id, over and under to processed
                self.processed.append(minus_id)
                self.processed = self.processed + over + under

            else:
                new_ids.append(minus_id)

        return new_ids
    '''

    def is_equalsign(self, id1, id2):
        try:
            coords1 = self.segments[id1].boundingbox
            coords2 = self.segments[id2].boundingbox
        except KeyError:
            return False

        return np.abs(coords1.mid_x - coords2.mid_x) < 50


    def find_equalsigns(self, ids):
        still_equalsigns = True
        while still_equalsigns:
            for pair in combinations(ids, r=2):
                if self.is_equalsign(pair[0], pair[1]):
                    self.join_segments(pair[0], pair[1], truth='=')
                    del ids[ids.index(pair[0])]
                    del ids[ids.index(pair[1])]
                    break
            else:
                still_equalsigns = False


    def classify_segments(self):

        minus_ids = []
        for id, segment in self.segments.items():
            segment.truth = self.predictor.predict(segment.traces)

            if segment.truth == '-':
                minus_ids.append(segment.id)
        
        # Check if minus signs is fractions

        sorted_minus_ids = self.sort_ids_by_width(minus_ids)
        self.find_fractions(sorted_minus_ids)


        updated_ids = [i for i in minus_ids if i not in self.processed]


        # Check if minus signs is equalsigns
        if len(updated_ids) > 1:
            self.find_equalsigns(updated_ids)

        #for id, segment in self.segments.items():
        #    segment.print_info()
        for id, segment in self.segments.items():
            if id not in self.processed:

                mid_x = self.segments[id].boundingbox.mid_x

                self.groups.append(Regular(id, mid_x))
                self.processed.append(id)
        
        # Sort groups
        self.sort_groups()


    def create_segmentgroups(self):
        pass


    def  get_latex_pwr(self, power):
        pass


    def get_latex_frac(self, frac):

        num_latex = ''
        den_latex = ''

        for group in frac.numerator:
            if type(group) is Fraction:
                num_latex += self.get_latex_frac(group)
            elif type(group) is Regular:
                num_latex += self.segments[group.id].truth
        
        for group in frac.denominator:
            if type(group) is Fraction:
                den_latex += self.get_latex_frac(group)
            elif type(group) is Regular:
                den_latex += self.segments[group.id].truth

        return '\\frac{' + num_latex + '}{' + den_latex + '}'


    def get_latex(self):
        latex = ''
        for group in self.groups:
            if type(group) is Fraction:
                latex += self.get_latex_frac(group)
            elif type(group) is Regular:
                latex += self.segments[group.id].truth
                #latex += self.segments[group.id].truth
        
        print(latex)
        return latex

    def get_truth(self):

        pass


class Predictor:
    MODEL_PATH = os.getcwd() + '/machine_learning/my_model.h5'
    #MODEL_PATH = os.getcwd() + '/machine_learning/new_model.h5'
    #print(os.listdir(os.getcwd() + '/machine_learning' + '/train'))
    CLASSES = ['+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=']
    #CLASSES = os.listdir(os.getcwd() + '/machine_learning' + '/train')    
    #CLASSES = ["+", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "="]#os.listdir(os.getcwd() + '/machine_learning' + '/train')    

    #MODEL_PATH = os.getcwd() + '/my_model.h5'
    CLASSES = os.listdir(os.getcwd() + '/machine_learning/train2')
    CLASS_INDICES = {']': 17, 'z': 38, 'int': 23, 'sqrt': 32, '3': 7, '\\infty': 22, 'neq': 27, '6': 10, '0': 4, '[': 16, '7': 11, '4': 8, '(': 0, 'x': 36, '\\alpha': 18, '\\lambda': 24, '\\beta': 19, '\\rightarrow': 30, '8': 12, ')': 1, '=': 14, 'y': 37, '\\phi': 28, '\\times': 35, '1': 5, '<': 25, '\\Delta': 15, '\\gamma': 20, '9': 13, '\\pi': 29, '2': 6, '\\sum': 33, '\\theta': 34, '\\mu': 26, '-': 3, '>': 21, '+': 2, '\\sigma': 31, '5': 9}

    #{'gamma': 20, 'pi': 29, 'sum': 33, 'int': 23, 'theta': 34, '9': 13, 'lt': 25, '4': 8, 'times': 35, '5': 9, '(': 0, 'infty': 22, 'rightarrow': 30, 'neq': 27, 'gt': 21, '+': 2, '2': 6, '-': 3, '7': 11, 'sqrt': 32, ')': 1, '8': 12, 'beta': 19, 'y': 37, 'z': 38, '[': 16, '6': 10, 'x': 36, '=': 14, 'alpha': 18, 'mu': 26, 'sigma': 31, '0': 4, ']': 17, '3': 7, '1': 5, 'lambda': 24, 'Delta': 15, 'phi': 28}


    def __init__(self):
        self.model = keras.models.load_model(Predictor.MODEL_PATH)

    def predict(self, segment_traces):
        start = time()
        processed = self.pre_process(segment_traces)
        print("Preprocess time", str(time() - start) + "ms")
        start = time()
        output = self.model.predict_proba(processed)
        print("Predicted", output)
        print("Predict Time", str(time() - start) + "ms")
        
        proba_index = np.argmax(output[0])
        for key, value in Predictor.CLASS_INDICES.items():
            if value == proba_index:
                return key
        """
        for i, p in enumerate(output[0]):

            if p > best_pred[1]:
                best_pred = (i, p)
                Predictor.CLASS_INDICES
                prediction = Predictor.CLASSES[i]
        """
        #return prediction
        
    #https://gist.github.com/perrygeo/4512375
    def scale_linear_bycolumn(self, rawpoints, high=24, low=0, ma=0, mi=0):
        mins = mi
        maxs = ma

        rng = maxs - mins

        output = high - (((high - low) * (maxs - rawpoints)) / rng)

        return output

    def pre_process(self, traces):
        resolution = 45
        image_resolution = 45

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
                new_y = self.scale_linear_bycolumn(y, high=resolution, low=0, ma=max_y, mi=min_y)
                side = (resolution - width_scale) / 2
                new_x = self.scale_linear_bycolumn(x, high=(resolution - side), low=(side), ma=max_x, mi=min_x)
            else:
                # add padding in y-direction
                new_x = self.scale_linear_bycolumn(x, high=resolution, low=0, ma=max_x, mi=min_x)  # , maximum=(max_x, max_y), minimum=(min_x, min_y))
                side = (resolution - height_scale) / 2
                new_y = self.scale_linear_bycolumn(y, high=(resolution - side), low=(side), ma=max_y, mi=min_y)  # , maximum=(max_x, max_y), minimum=(min_x, min_y))

            coordinates = list(zip(new_x, new_y))
            xy_cycle = cycle(coordinates)

            next(xy_cycle)

            for x_coord, y_coord in coordinates[:-1]:
                next_coord = next(xy_cycle)
                draw.line([x_coord, y_coord, next_coord[0], next_coord[1]], fill="black", width=1)


        i = image.convert('LA')
        i.thumbnail((26, 26))

        #i.show()

        arr = np.asarray(i)

        formatted = []
        for row in arr:
            new_row = []
            for col in row:
                new_row.append(col[0])

            formatted.append(new_row)

        return np.asarray([np.asarray(formatted).reshape((26, 26, 1))])

if __name__ == '__main__':

    traces = [0,1,2,3,4,5,6,7]
    overlap = [(1,2),(2,3),(4,5),(0,7)]

    exp = Expression()

    b = exp.create_tracegroups(traces, overlap)

    print(b)
    