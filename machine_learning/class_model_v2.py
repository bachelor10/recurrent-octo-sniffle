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


class Segment:
    def __init__(self, id, truth):#, traces):
        self.id = id
        #self.boundingbox = Boundingbox(traces)
        self.truth = truth

    def to_latex(self):
        return self.truth


class Group:
    def __init__(self, id, mid_x):
        self.id = id
        self.mid_x = mid_x
            
    def to_latex(self):

        latex = ''

        if type(self) == Segmentgroup:

            for obj in self.objects:
                latex += obj.to_latex()

        elif type(self) == Fraction:
            
            latex += '\\frac{'
            
            for obj in self.numerator:
                latex += obj.to_latex()

            latex += '}{'

            for obj in self.denominator:
                latex += obj.to_latex()

            latex += '}'

        elif type(self) == Power:

            for obj in self.base:
                latex += obj.to_latex()
            
            latex += '^{'

            for obj in self.exponent:
                latex += obj.to_latex()
            
            latex += '}'

        elif type(self) == Root:

            latex += '\\sqrt{'

            for obj in self.core:
                latex += obj.to_latex()

            latex += '}'

        return latex


class Segmentgroup(Group):
    def __init__(self, id, mid_x, objects):
        super().__init__(id, mid_x)
        self.objects = objects


    def add_object(self, obj):
        self.objects.append(obj)


    def to_latex(self):
        return super().to_latex()

        
class Fraction(Group):
    def __init__(self, id, mid_x, numerator, denominator):
        super().__init__(id, mid_x)
        self.numerator = numerator
        self.denominator = denominator

    def to_latex(self):
        return super().to_latex()


class Power(Group):
    def __init__(self, id, mid_x, base, exponent):
        super().__init__(id, mid_x)
        self.base = base
        self.exponent = exponent

    def to_latex(self):
        return super().to_latex()


class Root(Group):
    def __init__(self, id, mid_x, core):
        super().__init__(id, mid_x)
        self.core = core

    def to_latex(self):
        return super().to_latex()


class Integral(Group):
    pass


class Expression:
    def __init__(self, id, mid_x):
        self.segments = []
        self.groups = []

    def feed_traces(self):
        pass

    
    def classify_segments(self):
        pass


    def find_contexts(self):
        # Roots/Integral
        # Fractions
        # Power
        # 
        pass


    


    def to_latex(self):
        latex = ''
        for group in self.groups:
            latex += group.to_latex()

        return latex


class Searchagent:

    pass


class Preprocessor:
    pass


class Predictor:
    pass


def main():
    s1 = Segment(0, '1')
    s2 = Segment(1, '+')
    s3 = Segment(2, '2')

    sg1 = Segmentgroup(3, 3.4, [s1, s2, s3])

    f1 = Fraction(4, 2.4, [sg1], [s3])
    f2 = Fraction(4, 2.4, [f1], [s3])
    
    print(f2.to_latex())


    
    pass

if __name__ == '__main__':
    main()