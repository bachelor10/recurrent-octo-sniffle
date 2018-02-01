class Client:
    def __init__(self, uuid, current_equation):
        self.buffer = []  # this is going to be a list of lists, but it's initialized in to_buffer method
        self.uuid = uuid
        self.current_equation = current_equation
        self.data = None
        print("New client with uuid: ", uuid)

    def __str__(self):
        print(self.uuid , self.current_equation)

    # msg is a dict
    def to_buffer(self, msg):
        if 'traceid' in msg:
            traceid = msg['traceid']
            if len(self.buffer) - 1 < traceid:
                self.buffer.append([])
                """if 'x1' in msg and 'y1' in msg:  # Make sure to only add pairs of coordinates.
                    self.buffer[traceid].append(int(msg['x1']))
                    self.buffer[traceid].append(int(msg['y1']))
                if 'x2' in msg and 'y2' in msg:
                    self.buffer[traceid].append(int(msg['x2']))
                    self.buffer[traceid].append(int(msg['y2']))"""

            if 'x1' in msg and 'y1' in msg:  # Make sure to only add pairs of coordinates.
                self.buffer[traceid].append([int(msg['x1']), int(msg['y1'])])
            if 'x2' in msg and 'y2' in msg:
                self.buffer[traceid].append([int(msg['x2']), int(msg['y2'])])

            else:
                print("Found no traceid.")

    # now the buffers are filled, with each trace as a list.
    def to_inkml(self):
        raise NotImplementedError()

    def parse_overlapping(self):
        raise NotImplementedError()
