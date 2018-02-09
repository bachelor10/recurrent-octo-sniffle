from tornado import websocket, web, ioloop
from predictor import Predictor
import os, uuid, json, base64_converter
from client import Client
from client_controller import ClientController
from time_intersect import time_segmenter

#predictor = Predictor()
cc = ClientController()
ts = time_segmenter()


class WebSocket(websocket.WebSocketHandler):
    def open(self):
        print('Client connected to websocket!')

    def on_message(self, message):
        parsed_message = json.loads(message)
        # print("on_message: ", parsed_message)

        # Find client
        client = cc.find_client(parsed_message['uuid'])
        client.add_time = True

        # Remove uuid from json
        parsed_message.pop('uuid', None)

        # Add data to buffer
        # if 'status' in parsed_message and 'traceid' in parsed_message:
        #    if parsed_message['status'] == 'End':
        #        client.buffer.append(message)  # this should make the buffer of a client a list og lists.
        #        print(client.buffer)

        # this passes data in inkml format to to_buffer method.
        # param is parsed json
        if client:
            if 'status' in parsed_message:
                if parsed_message['status'] == 201:  # http created = 201
                    # pass to inkml creation
                    print("Running prediction")
                    #prediction = predictor.predict(client.buffer)

                    #print("Predicted:", prediction)

                    ts.find_time_between(buffer=client.buffer)
                    # print("Predicted", prediction[0], "as", prediction[1])
                    self.write_message("Predicted: " + str("hest"))

                    client.buffer = []

            # elif 'status' in parsed_message:
            else:
                client.to_buffer(parsed_message)

                # Set client data if not set
        if not client.data:
            client.data = self

    def on_close(self):
        print('Client disconnected with: ', self)

        # Find client
        # client = find_client_with_request_data(self) # this can be none

        # Delete client buffer
        # if client is not None:
        #   if client.buffer is not None:
        #        del client.buffer

        # Remove client from set
        # del clients[client.uuid] #TODO handle possible keyerror


class rest_handler(web.RequestHandler):
    def get(self, *args):
        # Create client object and add to set
        id = uuid.uuid4()
        equation = '2 + 2 = 4'

        client = Client(uuid=str(id), current_equation=equation)
        cc.clients[str(id)] = client

        # Get an equation from db

        # Create json
        data = {
            'equation': equation,
            'uuid': str(id)
        }

        # Send equation and uuid to client
        self.write(json.dumps(data))

    def post(self):
        # Extract UUID
        client_id = self.get_body_argument("uuid")
        print(self.__str__())
        # Find client in set
        client = cc.find_client(client_id)

        # Extract image and save image
        # TODO handle NONETYPE
        # base64_converter.convertToImg(self.get_body_argument("b64_str"), client.current_equation)

        # Send client buffer to db

        # Reset buffer

        # Get an equation from db
        equation = '4 - 1 = 3'
        client.current_equation = equation

        # Send equation to client
        data = {
            'equation': equation
        }

        self.write(json.dumps(data))


class IndexHandler(web.RequestHandler):
    def get(self):
        self.render("./client/index.html")


app = web.Application([
    (r'/', IndexHandler),
    (r'/ws', WebSocket),
    (r'/api', rest_handler),
    (r'/client/(.*)', web.StaticFileHandler, {'path': './client/'})
])

if __name__ == '__main__':
    app.listen(8080)
    ioloop.IOLoop.instance().start()
