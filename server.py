from tornado import websocket, web, ioloop

import os, uuid, json, base64_converter


class Client:
    def __init__(self, uuid, current_equation):
        self.buffer = []
        self.uuid = uuid
        self.current_equation = current_equation
        self.data = None


def find_client(uuid):
    # Can return None
    return clients.get(uuid)


def find_client_with_request_data(data):
    for k, client in clients.items():
        if (client.data == data):
            return client

    return None


clients = dict()


class WebSocket(websocket.WebSocketHandler):
    def open(self):
        print('Client connected to websocket!')

    def on_message(self, message):
        parsed_message = json.loads(message)

        # Find client
        client = find_client(parsed_message['uuid'])

        # Remove uuid from json
        parsed_message.pop('uuid', None)

        # Add data to buffer
        client.buffer.append(message)

        # Set client data if not set
        if not client.data:
            client.data = self

    def on_close(self):
        print('Client disconnected!')

        # Find client
        # client = find_client_with_request_data(self)

        # Delete client buffer
        # if client.buffer is not None:
        #    del client.buffer

        # Remove client from set
        # del clients[client.uuid]


class rest_handler(web.RequestHandler):
    def get(self, *args):
        # Create client object and add to set
        id = uuid.uuid4() 
        equation = '2 + 2 = 4'

        client = Client(uuid=str(id), current_equation=equation)
        clients[str(id)] = client

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
        id = self.get_body_argument("uuid")

        # Find client in set
        client = find_client(id)

        # Extract image and save image
        base64_converter.convertToImg(self.get_body_argument("b64_str"), client.current_equation)

        
        # Send client buffer to db

        # Reset buffer

        # Get an equation from db
        equation = '4 - 1 = 3'

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
