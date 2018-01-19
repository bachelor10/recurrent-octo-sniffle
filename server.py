from tornado import websocket, web, ioloop

import os, uuid, json

class Client:
    def __init__(self, ip, uuid):
        self.buffer = []
        self.ip = ip
        self.uuid = uuid
        self.data = None

def find_client(ip, uuid):
    for client in clients:
        if client.ip == ip and client.uuid == uuid:
            return client
            
    return None

def find_client_with_request_data(data):
    for client in clients:
        if client.data == data:
            return client
            
    return None
    

clients = set()

class WebSocket(websocket.WebSocketHandler):

    def open(self):
        print('Client connected to websocket!')
        

    def on_message(self, message):

        parsed_message = json.loads(message)

        # Find client
        client = find_client(self.request.remote_ip, parsed_message['uuid'])

        # Remove uuid from json
        parsed_message.pop('uuid', None)

        # Add data to buffer
        client.buffer.append(message)

        # Set client data if not set
        if(client.data == None):
            client.data = self

    def on_close(self):
        print('Client disconnected!')

        # Find client
        client = find_client_with_request_data(self)

        # Delete client buffer
        del client.buffer

        # Remove client from set
        clients.remove(client)

class rest_handler(web.RequestHandler):
    
    def get(self, *args):
        # Create client object and add to set
        client = Client(ip = self.request.remote_ip, uuid = uuid.uuid4())
        clients.add(client)

        # Get an equation from db

        # Create json
        data = {
            'equation': '2 + 2 = 4',
            'uuid': str(client.uuid)
        }

        # Send equation and uuid to client
        self.write(json.dumps(data))


    def post(self):
        print(self.request.remote_ip)
        # Extract IP and UUID

        # Find client in set

        # Extract image

        # Save image

        # Send client buffer to db

        # Reset buffer

        # Get an equation from db

        # Send equation to client


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