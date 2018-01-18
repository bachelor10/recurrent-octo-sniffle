from tornado import websocket, web, ioloop

import os

class Client:
    def __init__(self, client):
        self.buffer = []
        self.identity = client

def find_client(identity):
    for client in WebSocket.clients:
        if client.identity == identity:
            return client
            
    return None

class WebSocket(websocket.WebSocketHandler):
    clients = set()

    def open(self):
        print('Client connected!')
        # create client object
        client = Client(self)

        # add client object to list
        WebSocket.clients.add(client)

    def on_message(self, message):
        find_client(self).buffer.append(message)
        
    def on_close(self):
        print('Client disconnected!')

        # Find client
        client = find_client(self)
        print(client.buffer)

        # Delete client buffer
        del client.buffer

        # Remove client from set
        WebSocket.clients.remove(client)

class rest_handler(web.RequestHandler):
    
    def get(self, *args):
        print('get')

        for client in WebSocket.clients:
            if client.identity == self:
                print('match!')

        # Fetch a trace from db
        # Initiate a trace
        # Send client trace instructions

    def post(self):
        print('post')
        # send bufferdata in clientobject to db
        # Send client a new trace

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