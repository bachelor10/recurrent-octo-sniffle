from tornado import websocket, web, ioloop

class IndexHandler(web.RequestHandler):
    def get(self):
        self.render("index.html")

class WebSocket(websocket.WebSocketHandler):
    def open(self):
        print('Client connected!')
        
    def on_message(self, message):
        self.write_message(message)

    def on_close(self):
        print('Client disconnected!')

app = web.Application([
    (r'/', IndexHandler),
    (r'/ws', WebSocket)
])

if __name__ == '__main__':
    app.listen(8080)
    ioloop.IOLoop.instance().start()