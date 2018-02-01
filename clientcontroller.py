import client


class ClientController():
    def __init__(self):
        self.clients = dict()

    def find_client(self, uuid):
        return self.clients.get(uuid)

    def find_client_with_request_data(self, data):
        for k, curr_client in self.clients.items():
            if curr_client.data == data:
                return curr_client

        return None
