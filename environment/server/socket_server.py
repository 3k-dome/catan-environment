import logging
import socket
import struct
from typing import Callable

import orjson

from environment.models import ReceivedStateModel
from environment.queues import ENVIRONMENT_ACTION, ENVIRONMENT_STATE

LOGGER = logging.getLogger("catan-environment")


class EnvironmentSocketServer:
    def __init__(self, port: int) -> None:
        self.port = port
        self.serve = True
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    @staticmethod
    def encode_message(message: bytes) -> bytes:
        return struct.pack(">I", len(message)) + message

    @staticmethod
    def decode_message(message: bytes) -> bytes:
        length = struct.unpack(">I", message[:4])[0]
        return message[4 : (4 + length)]

    def start(self) -> None:
        self.socket.bind(("127.0.0.1", self.port))
        self.socket.listen()
        self.connection, _ = self.socket.accept()

        while True:
            try:
                self.run()
            except ConnectionResetError:
                return

    def run(self) -> None:
        message = self.connection.recv(4096)
        decoded = EnvironmentSocketServer.decode_message(message)
        state_model = ReceivedStateModel(**orjson.loads(decoded))
        ENVIRONMENT_STATE.put(state_model)
        LOGGER.debug(f"Received and decoded 'StateModel' with message type '{state_model.type}'.")

        action_model = ENVIRONMENT_ACTION.get()
        action_model_encoded = orjson.dumps(action_model)
        self.connection.sendall(EnvironmentSocketServer.encode_message(action_model_encoded))
        LOGGER.debug(f"Encoded and set 'ActionModel', selected action index was '{action_model.index}'.")

    @staticmethod
    def server_factory(host: str, port: int) -> tuple["EnvironmentSocketServer", Callable[[], None], Callable[[], None]]:
        server = EnvironmentSocketServer(port)

        def start_server():
            LOGGER.debug(f"Environment listening on {host or '127.0.0.1'}:{port}.")
            server.start()

        def stop_server():
            LOGGER.debug(f"Environment stopped listening.")
            server.connection.close()

        return server, start_server, stop_server
