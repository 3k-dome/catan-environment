import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable

from environment.models import ReceivedStateModel
from environment.queues import ENVIRONMENT_ACTION, ENVIRONMENT_STATE
from environment.server import reader

LOGGER = logging.getLogger("catan-environment")


class EnvironmentHttpServer(BaseHTTPRequestHandler):
    def do_POST(self):
        """POST-Request handler for the environment server.

        The catan environment work by communicating back and forth with the
        actual `catan-engine` that runs in a different process and simulates
        the actual game.

        Received observations are pushed into the `ENVIRONMENT_STATE` queue
        where they are then consumed by the actual environment. The environment
        then pushes the chosen action of the agent back into the `ENVIRONMENT_ACTION`
        queue which intern is consumed by this server as an response to incoming
        requests.
        """
        # receive new states and push them into the state queue
        data = reader.read_stream_as_json(self.rfile)
        state_model = ReceivedStateModel(**data)
        ENVIRONMENT_STATE.put(state_model)

        LOGGER.debug(f"Received and decoded 'StateModel' with message type '{state_model.type}'.")

        # wait until the state is processed and respond
        action_model = ENVIRONMENT_ACTION.get()
        action_model_encoded = action_model.to_json()

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(action_model_encoded)

        LOGGER.debug(f"Encoded and set 'ActionModel', selected action index was '{action_model.index}'.")

    def log_message(self, format: str, *args: Any) -> None:
        return

    @staticmethod
    def server_factory(host: str, port: int) -> tuple[HTTPServer, Callable[[], None], Callable[[], None]]:
        """Creates a new `HTTPServer` running on the given ip and port.

        :param host: Host address of the `HTTPServer`.
        :param port: Port of the `HTTPServer`.
        :return: The `HTTPServer` itself as well as a callback to start and terminate the server.
        """
        server_address = (host, port)
        server = HTTPServer(server_address, EnvironmentHttpServer)

        def start_server():
            LOGGER.debug(f"Environment listening on {host or '127.0.0.1'}:{port}.")
            server.serve_forever()

        def stop_server():
            LOGGER.debug(f"Environment stopped listening.")
            server.shutdown()
            server.server_close()

        return server, start_server, stop_server
