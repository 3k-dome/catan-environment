import logging
import sys
from logging import Logger
from logging.handlers import QueueHandler, QueueListener
from queue import Queue
from typing import Any


def setup_logging() -> Logger:
    que: Queue[Any] = Queue()
    queue_handler = QueueHandler(que)
    handler = logging.StreamHandler(sys.stdout)
    listener = QueueListener(que, handler)
    formatter = logging.Formatter("%(levelname)s,%(asctime)s,%(message)s", "%d-%m-%Y %H:%M:%S")
    handler.setFormatter(formatter)

    root = logging.getLogger("catan-environment")
    root.addHandler(queue_handler)
    root.setLevel("DEBUG")
    root.propagate = False
    
    listener.start()
    return root
