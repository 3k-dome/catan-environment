from queue import Queue
from environment.models import ReceivedStateModel, SubmittedActionModel

ENVIRONMENT_STATE: Queue[ReceivedStateModel] = Queue()
ENVIRONMENT_ACTION: Queue[SubmittedActionModel] = Queue()
