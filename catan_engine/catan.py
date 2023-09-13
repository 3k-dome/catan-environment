import subprocess
import time
from threading import Thread
from typing import Callable

from catan_engine.parameters import EngineParameters


def _run(parameters: EngineParameters, delay: int) -> subprocess.Popen[bytes]:
    time.sleep(delay)
    return subprocess.Popen(
        [
            "start",
            "cmd",
            "/c",
            rf"title Environment ({'single' if parameters.single else 'dynamic'}, {parameters.seed}, {parameters.port}) && .\catan_engine\catan-engine-console.exe {parameters.as_args()}",
        ],
        shell=True,
    )


def get_launch_callback(parameters: EngineParameters, delay: int = 5) -> Callable[..., subprocess.Popen[bytes] | None]:
    is_running = False

    def run() -> subprocess.Popen[bytes] | None:
        nonlocal is_running

        if not is_running:
            start_thread = Thread(target=lambda: _run(parameters, delay))
            start_thread.daemon = True
            start_thread.start()
            is_running = True

    return run
