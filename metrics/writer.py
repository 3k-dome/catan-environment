from dataclasses import dataclass
from pathlib import Path

from metrics.evaluation import EvaluationMetrics


@dataclass
class EvaluationWriter:
    file_path: Path

    def __post_init__(self) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.file_path, "a") as file:
            file.write(f"{EvaluationMetrics.header()}\n")

    def add(self, metrics: EvaluationMetrics) -> None:
        with open(self.file_path, "a") as file:
            file.write(f"{metrics}\n")


@dataclass
class LossWriter:
    file_path: Path

    def __post_init__(self) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.file_path, "a") as file:
            file.write(f"Loss\n")

    def add(self, loss: list[float]) -> None:
        with open(self.file_path, "a") as file:
            file.write("\n".join([str(x) for x in loss]) + "\n")
