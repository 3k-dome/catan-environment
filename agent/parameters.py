from dataclasses import dataclass
from typing import Literal


@dataclass
class DropoutDefinition:
    rate: float
    seed: int


@dataclass
class LayerDefinition:
    activation: Literal["relu", "linear"]
    size: int
    seed: int
    dropout: DropoutDefinition | None


@dataclass
class AgentParams:
    gamma: float
    n_steps: int
    batchsize: int
    soft_updates: bool
    epsilon_steps: int
    epsilon_start: float = 1.00
    epsilon_end: float = 0.1
    network_update_frequency: int = 4  # [sampled actions]
    buffer_size: int = 100_000

    # dqn: after 10_000 trainings steps a hard updated (t = 1) is performed
    # t-soft: after each training step a soft update with (t = 0.001) is performed

    target_update_frequency: int = 10_000  # [trainings steps]
    target_update_tau: float = 1

    def __post_init__(self) -> None:
        if self.soft_updates:
            self.target_update_frequency = 1
            self.target_update_tau = 0.001
