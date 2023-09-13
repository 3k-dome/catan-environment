from dataclasses import dataclass


@dataclass
class EngineParameters:
    single: bool
    verbose: bool
    assisted: bool
    init: bool
    episodes: int
    port: int
    seed: int

    def as_args(self) -> str:
        return (
            f"{'--single' if self.single else ''} {'--verbose' if self.verbose else ''} {'--assisted' if self.assisted else ''} "
            + f"--episodes {self.episodes} --port {self.port} --seed {self.seed} {'--init' if self.init else ''}"
        )
