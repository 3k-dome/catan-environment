from dataclasses import dataclass


@dataclass
class SlaveParameters:
    port: int
    episodes: int

    adaptive: bool = False
    name: str = ""
    swap_start: int = 0
    swap_interval: int = 0
    window_width: int = 0
    window_offset: int = 0

    def __post_init__(self) -> None:
        if self.adaptive:
            has_name = self.name and self.name != ""
            valid_interval = (self.episodes - self.swap_start) % self.swap_interval == 0
            if not has_name or not valid_interval:
                raise Exception()

    def as_args(self, offset: int) -> str:
        base = f"--port {self.port + offset} --episodes {self.episodes}"
        if self.adaptive:
            base += f" --adaptive --name {self.name} --swap_start {self.swap_start} --swap_interval {self.swap_interval} --window_width {self.window_width} --window_offset {self.window_offset}"
        return base
