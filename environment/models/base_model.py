import json
from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Any


@dataclass
class BaseModel:
    def to_dictionary(self) -> dict[str, Any]:
        """Converts this model into a plain dictionary.

        :return: A dictionary representation of this model.
        """
        return asdict(self)

    def to_json(self) -> bytes:
        """Converts this model to a json byte stream.

        :return: The 'utf-8' encoded json representation of this model.
        """
        json_str = json.dumps(self.to_dictionary())
        return json_str.encode("utf-8")

    @classmethod
    def from_json(cls, json_stream: BytesIO) -> "BaseModel":
        """Factory method to create an instance of this model from a json byte stream.

        :param json_stream: A readable json byte stream.
        :return: A instance of this model.
        """
        decoded = json.load(json_stream)
        return cls(**decoded)
