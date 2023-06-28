from io import BytesIO
from typing import Any, BinaryIO
import json


def read_chunked_content(input_stream: BinaryIO, output_stream: BytesIO) -> None:
    """Reads and moves the given chunked binary stream into the given byte stream.

    Source: https://stackoverflow.com/a/63037533

    :param input_stream: Open binary stream to read from.
    :param output_stream: Open byte stream to write to (file position is reset once done).
    """

    while True:
        line = input_stream.readline().strip()
        chunk_size = int(line, 16)

        if chunk_size != 0:
            chunk = input_stream.read(chunk_size).strip()
            output_stream.write(chunk)

        input_stream.readline()
        if chunk_size == 0:
            break

    output_stream.seek(0)


def read_stream_as_json(input_stream: BinaryIO) -> dict[str, Any]:
    with BytesIO() as content:
        read_chunked_content(input_stream, content)
        return json.load(content)
