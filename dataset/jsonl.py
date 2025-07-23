import json
import sys
from typing import Any, Generator

def parse_jsonl(filepath: str) -> Generator[dict[str, Any], None, None]:
    """Parse a JSONL file (one JSON per line)."""
    with open(filepath, "r") as f:
        # If there is a JSONDecodeError, catch it and print the line number.
        for line_no, line in enumerate(f, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSONL on line {line_no}:\n{line}\n{e}\nSkipping line…", file=sys.stderr)
                continue
            yield row


def write_jsonl(filepath: str, data: list[dict[str, Any]]) -> None:
    """Write a list of dictionaries to a JSONL file."""
    with open(filepath, "w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")
