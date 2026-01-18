import csv
import os
from typing import Iterable, Tuple

from ti.utils import ensure_dir


class LossLogger:
    def __init__(self, path: str, flush_every: int = 200):
        self.path = path
        self.flush_every = int(flush_every)
        self.buffer = []
        self._init_file()

    def _init_file(self):
        ensure_dir(os.path.dirname(self.path))
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "loss"])

    def log(self, step: int, loss: float):
        self.buffer.append((int(step), float(loss)))
        if len(self.buffer) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(self.buffer)
        self.buffer = []


def append_rows(path: str, rows: Iterable[Tuple[int, float]], header=("step", "loss")):
    ensure_dir(os.path.dirname(path))
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(list(header))
        writer.writerows(rows)
