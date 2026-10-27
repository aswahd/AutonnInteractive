"""Append-only evaluation records bound to a fixed checkpoint and protocol.

These records contain scores and crop provenance, never cached support features.
A short allocation may stop between tasks and resume the exact remaining tasks.
"""

import json
import os
import time
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class TaskJournal:
    def __init__(self, directory, binding, rank=0, world_size=1, deadline=None, max_tasks=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rank, self.world_size = rank, world_size
        self.deadline, self.max_tasks = deadline, max_tasks
        self.added = 0
        path = self.directory / f"rank{rank}.binding.json"
        canonical = {"binding": binding, "rank": rank, "world_size": world_size}
        if path.exists():
            if json.loads(path.read_text()) != canonical:
                raise ValueError("validation journal checkpoint/protocol/world-size mismatch")
        else:
            atomic_json(path, canonical)
        self.path = self.directory / f"rank{rank}.jsonl"
        self.tasks = {}
        if self.path.exists():
            data = self.path.read_bytes()
            # Only a complete newline-terminated record is committed. Recover a
            # partial final write after interruption, without discarding prior work.
            end = data.rfind(b"\n") + 1
            if end != len(data):
                with self.path.open("r+b") as handle:
                    handle.truncate(end)
            for line in data[:end].splitlines():
                record = json.loads(line)
                index = record["task_index"]
                if index in self.tasks or index % world_size != rank:
                    raise ValueError("duplicate or incorrectly assigned validation task")
                self.tasks[index] = record["rows"]

    def indices(self, total):
        assigned = set(range(self.rank, total, self.world_size))
        if not set(self.tasks) <= assigned:
            raise ValueError("journal contains a task outside this evaluation protocol")
        for index in sorted(assigned - set(self.tasks)):
            if self.deadline is not None and time.monotonic() >= self.deadline:
                return
            if self.max_tasks is not None and self.added >= self.max_tasks:
                return
            yield index

    def record(self, index, rows):
        if index in self.tasks or index % self.world_size != self.rank or not rows:
            raise ValueError("invalid or duplicate validation task commit")
        data = json.dumps({"task_index": index, "rows": rows}, allow_nan=False) + "\n"
        with self.path.open("a") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        self.tasks[index] = rows
        self.added += 1

    def complete(self, total):
        return set(self.tasks) == set(range(self.rank, total, self.world_size))

    def rows(self):
        return [row for index in sorted(self.tasks) for row in self.tasks[index]]


def task_indices(total, rank, world_size, progress=None):
    return range(rank, total, world_size) if progress is None else progress.indices(total)


def commit_task(progress, index, rows):
    if progress is not None:
        progress.record(index, rows)
