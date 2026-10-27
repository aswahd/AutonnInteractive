"""Bounded CPU episode prefetch."""

import json
import multiprocessing
import signal
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import deepcopy
from threading import Lock, local

import torch

from .data import InvalidCropError, collate_episodes
from .optimization import episode_rng


def _initialize_worker(sampler, config, rejected_path, lock):
    global _worker_stream
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)  # The training process handles Slurm's checkpoint signal.
    torch.set_num_threads(1)
    _worker_stream = EpisodeStream(sampler, config, 0, 1, config.updates, rejected_path, prefetch=False)
    _worker_stream.rejection_lock = lock


def _sample_in_process(episode_index):
    return _worker_stream._sample(episode_index)


class EpisodeStream:
    def __init__(self, sampler, config, rank, world_size, limit, rejected_path, prefetch=True, collate=False):
        self.sampler, self.config = sampler, config
        self.rank, self.world_size = rank, world_size
        self.limit, self.rejected_path = limit, rejected_path
        self.accumulation = config.accumulation_steps(world_size)
        self.workers, self.rejection_lock = local(), Lock()
        self.executor = None
        if prefetch and config.worker_backend == "process":
            context = multiprocessing.get_context("spawn")
            self.rejection_lock = context.Lock()
            self.executor = ProcessPoolExecutor(
                max_workers=config.num_workers,
                mp_context=context,
                initializer=_initialize_worker,
                initargs=(sampler, config, rejected_path, self.rejection_lock),
            )
        elif prefetch:
            self.executor = ThreadPoolExecutor(max_workers=config.num_workers)
        self.pending, self.expected = {}, None
        self.collate = collate
        self.pin_memory = collate and config.pin_memory and torch.cuda.is_available()
        self.pinner = (
            ThreadPoolExecutor(
                max_workers=1, initializer=torch.cuda.set_device, initargs=(torch.cuda.current_device(),)
            )
            if self.executor and self.pin_memory
            else None
        )
        self.ready = {}

    def _sample(self, episode_index):
        started = time.monotonic()
        if not hasattr(self.workers, "sampler"):
            # Each worker owns its decompressor handles and prompt generator.
            self.workers.sampler = deepcopy(self.sampler) if self.executor else self.sampler
        for attempt in range(16):
            try:
                episode = self.workers.sampler.sample_episode(episode_rng(self.config.seed, episode_index, attempt))
                break
            except InvalidCropError as error:
                with self.rejection_lock, self.rejected_path.open("a") as handle:
                    handle.write(
                        json.dumps({"episode_index": episode_index, "attempt": attempt, "reason": str(error)}) + "\n"
                    )
        else:
            raise RuntimeError("could not sample a valid episode in 16 attempts")
        episode.metadata["global_episode_index"] = episode_index
        return episode, time.monotonic() - started

    def _episode_index(self, local_index):
        micro, within = divmod(local_index, self.config.microbatch)
        update, micro = divmod(micro, self.accumulation)
        return (
            update * self.config.effective_batch
            + micro * self.world_size * self.config.microbatch
            + self.rank * self.config.microbatch
            + within
        )

    def get(self, update, micro):
        key = (update * self.accumulation + micro) * self.config.microbatch
        if self.expected is not None and self.expected != key:
            raise ValueError("episode prefetch consumed out of order")
        if self.executor is None:
            results = [self._sample(self._episode_index(i)) for i in range(key, key + self.config.microbatch)]
        else:
            # Prefetch is bounded in episodes, independent of the GPU microbatch size.
            depth = max(
                (2 if self.pinner else 1) * self.config.microbatch,
                self.config.num_workers * self.config.prefetch_factor,
            )
            for index in range(key, min(key + depth, self.limit * self.accumulation * self.config.microbatch)):
                if index not in self.pending:
                    sample = _sample_in_process if self.config.worker_backend == "process" else self._sample
                    self.pending[index] = self.executor.submit(sample, self._episode_index(index))
            if self.pinner:
                for start in (key, key + self.config.microbatch):
                    if start < self.limit * self.accumulation * self.config.microbatch and start not in self.ready:
                        futures = [self.pending[i] for i in range(start, start + self.config.microbatch)]
                        self.ready[start] = self.pinner.submit(self._pin_batch, futures)
                        # The collator owns/reclaims shared tensors; GPU dispatch must not unmap gigabytes.
                        for index in range(start, start + self.config.microbatch):
                            self.pending[index] = None
                result = self.ready.pop(key).result()
                for index in range(key, key + self.config.microbatch):
                    del self.pending[index]
                self.expected = key + self.config.microbatch
                return result
            results = [self.pending.pop(i).result() for i in range(key, key + self.config.microbatch)]
        self.expected = key + self.config.microbatch
        episodes, durations = zip(*results)
        batch = collate_episodes(episodes, pin_memory=self.pin_memory) if self.collate else list(episodes)
        return batch, sum(durations)

    @staticmethod
    def _pin_batch(futures):
        episodes, durations = zip(*(future.result() for future in futures))
        return collate_episodes(episodes, pin_memory=True), sum(durations)

    def quiesce(self):
        """Finish outstanding CPU work before validation/checkpoint RNG capture."""
        for future in self.pending.values():
            if future is not None:
                future.result()
        for future in self.ready.values():
            future.result()

    def close(self):
        if self.pinner is not None:
            self.pinner.shutdown(wait=True, cancel_futures=True)
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
