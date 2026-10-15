"""Per-step timing shared by all benchmark trainers (CascadeSnap and baselines).

At the end of every optimizer step, mark() records a CUDA event on the
training (current) stream and the CPU time.  No synchronization is inserted,
so the overlap between training and background checkpoint work is left
untouched.  The gap between two consecutive events is the step duration seen
by the training stream.  It includes GPU-side waits (wait_event) as well as
any CPU blocking that starves the GPU, so it measures training-visible stall
directly.

Enable with PCCHECK_STEP_EVENTS_PATH=<file.json>; dump() writes
  {"method", "steps", "is_ckpt", "gpu_ms", "cpu_ms", "warmup", "meta"}
where gpu_ms[i] / cpu_ms[i] is the duration of step steps[i] (i >= 1).
"""
import json
import os
import time

import torch


class StepTimer:
    def __init__(self, method, warmup=0, meta=None):
        self.path = os.environ.get("PCCHECK_STEP_EVENTS_PATH", "")
        self.enabled = bool(self.path) and torch.cuda.is_available()
        self.method = method
        self.warmup = int(warmup)
        self.meta = dict(meta or {})
        self._rows = []  # (step, is_ckpt, event, cpu_time)

    def mark(self, step, is_ckpt):
        if not self.enabled:
            return
        ev = torch.cuda.Event(enable_timing=True)
        ev.record(torch.cuda.current_stream())
        self._rows.append((int(step), bool(is_ckpt), ev, time.perf_counter()))

    def dump(self, extra=None):
        if not self.enabled or not self._rows:
            return None
        torch.cuda.synchronize()
        steps = [r[0] for r in self._rows]
        is_ckpt = [r[1] for r in self._rows]
        gpu_ms = [None] + [self._rows[i - 1][2].elapsed_time(self._rows[i][2])
                           for i in range(1, len(self._rows))]
        cpu_ms = [None] + [(self._rows[i][3] - self._rows[i - 1][3]) * 1000.0
                           for i in range(1, len(self._rows))]
        out = {"method": self.method, "warmup": self.warmup, "steps": steps,
               "is_ckpt": is_ckpt, "gpu_ms": gpu_ms, "cpu_ms": cpu_ms,
               "meta": dict(self.meta, **(extra or {}))}
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(out, f)
        os.replace(tmp, self.path)
        return out
