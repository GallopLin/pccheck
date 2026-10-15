import json
import os
import torch
import time
from checkpoint_eval.pccheck.chk_checkpoint_pipeline import Checkpoint
from torch.multiprocessing import Pool, Process, set_start_method, Manager, Value, Lock, Barrier

# 🔧 FIX: 尝试设置 spawn 启动方式以避免 CUDA fork 问题
# 这确保子进程有干净的 CUDA 上下文
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    # 如果已经设置过，忽略错误
    pass


def _write_recovery_metadata(
    metadata_file,
    *,
    completed_steps=None,
    checkpoint_file=None,
    max_async=None,
    extra_metadata=None,
):
    payload = {}
    if metadata_file and os.path.exists(metadata_file):
        with open(metadata_file, "r", encoding="utf-8") as f:
            try:
                payload = json.load(f)
            except Exception:
                payload = {}

    payload["format"] = "pccheck-linear-v1"
    if completed_steps is not None:
        payload["completed_steps"] = int(completed_steps)
    if checkpoint_file is not None:
        payload["checkpoint_file"] = checkpoint_file
    if max_async is not None:
        payload["max_async"] = int(max_async)
    if extra_metadata:
        payload.update(extra_metadata)

    if metadata_file:
        os.makedirs(os.path.dirname(os.path.abspath(metadata_file)), exist_ok=True)
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


class Chk_monitor:

    def __init__(
        self,
        c_lib_path,
        total_size,
        num_threads,
        max_async,
        gpu_copy,
        gpu_ar,
        ratio=2.0,
        is_sync=False,
        bsize=None,
        memory_saving=False,
        is_distributed=False,
        rank=0,
        world_size=1,
        checkpoint_file=None,
        recovery_metadata_file=None,
        recovery_checkpoint_file=None,
        **kwargs,
    ):

        # only 1 background process
        basic_path = checkpoint_file or f"pccheck_checkpoint_rank{rank}.chk"
        self.checkpoint_file = basic_path
        self.max_async = max_async
        self.recovery_metadata_file = recovery_metadata_file
        self.recovery_checkpoint_file = recovery_checkpoint_file or basic_path
        self.lock = Lock()
        self.cp_in_progress = Value("i", 0)
        self.active_async = Value("i", 0)
        self.start = Value("i", 0)
        self.stop = Value("i", 0)
        self.barrier = Barrier(2)

        self.checkpoint_dict = {}

        for name, ref in kwargs.items():
            self.checkpoint_dict[name] = ref

        print(f"BSIZE IS {bsize}")

        chk = Checkpoint(
            total_size,
            num_threads,
            basic_path,
            c_lib_path,
            max_async,
            ratio=ratio,
            gpu_ar=gpu_ar,
            bsize=bsize,
            memory_saving=memory_saving,
            is_distributed=is_distributed,
            rank=rank,
            world_size=world_size
        )

        self.chk_process = Process(
            target=chk.start_chk,
            args=[
                self.barrier,
                self.lock,
                self.checkpoint_dict,
                self.cp_in_progress,
                self.active_async,
                self.start,
                self.stop,
                gpu_copy,
                is_sync,
            ],
        )
        self.chk_process.start()
        if os.environ.get("PCCHECK_DIAG") == "1":
            self._start_diag()
        self.barrier.wait()
        # print("Chk process started! PID is: ", self.chk_process.pid)

    def _start_diag(self):
        """诊断：每 10 s 不加锁地读取共享标志，并每 60 s 打印本进程线程栈。"""
        import faulthandler
        import sys as _sys
        import threading
        faulthandler.dump_traceback_later(60, repeat=True, file=_sys.stderr)

        def loop():
            while True:
                time.sleep(10)
                vals = {k: getattr(self, k).get_obj().value
                        for k in ("cp_in_progress", "start", "active_async", "stop")}
                print(f"[pccheck-diag] {time.strftime('%H:%M:%S')} {vals} "
                      f"child_alive={self.chk_process.is_alive()}", flush=True)
        threading.Thread(target=loop, daemon=True).start()

    def gpu_copy_in_progress(self):

        # return True if at least one of the background processes is copying
        with self.lock:
            if self.cp_in_progress.value == 1:
                return True

        return False

    def checkpoint_in_progress(self):
        with self.lock:
            if self.start.value == 1:
                return True

        return False

    def async_checkpoint_count(self):
        with self.lock:
            return self.active_async.value

    def try_save(self):
        with self.lock:
            if self.start.value != 0:
                return False
            if self.active_async.value >= 1:
                return False
            self.cp_in_progress.value = 1
            self.start.value = 1
            return True

    def save(self, completed_steps=None, extra_metadata=None):
        # print(f"******************** CALL SAVE ********************")  # 减少输出

        _write_recovery_metadata(
            self.recovery_metadata_file,
            completed_steps=completed_steps,
            checkpoint_file=self.recovery_checkpoint_file,
            max_async=self.max_async,
            extra_metadata=extra_metadata,
        )

        while True:
            with self.lock:
                if self.start.value == 0:
                    break

        with self.lock:
            self.cp_in_progress.value = 1
            self.start.value = 1

    def drain(self, timeout=1800, poll=0.02):
        """等待在途 checkpoint 全部写完（start==0 且 active_async==0，连续 3 次观测）。

        基准结束时先 drain 再 kill：否则尾部那次 checkpoint 会被 30 s 超时直接终止，
        既没写完，耗时也不计入 EXECUTION TIME（对比会偏向本基线）。返回等待秒数。"""
        t0 = time.time()
        stable = 0
        while stable < 3:
            with self.lock:
                idle = self.start.value == 0 and self.active_async.value == 0
            stable = stable + 1 if idle else 0
            if time.time() - t0 > timeout:
                print(f"[WARN] PCcheck drain timed out after {timeout}s")
                break
            time.sleep(poll)
        return time.time() - t0

    def kill_checkpoint(self, timeout=30):

        with self.lock:
            self.stop.value = 1
            # 避免 save()/后台循环卡在旧状态
            self.start.value = 0
            self.cp_in_progress.value = 0

        self.chk_process.join(timeout=timeout)
        if self.chk_process.is_alive():
            try:
                self.chk_process.terminate()
            except Exception:
                pass
            self.chk_process.join(timeout=5)

        if self.chk_process.is_alive():
            try:
                self.chk_process.kill()
            except Exception:
                pass
            self.chk_process.join(timeout=5)
