import torch
import os
import re
from os.path import isfile
from torch.multiprocessing import Pool, Process, set_start_method, Manager, Value, Lock


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    Sort helper for natural sorting of checkpoint filenames.
    Based on CheckFreq source code (cf_manager.py).
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def get_latest_checkpoint(chk_dir, checkpoint_format="./checkpoint-{epoch}-{it}.chk"):
    """
    Finds the latest checkpoint file in the given directory.
    Based on CheckFreq source code (CFManager.get_latest_checkpoint and initalize_chk_dir).
    
    Args:
        chk_dir: Directory to search for checkpoint files.
        checkpoint_format: The checkpoint filename format pattern.
    
    Returns:
        Full path to the latest checkpoint file, or None if no checkpoint found.
    """
    if not os.path.exists(chk_dir):
        print("Checkpoint directory {} does not exist.".format(chk_dir))
        return None

    # Get list of all .chk files in the directory
    chk_files = [f for f in os.listdir(chk_dir) if f.endswith('.chk') and isfile(os.path.join(chk_dir, f))]

    if len(chk_files) == 0:
        print("No checkpoint files found in {}".format(chk_dir))
        return None

    # Sort by natural order to get the latest checkpoint
    chk_files.sort(key=natural_keys)

    latest_chk = chk_files[-1]
    filepath = os.path.join(chk_dir, latest_chk)
    print("Latest checkpoint found: {}".format(filepath))
    return filepath


def restore_checkpoint(
    chk,
    chk_dir="./",
    checkpoint_path=None,
    checkpoint_format="./checkpoint-{epoch}-{it}.chk",
    gpu=0,
):
    """
    Restores the latest checkpoint, based on CheckFreq source code
    (CFManager.restore and CFCheckpoint._restore).

    Args:
        chk: An instance of CFCheckpoint that tracks model/optimizer state.
        chk_dir: Directory containing checkpoint files.
        checkpoint_path: If provided, restore from this specific path instead of
                         searching for the latest. 
        checkpoint_format: The checkpoint filename format pattern.
        gpu: GPU device ID for loading the checkpoint.

    Returns:
        A dict of extra state (epoch, iter, etc.) that was saved alongside 
        the tracked objects, or None if no checkpoint was found/restored.
    """
    if checkpoint_path is None:
        # Search for the latest checkpoint in the directory
        checkpoint_path = get_latest_checkpoint(chk_dir, checkpoint_format)

    if checkpoint_path is None:
        print("No checkpoint to restore from.")
        return None

    if not os.path.isfile(checkpoint_path):
        print("Checkpoint file {} does not exist.".format(checkpoint_path))
        return None

    print("Restoring checkpoint from: {}".format(checkpoint_path))
    extra_state = chk._restore(filepath=checkpoint_path, gpu=gpu)
    return extra_state


def save_checkpoint(
    checkpoint_format,
    path_to_pmem,
    filepath,
    additional_snapshot,
    chk,
    active_snapshot,
    in_progress_snapshot,
    lock,
    epoch,
    it,
    last_chk_it,
    change,
    profile_snap,
    sync=False,
):

    filepath.value = checkpoint_format.format(epoch=epoch, it=0)
    if path_to_pmem != "":
        filepath.value = f"{path_to_pmem}/{filepath.value}"

    additional_snapshot["epoch"] = epoch
    additional_snapshot["iter"] = it
    # print(f"Call {chk.chk_process} at epoch {epoch} and iteration {it}")

    if not chk.spawned:
        print("------------- START A NEW PROCESS!! ------------")
        keywords = {
            "snapshot_ready": False,
            "profile_snap": profile_snap,
            "background": True,
            "iter_chk": last_chk_it,
            "overwrite": True,
        }
        chk.chk_process = Process(
            target=chk._serialize_and_persist,
            args=[
                filepath,
                active_snapshot,
                in_progress_snapshot,
                lock,
                change,
                additional_snapshot,
            ],
            kwargs=keywords,
        )
        chk.chk_process.start()
        chk.spawned = True
        print("-------------- PROCESS STARTED!! ----------")

    if chk.chk_process is not None:
        while change.value == 1:
            # this means a checkpoint is on progress (wait for process doing the checkpoint to set variable to 0)
            continue

    # Once complete, initiate the next checkpoint synchronously
    with lock:
        in_progress_snapshot.value = 1
        change.value = 1


def make_shm(obj):

    if obj is None:
        return

    if torch.is_tensor(obj):
        obj.share_memory_()

    elif isinstance(obj, dict):
        for name, ref in obj.items():
            make_shm(ref)

    elif isinstance(obj, list):
        for x in obj:
            make_shm(x)
    else:
        return
