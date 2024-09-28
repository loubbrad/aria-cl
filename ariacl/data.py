import mmap
import os
import gc
import io
import base64
import orjson
import torch

from multiprocessing import Queue, Lock, Process
from queue import Empty
from typing import List

from ariacl.audio import get_wav_segments_b, get_audio_intervals
from ariacl.config import load_config


def _get_supervised_batches_and_intervals(
    piano_path: str, config: dict, device: str
):
    piano_wav_batch = get_wav_segments_b(
        audio_path=piano_path,
        stride_factor=config["data"]["stride_factor"],
        device=device,
    )
    silent_intervals_batch = get_audio_intervals(
        wav=piano_wav_batch,
        min_window_s=config["data"]["piano_detection"]["min_window_silence_s"],
        threshold_db=config["data"]["piano_detection"]["silence_threshold_db"],
        detect_silent_intervals=True,
    )

    return piano_wav_batch, silent_intervals_batch


# TODO: Debug why this is slowing down overtime (torch.save issue?)
def supervised_worker(
    load_path_queue: Queue,
    save_file_lock,
    save_path: str,
    label: int,
    device="cuda",
):
    config = load_config()
    num_files_processed = 0
    label = base64.b64encode(orjson.dumps(label)).decode("utf-8")
    non_piano_label = base64.b64encode(orjson.dumps(0)).decode("utf-8")

    if device == "cuda":
        assert torch.cuda.is_available()

    while True:
        try:
            load_path = load_path_queue.get(timeout=0.01)
        except Empty:
            if load_path_queue.empty():
                print(f"Worker {os.getpid()} finished")
                return
            continue

        try:
            with torch.no_grad():
                wav_batch, silent_intervals_batch = (
                    _get_supervised_batches_and_intervals(
                        piano_path=load_path,
                        config=config,
                        device=device,
                    )
                )
        except torch.OutOfMemoryError as e:
            print("Not enough CUDA memory, offloading to CPU")
            torch.cuda.empty_cache()
            gc.collect()

            try:
                with torch.no_grad():
                    wav_batch, silent_intervals_batch = (
                        _get_supervised_batches_and_intervals(
                            piano_path=load_path,
                            config=config,
                            device="cpu",
                        )
                    )
            except Exception as e:
                print(f"Failed to process {load_path}")
                continue
        except Exception as e:
            print(f"Failed to process {load_path}")
            torch.cuda.empty_cache()
            gc.collect()
            continue

        for idx, silent_intervals in enumerate(silent_intervals_batch):
            _label = non_piano_label if silent_intervals else label

            wav_to_save = wav_batch[idx].clone()

            with io.BytesIO() as wav_buffer:
                torch.save(wav_to_save.cpu(), wav_buffer)
                wav_buffer.seek(0)
                wav_bytes = wav_buffer.read()
                wav_str = base64.b64encode(wav_bytes).decode("utf-8")

            with save_file_lock:
                with open(save_path, mode="a") as f:
                    f.write(wav_str)
                    f.write("\n")
                    f.write(_label)
                    f.write("\n")

        torch.cuda.empty_cache()
        gc.collect()

        num_files_processed += 1
        if num_files_processed % 25 == 0:
            print(
                f"Worker {os.getpid()} has finished {num_files_processed} files, {load_path_queue.qsize()} remaining"
            )


def _get_source_separated_batches_and_intervals(
    piano_path: str, other_path: str, config: dict, device: str
):
    piano_wav_batch = get_wav_segments_b(
        audio_path=piano_path,
        stride_factor=config["data"]["stride_factor"],
        device=device,
    )
    other_wav_batch = get_wav_segments_b(
        audio_path=other_path,
        stride_factor=config["data"]["stride_factor"],
        device=device,
    )

    if piano_wav_batch.shape[0] != other_wav_batch.shape[0]:
        raise ValueError

    silent_intervals_batch = get_audio_intervals(
        wav=piano_wav_batch,
        min_window_s=config["data"]["piano_detection"]["min_window_silence_s"],
        threshold_db=config["data"]["piano_detection"]["silence_threshold_db"],
        detect_silent_intervals=True,
    )
    non_piano_intervals_batch = get_audio_intervals(
        wav=other_wav_batch,
        min_window_s=config["data"]["piano_detection"]["min_window_other_s"],
        threshold_db=config["data"]["piano_detection"]["noise_threshold_db"],
        detect_silent_intervals=False,
    )

    return (
        piano_wav_batch,
        other_wav_batch,
        silent_intervals_batch,
        non_piano_intervals_batch,
    )


# Some good parameters for non-piano detection
# GiantMIDI (Not 100% sure): -25dB, W=1.5
# Noisy piano: -20dB, W=1.5
# Piano concerto: -30dB, W=1.0
# Score-4: -22dB, W=1.0 - This must be wrong, reset this
# Jazz trio (??): -30dB, W=1.0


# TODO: Fix deadlock issue from using multiple workers
def source_separated_worker(
    load_path_queue: Queue, save_file_lock, save_path: str, device="cuda"
):
    config = load_config()
    piano_label = base64.b64encode(orjson.dumps(1)).decode("utf-8")
    non_piano_label = base64.b64encode(orjson.dumps(0)).decode("utf-8")
    num_files_processed = 0

    if device == "cuda":
        assert torch.cuda.is_available()

    while True:
        try:
            load_path = load_path_queue.get(timeout=0.01)
        except Empty:
            if load_path_queue.empty():
                print(f"Worker {os.getpid()} finished")
                return
            continue

        try:
            with torch.no_grad():
                (
                    piano_wav_batch,
                    other_wav_batch,
                    silent_intervals_batch,
                    non_piano_intervals_batch,
                ) = _get_source_separated_batches_and_intervals(
                    piano_path=load_path["piano"],
                    other_path=load_path["other"],
                    config=config,
                    device=device,
                )
        except torch.OutOfMemoryError as e:
            print("Not enough CUDA memory, offloading to CPU")
            torch.cuda.empty_cache()
            gc.collect()

            try:
                with torch.no_grad():
                    (
                        piano_wav_batch,
                        other_wav_batch,
                        silent_intervals_batch,
                        non_piano_intervals_batch,
                    ) = _get_source_separated_batches_and_intervals(
                        piano_path=load_path["piano"],
                        other_path=load_path["other"],
                        config=config,
                        device="cpu",
                    )
            except Exception as e:
                print(f"Failed to process {load_path}")
                continue
        except Exception as e:
            print(f"Failed to process {load_path}")
            torch.cuda.empty_cache()
            gc.collect()
            continue

        for idx, (silent_intervals, non_piano_intervals) in enumerate(
            zip(silent_intervals_batch, non_piano_intervals_batch)
        ):

            _label = (
                non_piano_label
                if non_piano_intervals or silent_intervals
                else piano_label
            )

            wav_to_save = piano_wav_batch[idx] + other_wav_batch[idx]

            with io.BytesIO() as wav_buffer:
                torch.save(wav_to_save.cpu(), wav_buffer)
                wav_buffer.seek(0)
                wav_bytes = wav_buffer.read()
                wav_str = base64.b64encode(wav_bytes).decode("utf-8")

            with save_file_lock:
                with open(save_path, mode="a") as f:
                    f.write(wav_str)
                    f.write("\n")
                    f.write(_label)
                    f.write("\n")

        torch.cuda.empty_cache()
        gc.collect()

        num_files_processed += 1
        if num_files_processed % 25 == 0:
            print(
                f"Worker {os.getpid()} has finished {num_files_processed} files, {load_path_queue.qsize()} remaining"
            )


# TODO: Replace this mmap implementation with numpy memmap
class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, load_paths: str | list):
        super().__init__()

        if isinstance(load_paths, str):
            load_paths = [load_paths]
        self.file_buffs = []
        self.file_mmaps = []
        self.index = []

        for path in load_paths:
            buff = open(path, mode="r")
            self.file_buffs.append(buff)
            mmap_obj = mmap.mmap(buff.fileno(), 0, access=mmap.ACCESS_READ)
            self.file_mmaps.append(mmap_obj)

            _index = self._build_index(mmap_obj)
            self.index.extend(
                [(len(self.file_mmaps) - 1, pos) for pos in _index]
            )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        file_id, pos = self.index[idx]
        mmap_obj = self.file_mmaps[file_id]
        mmap_obj.seek(pos)

        # Hardcoded, perhaps remove?
        wav = torch.load(
            io.BytesIO(base64.b64decode(mmap_obj[pos : pos + 589521 - 1])),
            weights_only=True,
        )
        label = orjson.loads(
            base64.b64decode(mmap_obj[pos + 589521 : pos + 589521 + 5])
        )

        return wav, torch.tensor(label, dtype=torch.float32)

    def close(self):
        for buff in self.file_buffs:
            buff.close()
        for mmap in self.file_mmaps:
            mmap.close()

    def __del__(self):
        self.close()

    def _build_index(self, mmap_obj):
        mmap_obj.seek(0)
        pos = mmap_obj.find(b"\n", 0)
        pos = mmap_obj.find(b"\n", pos + 1)
        audio_bytes_len = pos + 1

        return list(range(0, len(mmap_obj), audio_bytes_len))

    @classmethod
    def build_supervised(
        cls,
        load_paths: list,
        save_path: str,
        label: int,
        num_processes: int = 1,
    ):
        assert isinstance(label, int) and label in {0, 1}
        assert os.path.isfile(save_path) is False, f"{save_path} already exists"
        assert (
            len(save_path.rsplit(".", 1)) == 2
        ), "path is missing a file extension"

        load_path_queue = Queue()
        for entry in load_paths:
            load_path_queue.put(entry)

        processes = []
        save_file_lock = Lock()
        for _ in range(num_processes):
            p = Process(
                target=supervised_worker,
                args=(load_path_queue, save_file_lock, save_path, label),
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        return TrainingDataset(load_paths=save_path)

    @classmethod
    def build_source_separated(
        cls,
        load_paths: List[dict],
        save_path: str,
        num_processes: int = 1,
    ):
        assert os.path.isfile(save_path) is False, f"{save_path} already exists"
        assert (
            len(save_path.rsplit(".", 1)) == 2
        ), "path is missing a file extension"

        load_path_queue = Queue()
        for entry in load_paths:
            load_path_queue.put(entry)

        processes = []
        save_file_lock = Lock()
        for _ in range(num_processes):
            p = Process(
                target=source_separated_worker,
                args=(load_path_queue, save_file_lock, save_path),
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        return TrainingDataset(load_paths=save_path)
