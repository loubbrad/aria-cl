import mmap
import os
import io
import base64
import orjson
import torch

from multiprocessing import Queue, Lock, Process
from queue import Empty
from typing import List

from ariacl.audio import get_wav_segments, get_audio_intervals
from ariacl.config import load_config


# TODO: Add librosa silence tagging on this
def supervised_worker(
    load_path_queue: Queue, save_file_lock, save_path: str, label: int
):
    config = load_config()
    num_files_processed = 0
    label_str = base64.b64encode(orjson.dumps(0)).decode("utf-8")
    while True:
        try:
            load_path = load_path_queue.get_nowait()
        except Empty:
            print(f"Worker {os.getpid()} finished")
            return

        for wav in get_wav_segments(
            audio_path=load_path,
            stride_factor=config["data"]["stride_factor"],
        ):
            wav_buffer = io.BytesIO()
            torch.save(wav, wav_buffer)
            wav_buffer.seek(0)
            wav_bytes = wav_buffer.read()
            wav_str = base64.b64encode(wav_bytes).decode("utf-8")

            with save_file_lock:
                with open(save_path, mode="a") as f:
                    f.write(wav_str)
                    f.write("\n")
                    f.write(label_str)
                    f.write("\n")

        num_files_processed += 1
        if num_files_processed % 25 == 0:
            print(f"Worker {os.getpid()} has finished {num_files_processed}")


# TODO: Recordings with lots of background noise are incorrectly tagged
def source_separated_worker(
    load_path_queue: Queue, save_file_lock, save_path: str
):
    config = load_config()
    stride_factor = config["data"]["stride_factor"]
    piano_label = base64.b64encode(orjson.dumps(1)).decode("utf-8")
    non_piano_label = base64.b64encode(orjson.dumps(0)).decode("utf-8")
    num_files_processed = 0

    while True:
        try:
            load_path = load_path_queue.get_nowait()
        except Empty:
            print(f"Worker {os.getpid()} finished")
            return

        for piano_wav, other_wav in zip(
            get_wav_segments(
                audio_path=load_path["piano"],
                stride_factor=stride_factor,
            ),
            get_wav_segments(
                audio_path=load_path["other"],
                stride_factor=stride_factor,
            ),
        ):
            silent_intervals = get_audio_intervals(
                wav=piano_wav,
                min_window_s=config["data"]["piano_detection"][
                    "min_window_silence_s"
                ],
                threshold_db=config["data"]["piano_detection"][
                    "silence_threshold_db"
                ],
                detect_silent_intervals=True,
            )
            non_piano_intervals = get_audio_intervals(
                wav=other_wav,
                min_window_s=config["data"]["piano_detection"][
                    "min_window_other_s"
                ],
                threshold_db=config["data"]["piano_detection"][
                    "noise_threshold_db"
                ],
                detect_silent_intervals=False,
            )

            _label = (
                non_piano_label
                if non_piano_intervals or silent_intervals
                else piano_label
            )

            wav_buffer = io.BytesIO()
            torch.save(piano_wav + other_wav, wav_buffer)
            wav_buffer.seek(0)
            wav_bytes = wav_buffer.read()
            wav_str = base64.b64encode(wav_bytes).decode("utf-8")

            with save_file_lock:
                with open(save_path, mode="a") as f:
                    f.write(wav_str)
                    f.write("\n")
                    f.write(_label)
                    f.write("\n")

        num_files_processed += 1
        if num_files_processed % 25 == 0:
            print(f"Worker {os.getpid()} has finished {num_files_processed}")


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

            index_path = TrainingDataset._get_index_path(load_path=path)
            if os.path.isfile(index_path):
                _index = self._load_index(load_path=index_path)
            else:
                print("Calculating index...")
                _index = self._build_index(mmap_obj)
                print(
                    f"Index of length {len(_index)} calculated, saving to {index_path}"
                )
                self._save_index(index=_index, save_path=index_path)

            self.index.extend(
                [(len(self.file_mmaps) - 1, pos) for pos in _index]
            )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        # Seek position
        file_id, pos = self.index[idx]
        mmap_obj = self.file_mmaps[file_id]
        mmap_obj.seek(pos)

        # Load data from line
        wav = torch.load(
            io.BytesIO(base64.b64decode(mmap_obj.readline())), weights_only=True
        )
        label = orjson.loads(base64.b64decode(mmap_obj.readline()))

        return wav, torch.tensor(label, dtype=torch.float32)

    def close(self):
        for buff in self.file_buffs:
            buff.close()
        for mmap in self.file_mmaps:
            mmap.close()

    def __del__(self):
        self.close()

    def _save_index(self, index: list, save_path: str):
        with open(save_path, "w") as file:
            for idx in index:
                file.write(f"{idx}\n")

    def _load_index(self, load_path: str):
        with open(load_path, "r") as file:
            return [int(line.strip()) for line in file]

    @staticmethod
    def _get_index_path(load_path: str) -> str:
        parts = load_path.rsplit(".", 1)
        if len(parts) == 1:
            return f"{load_path}_index"
        else:
            base, ext = parts
            return f"{base}_index.{ext}"

    def _build_index(self, mmap_obj):
        mmap_obj.seek(0)
        index = []
        pos = 0
        while True:
            pos_buff = pos

            pos = mmap_obj.find(b"\n", pos)
            if pos == -1:
                break
            pos = mmap_obj.find(b"\n", pos + 1)
            if pos == -1:
                break

            index.append(pos_buff)
            pos += 1

        return index

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

        index_path = TrainingDataset._get_index_path(load_path=save_path)
        if os.path.isfile(index_path):
            print(f"Removing existing index file at {index_path}")
            os.remove(index_path)

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

        # Wait for all processes to complete
        for p in processes:
            p.join()

        # Create index by loading object
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

        index_path = TrainingDataset._get_index_path(load_path=save_path)
        if os.path.isfile(index_path):
            print(f"Removing existing index file at {index_path}")
            os.remove(index_path)

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

        # Wait for all processes to complete
        for p in processes:
            p.join()

        # Create index by loading object
        return TrainingDataset(load_paths=save_path)
