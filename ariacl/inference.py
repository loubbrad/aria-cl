import os
import json
import torch
import torch.multiprocessing as mp
import torch._dynamo.config
import torch._inductor.config

from typing import List
from tqdm import tqdm
from functools import wraps
from safetensors import safe_open
from queue import Empty

from ariacl.model import MelSpectrogramCNN, ModelConfig
from ariacl.audio import get_wav_segments_b_stream, AudioTransform
from ariacl.config import load_config, load_model_config

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True


def _get_model(checkpoint_path, model_config="large"):
    model_config = ModelConfig(**load_model_config(model_config))
    model = MelSpectrogramCNN(model_config)

    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        state_dict = {key[10:]: f.get_tensor(key) for key in f.keys()}

    model.load_state_dict(state_dict)
    model.eval()

    return model


@torch.inference_mode()
def compiled_forward(model: MelSpectrogramCNN, mel_specs: torch.Tensor):
    return model.forward(mel_specs)


@torch.inference_mode()
def get_scores_from_batch(mel_specs: torch.Tensor, model: MelSpectrogramCNN):
    scores = torch.nn.functional.sigmoid(
        compiled_forward(model, mel_specs.unsqueeze(1))
    )

    # scores = torch.nn.functional.conv1d(
    #     scores.view(1, 1, -1),
    #     torch.ones(1, 1, 3).cuda() / 3,  # Changed from 5 to 3
    #     padding="same",
    # ).view(-1)
    # scores[0] *= 3 / 2  # Changed from 3/2 to 2
    # scores[-1] *= 3 / 2  # Changed from 3/2 to 2

    return scores.view(-1)


# TODO: Add different config setting for non-piano segments, and avg segment-scores
def get_segments_from_audio(
    audio_path: str,
    config: dict,
    gpu_task_queue=None,
    gpu_result_queue=None,
    model: MelSpectrogramCNN | None = None,
):
    scores = torch.tensor([], device="cpu")
    batched_wav_segments_itt = get_wav_segments_b_stream(
        audio_path=audio_path,
        stride_factor=5,
        batch_size=config["inference"]["batch_size"],
        device="cpu",
    )

    for batch in batched_wav_segments_itt:
        if model:
            _scores = get_scores_from_gpu_model(batch=batch, model=model)
        elif gpu_task_queue and gpu_result_queue is not None:
            _scores = get_scores_from_gpu_worker(
                batch=batch,
                gpu_task_queue=gpu_task_queue,
                gpu_result_queue=gpu_result_queue,
            )
        else:
            raise ValueError("Must either provide gpu queues or model.")

        scores = torch.cat(
            (
                scores,
                _scores,
            ),
            dim=0,
        )

    total_avg_score = scores.mean().item()
    # if total_avg_score < config["inference"]["min_avg_score"]:
    #     return [], total_avg_score

    padded = torch.cat(
        [
            torch.tensor([True], device=scores.device),
            scores > config["inference"]["threshold_score"],
            torch.tensor([True], device=scores.device),
        ]
    )

    edges = torch.diff(padded.int())
    non_piano_segment_starts = torch.where(edges == -1)[0]
    non_piano_segment_ends = torch.where(edges == 1)[0]
    non_piano_segment_lengths = (
        non_piano_segment_ends - non_piano_segment_starts
    )
    non_piano_segments_valid = torch.tensor(
        [
            (length >= config["inference"]["min_invalid_window_s"])
            or (start == 0)
            or (end == len(scores))  # Segment includes the end
            for start, end, length in zip(
                non_piano_segment_starts.tolist(),
                non_piano_segment_ends.tolist(),
                non_piano_segment_lengths.tolist(),
            )
        ]
    )

    piano_segments = []
    segment_start_buffer = 0
    for non_piano_start, non_piano_end, valid in zip(
        non_piano_segment_starts.tolist(),
        non_piano_segment_ends.tolist(),
        non_piano_segments_valid.tolist(),
    ):
        if valid is True:
            if (
                non_piano_start - segment_start_buffer
                >= config["inference"]["min_valid_window_s"]
            ):
                # Add buffer to end of segment
                _start = (
                    0 if segment_start_buffer == 0 else segment_start_buffer + 2
                )
                _end = non_piano_start + 3
                _segment_avg_score = torch.mean(scores[_start:_end]).item()

                if (
                    total_avg_score
                    >= config["inference"]["min_avg_segment_score"]
                ):
                    piano_segments.append((_start, _end, _segment_avg_score))

            segment_start_buffer = non_piano_end

    if (
        len(scores) - segment_start_buffer
        >= config["inference"]["min_valid_window_s"]
    ):
        _start = 0 if segment_start_buffer == 0 else segment_start_buffer + 2
        _end = len(scores) + 4
        _segment_avg_score = torch.mean(scores[_start:_end]).item()
        if total_avg_score >= config["inference"]["min_avg_segment_score"]:
            piano_segments.append((_start, _end, _segment_avg_score))

    return piano_segments, total_avg_score


def get_scores_from_gpu_model(batch: torch.Tensor, model: MelSpectrogramCNN):
    audio_transform = AudioTransform().cuda()

    batch = batch.cuda()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        mel_specs = audio_transform.log_mel(batch).to(torch.bfloat16)
        scores = get_scores_from_batch(mel_specs=mel_specs, model=model).cpu()

    return scores


def get_scores_from_gpu_worker(batch, gpu_task_queue, gpu_result_queue):
    pid = os.getpid()
    gpu_task_queue.put((batch, pid))

    while True:
        try:
            _score, _pid = gpu_result_queue.get()
        except Exception as e:
            pass
        else:
            if _pid == pid:
                if _score == None:
                    raise Exception("Failure sentinel seen")
                return _score
            else:
                gpu_result_queue.put((_score.clone(), _pid))


def gpu_worker(gpu_task_queue, gpu_result_queue, checkpoint_path):
    audio_transform = AudioTransform().cuda()
    model = (
        _get_model(checkpoint_path=checkpoint_path)
        .cuda()
        .eval()
        .to(torch.bfloat16)
    )

    global compiled_forward
    compiled_forward = torch.compile(
        compiled_forward,
        mode="reduce-overhead",
        fullgraph=True,
    )

    while True:
        try:
            batch, pid = gpu_task_queue.get(timeout=30)
        except Empty:
            break

        try:
            batch = batch.cuda()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                mel_specs = audio_transform.log_mel(batch).to(torch.bfloat16)
                scores = get_scores_from_batch(mel_specs=mel_specs, model=model)

        except Exception as e:
            print("GPU ERROR:", e)
            gpu_result_queue.put((None, pid))
        else:
            gpu_result_queue.put((scores.cpu(), pid))


def worker(audio_path_queue, gpu_task_queue, gpu_result_queue, segment_queue):
    config = load_config()

    while True:
        try:
            path = audio_path_queue.get(timeout=1)
        except Empty:
            break

        try:
            segments, avg_score = get_segments_from_audio(
                audio_path=path,
                config=config,
                gpu_task_queue=gpu_task_queue,
                gpu_result_queue=gpu_result_queue,
            )
        except Exception as e:
            print(f"Failed to process {path}: {e}")
        else:
            segment_queue.put(
                {"path": path, "segments": segments, "avg_score": avg_score}
            )


# This is a REALLY dumb way of doing this, but the bottleneck seems to
# be loading the batches (cpu), so...
def process_files(
    audio_paths: List[str], checkpoint_path: str, save_path: str | None = None
):
    assert torch.cuda.is_available(), "CUDA device not found"
    for _path in audio_paths:
        assert os.path.isfile(_path), f"File {_path} not found"

    mp.set_start_method("spawn", force=True)
    num_cpu_workers = 8

    audio_path_queue = mp.Queue()
    gpu_task_queue = mp.Queue()
    gpu_result_queue = mp.Queue()
    segment_queue = mp.Queue()

    for path in audio_paths:
        audio_path_queue.put(path)

    gpu_process = mp.Process(
        target=gpu_worker,
        args=(gpu_task_queue, gpu_result_queue, checkpoint_path),
    )
    gpu_process.start()

    cpu_processes = []
    for _ in range(num_cpu_workers):
        p = mp.Process(
            target=worker,
            args=(
                audio_path_queue,
                gpu_task_queue,
                gpu_result_queue,
                segment_queue,
            ),
        )
        p.start()
        cpu_processes.append(p)

    segments_by_path = {}
    cnt = 0
    with tqdm(total=len(audio_paths), desc="Processing audio files") as pbar:
        while len(segments_by_path) < len(audio_paths):
            try:
                result = segment_queue.get(timeout=0.1)
                segments_by_path[result["path"]] = {
                    "segments": result["segments"],
                    "avg_score": result["avg_score"],
                }
                pbar.update(1)
                pbar.set_postfix({"Current file": result["path"]})
                cnt += 1

                if cnt % 500 == 0:
                    if save_path is not None:
                        print(f"Saving {cnt} results to {save_path}")
                        with open(save_path, "w") as f:
                            json.dump(segments_by_path, f, indent=2)

            except Empty:
                if all(not p.is_alive() for p in cpu_processes):
                    break
                else:
                    continue

    for p in cpu_processes:
        p.join()

    gpu_process.terminate()
    gpu_process.join()

    return segments_by_path
