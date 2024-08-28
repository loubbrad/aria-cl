import os
import torch
import torch._dynamo.config
import torch._inductor.config

from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Iterator
from functools import wraps

from ariacl.model import MelSpectrogramCNN
from ariacl.audio import get_wav_segments_b, AudioTransform
from ariacl.config import load_config

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True


def optional_bf16_autocast(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_bf16_supported():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return func(*args, **kwargs)
        else:
            with torch.autocast("cuda", dtype=torch.float):
                return func(*args, **kwargs)

    return wrapper


@torch.no_grad()
def compiled_forward(model: MelSpectrogramCNN, mel_specs: torch.Tensor):
    return model.forward(mel_specs)


@optional_bf16_autocast
def get_scores_from_batch(mel_specs: torch.Tensor, model: MelSpectrogramCNN):
    scores = torch.nn.functional.sigmoid(
        compiled_forward(model, mel_specs.unsqueeze(1))
    )
    scores = torch.nn.functional.conv1d(
        scores.view(1, 1, -1),
        torch.ones(1, 1, 5).cuda() / 5,
        padding="same",
    ).view(-1)
    scores[0] *= 3 / 2
    scores[-1] *= 3 / 2

    return scores


# TODO: Tune exact times for start and end of segment
def get_segments_from_audio(
    model: MelSpectrogramCNN,
    audio_transform: AudioTransform,
    config: dict,
    audio_path: str | None = None,
    wav_segments: torch.Tensor | None = None,
):
    batch_size = config["inference"]["batch_size"]
    if wav_segments is None:
        assert audio_path is not None, "must provide audio_path or wav segments"
        wav_segments = get_wav_segments_b(
            audio_path=audio_path,
            stride_factor=5,
            device="cpu",
        )
    num_wav_segments, _ = wav_segments.shape

    scores = torch.tensor([]).cuda()
    for idx in range(0, num_wav_segments, batch_size):
        mel_specs = audio_transform.log_mel(
            wav_segments[idx : idx + batch_size].cuda()
        )
        scores = torch.cat(
            (
                scores,
                get_scores_from_batch(
                    mel_specs=mel_specs, model=model
                ).squeeze(),
            ),
            dim=0,
        )

    avg_score = scores.mean().item()
    if avg_score < config["inference"]["min_avg_score"]:
        return []

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
    non_piano_segments_valid = (
        non_piano_segment_lengths >= config["inference"]["min_invalid_window_s"]
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
                # Add three second buffer to end of segment
                piano_segments.append(
                    (segment_start_buffer, non_piano_start + 2)
                )

            segment_start_buffer = non_piano_end

    if (
        len(scores) - segment_start_buffer
        >= config["inference"]["min_valid_window_s"]
    ):
        piano_segments.append((segment_start_buffer, len(scores)))

    return piano_segments


def load_audio(audio_path: str) -> Tuple[torch.Tensor, str]:
    wav_segments = get_wav_segments_b(
        audio_path=audio_path,
        stride_factor=5,
        device="cpu",
    )
    return wav_segments, audio_path


def parallel_audio_segments_generator(
    audio_paths: List[str], workers: int = 2
) -> Iterator[Tuple[torch.Tensor, str]]:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(load_audio, audio_paths):
            yield result


def process_files(audio_paths: List[str], model: MelSpectrogramCNN):
    assert torch.cuda.is_available(), "CUDA device not found"
    for _path in audio_paths:
        assert os.path.isfile(_path), f"File {_path} not found"

    config = load_config()
    model.cuda()
    audio_transform = AudioTransform().cuda()

    # global compiled_forward
    # compiled_forward = torch.compile(
    #     compiled_forward,
    #     mode="reduce-overhead",
    #     fullgraph=True,
    # )

    results_by_path = {}
    num_processed = 0
    for wav_segments, _audio_path in parallel_audio_segments_generator(
        audio_paths=audio_paths,
    ):
        with torch.no_grad():
            segments = get_segments_from_audio(
                wav_segments=wav_segments,
                model=model,
                audio_transform=audio_transform,
                config=config,
            )
        results_by_path[_audio_path] = segments
        num_processed += 1

        if num_processed % 5 == 0:
            print(f"Finished {num_processed}/{len(audio_paths)} files.")

    return results_by_path
