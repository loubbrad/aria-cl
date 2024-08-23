import unittest
import logging
import os
import shutil
import torch
import torchaudio
import matplotlib.pyplot as plt

from ariacl.config import load_config
from ariacl.audio import (
    AudioTransform,
    get_audio_intervals,
    get_wav_segments,
)

CONFIG = load_config()
SAMPLE_RATE = CONFIG["audio"]["sample_rate"]
CHUNK_LEN = CONFIG["audio"]["chunk_len"]

logging.basicConfig(level=logging.INFO)

if os.path.isdir("tests/test_results") is True:
    shutil.rmtree("tests/test_results")

if os.path.isdir("tests/test_results") is False:
    os.mkdir("tests/test_results")


def plot_spec(mel: torch.Tensor, name: str | int):
    # mel tensor dimensions [height, width]
    height, width = mel.shape
    fig_width, fig_height = max(width // 50, 6), max(
        height // 50, 4
    )  # Ensure minimum size
    plt.figure(figsize=(fig_width, fig_height), dpi=100)
    plt.imshow(
        mel, aspect="auto", origin="lower", cmap="viridis", interpolation="none"
    )
    plt.axis("off")
    plt.savefig(f"tests/test_results/{name}.png", dpi=100, bbox_inches="tight")
    plt.close()


class TestSpec(unittest.TestCase):
    def setUp(self):
        self.start = 10000

    def test_spec(self):
        audio_transform = AudioTransform()
        wav, sr = torchaudio.load("./tests/test_data/concerto_piano.mp3")
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE).mean(
            0, keepdim=True
        )[:, self.start : self.start + SAMPLE_RATE * CHUNK_LEN]

        mel = audio_transform.log_mel(wav=wav)
        plot_spec(mel=mel[0], name="spec")

    def test_aug(self):
        audio_transform = AudioTransform()
        wav, sr = torchaudio.load("./tests/test_data/concerto_piano.mp3")
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE).mean(
            0, keepdim=True
        )[:, self.start : self.start + SAMPLE_RATE * CHUNK_LEN]

        mel = audio_transform.forward(wav=wav)
        plot_spec(mel=mel[0], name="aug")


class TestAug(unittest.TestCase):
    def test_aug(self):
        audio_transform = AudioTransform()
        config = load_config()

        for idx, wav in enumerate(
            get_wav_segments(
                audio_path="./tests/test_data/concerto_piano.mp3",
                stride_factor=1,
            )
        ):
            aug_wav = audio_transform.aug_wav(wav.unsqueeze(0))
            torchaudio.save(
                f"tests/test_results/{idx}.wav",
                aug_wav,
                config["audio"]["sample_rate"],
            )


# TODO: Test this when gpu and cpu idle to see which is better (CPU or batched on CUDA)
class TestDetection(unittest.TestCase):
    def test_silence_detection(self):
        for idx, wav in enumerate(
            get_wav_segments(
                audio_path="./tests/test_data/concerto_other.mp3",
                stride_factor=1,
            )
        ):
            print(
                get_audio_intervals(
                    wav,
                    min_window_s=1.0,
                    threshold_db=-20,
                    detect_silent_intervals=False,
                )
            )

            if idx == 9:
                break

    def test_silence_detection_b(self):
        config = load_config()
        batch = torch.stack(
            list(
                get_wav_segments(
                    audio_path="./tests/test_data/concerto_piano.mp3",
                    stride_factor=1,
                )
            )[:100]
        )

        for idx, intervals in enumerate(
            get_audio_intervals(
                batch,
                min_window_s=3.0,
                threshold_db=-20,
                detect_silent_intervals=True,
            )
        ):
            f_name = f"{idx}_0.mp3" if intervals else f"{idx}_1.mp3"
            torchaudio.save(
                f"tests/test_results/" + f_name,
                batch[idx].unsqueeze(0),
                config["audio"]["sample_rate"],
            )


if __name__ == "__main__":
    unittest.main()
