import unittest
import logging
import os
import torch
import torchaudio
import matplotlib.pyplot as plt

from ariacl.config import load_config
from ariacl.data import AudioTransform

CONFIG = load_config()
SAMPLE_RATE = CONFIG["audio"]["sample_rate"]
CHUNK_LEN = CONFIG["audio"]["chunk_len"]

logging.basicConfig(level=logging.INFO)
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
        wav, sr = torchaudio.load("./tests/test_data/piano.mp3")
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE).mean(
            0, keepdim=True
        )[:, self.start : self.start + SAMPLE_RATE * CHUNK_LEN]

        mel = audio_transform.log_mel(wav=wav)
        plot_spec(mel=mel[0], name="spec")

    def test_aug(self):
        audio_transform = AudioTransform()
        wav, sr = torchaudio.load("./tests/test_data/piano.mp3")
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE).mean(
            0, keepdim=True
        )[:, self.start : self.start + SAMPLE_RATE * CHUNK_LEN]

        mel = audio_transform.forward(wav=wav)
        plot_spec(mel=mel[0], name="aug")


if __name__ == "__main__":
    unittest.main()
