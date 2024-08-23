import unittest
import logging
import os
import random
import shutil
import torchaudio
import torch

from safetensors import safe_open

from ariacl.model import MelSpectrogramCNN, ModelConfig
from ariacl.audio import AudioTransform
from ariacl.data import TrainingDataset
from ariacl.config import load_config, load_model_config

logging.basicConfig(level=logging.INFO)
if os.path.isdir("tests/test_results") is True:
    shutil.rmtree("tests/test_results")

if os.path.isdir("tests/test_results") is False:
    os.mkdir("tests/test_results")


class TestInference(unittest.TestCase):
    @torch.no_grad()
    def _model_forward(self, mel):
        model_config = ModelConfig(**load_model_config("small"))
        model = MelSpectrogramCNN(model_config)

        with safe_open(
            "/home/loubb/work/aria-cl/experiments/0/checkpoints/epoch0_step8000/model.safetensors",
            framework="pt",
            device="cpu",
        ) as f:
            state_dict = {key[10:]: f.get_tensor(key) for key in f.keys()}

        model.load_state_dict(state_dict)
        model.eval()

        return model.forward(mel)

    def test_pred(self):
        dataset = TrainingDataset("/mnt/ssd1/aria-cl/score_4_val.txt")
        random.shuffle(dataset.index)
        dataset.index = dataset.index[:200]

        config = load_config()
        audio_transform = AudioTransform()
        wav = torch.stack([wav for wav, _ in dataset], dim=0)
        mel = audio_transform.log_mel(wav).unsqueeze(1)
        label = torch.stack([label for _, label in dataset], dim=0)
        logits = self._model_forward(mel)
        pred_label = [
            round(_.item(), 3) for _ in torch.nn.functional.sigmoid(logits)
        ]

        for idx in range(mel.shape[0]):
            torchaudio.save(
                f"tests/test_results/{idx}_{label[idx]}_{pred_label[idx]}.wav",
                wav[idx : idx + 1, :],
                config["audio"]["sample_rate"],
            )


if __name__ == "__main__":
    unittest.main()
