import unittest
import logging
import os
import torchaudio

from ariacl.data import TrainingDataset
from ariacl.config import load_config

logging.basicConfig(level=logging.INFO)
if os.path.isdir("tests/test_results") is False:
    os.mkdir("tests/test_results")


class TestDataset(unittest.TestCase):
    def test_build(self):
        if os.path.exists("./tests/test_results/dataset.txt"):
            os.remove("./tests/test_results/dataset.txt")

        dataset = TrainingDataset.build_supervised(
            load_paths=[
                "./tests/test_data/piano.mp3",
                "./tests/test_data/other.mp3",
            ],
            save_path="./tests/test_results/dataset.txt",
            label=0,
            num_processes=1,
        )

        config = load_config()
        for idx, (wav, label) in enumerate(dataset):
            print(wav.shape)
            print(label)
            torchaudio.save(
                f"tests/test_results/{idx}.wav",
                wav.unsqueeze(0),
                config["audio"]["sample_rate"],
            )


if __name__ == "__main__":
    unittest.main()
