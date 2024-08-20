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
    def test_supervised(self):
        if os.path.exists("./tests/test_results/supervised_dataset.txt"):
            os.remove("./tests/test_results/supervised_dataset.txt")
        if os.path.exists("./tests/test_results/supervised_dataset_index.txt"):
            os.remove("./tests/test_results/supervised_dataset_index.txt")

        dataset = TrainingDataset.build_supervised(
            load_paths=[
                "./tests/test_data/piano.mp3",
                "./tests/test_data/other.mp3",
            ],
            save_path="./tests/test_results/supervised_dataset.txt",
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

    def test_source_separated(self):
        if os.path.exists("./tests/test_results/source_separated_dataset.txt"):
            os.remove("./tests/test_results/source_separated_dataset.txt")
        if os.path.exists(
            "./tests/test_results/source_separated_dataset_index.txt"
        ):
            os.remove("./tests/test_results/source_separated_dataset_index.txt")

        dataset = TrainingDataset.build_source_separated(
            load_paths=[
                {
                    "piano": "./tests/test_data/etudes_piano.mp3",
                    "other": "./tests/test_data/etudes_other.mp3",
                }
            ],
            save_path="./tests/test_results/source_separated_dataset.txt",
            num_processes=1,
        )

        config = load_config()
        for idx, (wav, label) in enumerate(dataset):
            torchaudio.save(
                f"tests/test_results/{idx}_{label}.wav",
                wav.unsqueeze(0),
                config["audio"]["sample_rate"],
            )


if __name__ == "__main__":
    unittest.main()
