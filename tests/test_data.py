import unittest
import random
import logging
import os
import shutil
import torchaudio

from ariacl.data import TrainingDataset
from ariacl.config import load_config

logging.basicConfig(level=logging.INFO)

if os.path.isdir("tests/test_results") is True:
    shutil.rmtree("tests/test_results")

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
                "./tests/test_data/concerto_piano.mp3",
                "./tests/test_data/etudes_piano.mp3",
            ],
            save_path="./tests/test_results/supervised_dataset.txt",
            label=1,
            num_processes=1,
        )

        config = load_config()
        for idx, (wav, label) in enumerate(dataset):
            torchaudio.save(
                f"tests/test_results/{idx}_{label}.wav",
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
                    "piano": "./tests/test_data/concerto_piano.mp3",
                    "other": "./tests/test_data/concerto_other.mp3",
                },
                {
                    "piano": "./tests/test_data/etudes_piano.mp3",
                    "other": "./tests/test_data/etudes_other.mp3",
                },
                {
                    "piano": "./tests/test_data/old_piano.mp3",
                    "other": "./tests/test_data/old_other.mp3",
                },
            ],
            save_path="./tests/test_results/source_separated_dataset.txt",
            num_processes=1,
        )

        config = load_config()
        for idx, (wav, label) in enumerate(dataset):
            torchaudio.save(
                f"tests/test_results/{idx}_{label}.wav",
                wav.unsqueeze(0).cpu(),
                config["audio"]["sample_rate"],
            )

    def test_labels(self):
        dataset = TrainingDataset(
            [
                "/mnt/ssd1/aria-cl/giantmidi_train.txt",
                "/mnt/ssd1/aria-cl/jazz_trio_train.txt",
                "/mnt/ssd1/aria-cl/score_4_train.txt",
                "/mnt/ssd1/aria-cl/maestro_train.txt",
                "/mnt/ssd1/aria-cl/synth_train.txt",
                "/mnt/ssd1/aria-cl/keyboard_non_piano_train.txt",
            ]
        )
        config = load_config()

        for idx in range(50):
            _idx = random.randint(0, len(dataset))
            wav, label = dataset[_idx]

            torchaudio.save(
                f"tests/test_results/{idx}_{label}.wav",
                wav.unsqueeze(0).cpu(),
                config["audio"]["sample_rate"],
            )


if __name__ == "__main__":
    unittest.main()
