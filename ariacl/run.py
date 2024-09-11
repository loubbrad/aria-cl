import argparse
import os
import glob
import json

from typing import List

from ariacl.data import TrainingDataset
from ariacl.inference import process_files


def _add_supervised_args(parser):
    parser.add_argument(
        "--load_dir",
        type=str,
        required=True,
        help="Directory to load audio files from",
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to save the dataset"
    )
    parser.add_argument(
        "--label",
        type=int,
        choices=[0, 1],
        required=True,
        help="Label for the dataset (0 or 1)",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of processes to use",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.05,
        help="Validation split ratio (0.0 to 1.0)",
    )


def _add_source_separated_args(parser):
    parser.add_argument(
        "--load_dir",
        type=str,
        required=True,
        help="Directory to load audio files from",
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to save the dataset"
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of processes to use",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.05,
        help="Validation split ratio (0.0 to 1.0)",
    )


def _add_process_files_args(parser):
    parser.add_argument(
        "--load_dir",
        type=str,
        required=True,
        help="Directory to load audio files from",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save the results JSON file",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="small",
        help="Model configuration to use (default: small)",
    )


def _check_save_path(save_path: str):
    assert not os.path.exists(
        save_path
    ), f"Save path {save_path} already exists"
    assert (
        len(save_path.rsplit(".", 1)) == 2
    ), "Save path is missing a file extension"
    save_dir = os.path.dirname(save_path)
    assert os.path.isdir(save_dir), f"Directory {save_dir} does not exist"
    assert os.access(
        save_dir, os.W_OK
    ), f"No write permission for directory {save_dir}"


def build_supervised(args):
    _check_save_path(args.save_path)
    assert os.path.isdir(
        args.load_dir
    ), f"Load directory {args.load_dir} does not exist"

    audio_files = glob.glob(
        os.path.join(args.load_dir, "**/*.mp3"), recursive=True
    )
    audio_files.extend(
        glob.glob(os.path.join(args.load_dir, "**/*.wav"), recursive=True)
    )
    if not audio_files:
        print(f"No .mp3 or .wav files found in {args.load_dir}")

    print(f"Found {len(audio_files)} files")

    split_index = int(len(audio_files) * (1 - args.val_split))
    train_files = audio_files[:split_index]
    val_files = audio_files[split_index:]

    train_save_path = args.save_path.replace(".", "_train.")
    print(f"Building train dataset from {len(train_files)} pairs")
    TrainingDataset.build_supervised(
        load_paths=train_files,
        save_path=train_save_path,
        label=int(args.label),
        num_processes=args.num_processes,
    )
    print(f"Train dataset built and saved to {train_save_path}")

    val_save_path = args.save_path.replace(".", "_val.")
    print(f"Building val dataset from {len(val_files)} pairs")
    TrainingDataset.build_supervised(
        load_paths=val_files,
        save_path=val_save_path,
        label=args.label,
        num_processes=args.num_processes,
    )
    print(f"Validation dataset built and saved to {val_save_path}")


def build_source_separated(args):
    _check_save_path(args.save_path)
    assert os.path.isdir(
        args.load_dir
    ), f"Load directory {args.load_dir} does not exist"

    piano_files = glob.glob(
        os.path.join(args.load_dir, "**/*_piano.mp3"), recursive=True
    )
    other_files = glob.glob(
        os.path.join(args.load_dir, "**/*_other.mp3"), recursive=True
    )
    if not piano_files or not other_files:
        print(f"No matching piano and other files found in {args.load_dir}")
        return

    load_paths = []
    for piano_file in piano_files:
        other_file = piano_file.replace("_piano.mp3", "_other.mp3")
        if other_file in other_files:
            load_paths.append({"piano": piano_file, "other": other_file})
    if not load_paths:
        print(
            f"No matching piano and other file pairs found in {args.load_dir}"
        )
        return

    print(f"Found {len(load_paths)} total pairs")

    split_index = int(len(load_paths) * (1 - args.val_split))
    train_paths = load_paths[:split_index]
    val_paths = load_paths[split_index:]

    # Build train dataset
    print(f"Building train dataset from {len(train_paths)} pairs")
    train_save_path = args.save_path.replace(".", "_train.")
    TrainingDataset.build_source_separated(
        load_paths=train_paths,
        save_path=train_save_path,
        num_processes=args.num_processes,
    )
    print(f"Train dataset built and saved to {train_save_path}")

    # Build validation dataset
    val_save_path = args.save_path.replace(".", "_val.")
    print(f"Building val dataset from {len(val_paths)} pairs")
    TrainingDataset.build_source_separated(
        load_paths=val_paths,
        save_path=val_save_path,
        num_processes=args.num_processes,
    )
    print(f"Validation dataset built and saved to {val_save_path}")


def process_mp3_files(args):
    assert os.path.isdir(
        args.load_dir
    ), f"Load directory {args.load_dir} does not exist"
    assert os.path.isfile(
        args.checkpoint_path
    ), f"Checkpoint file {args.checkpoint_path} does not exist"
    assert (
        os.path.isfile(args.save_path) is False
    ), f"File already exists at save location {args.save_path}"

    # Find all MP3 files in the directory
    audio_files = glob.glob(
        os.path.join(args.load_dir, "**/*.mp3"), recursive=True
    )
    if not audio_files:
        print(f"No .mp3 files found in {args.load_dir}")
        return

    print(f"Found {len(audio_files)} MP3 files")

    results = process_files(
        audio_files,
        checkpoint_path=args.checkpoint_path,
        save_path=args.save_path,
    )

    with open(args.save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.save_path}")


def main():
    parser = argparse.ArgumentParser(description="Build training datasets")
    subparsers = parser.add_subparsers(help="sub-command help", dest="command")
    parser_supervised = subparsers.add_parser(
        "build-supervised-dataset", help="Build from supervised dataset"
    )
    parser_source_separated = subparsers.add_parser(
        "build-source-separated-dataset",
        help="Build from source-separated dataset",
    )
    parser_process_files = subparsers.add_parser(
        "process-files", help="Process MP3 files in a directory"
    )
    _add_supervised_args(parser_supervised)
    _add_source_separated_args(parser_source_separated)
    _add_process_files_args(parser_process_files)

    args = parser.parse_args()

    try:
        if args.command == "build-supervised-dataset":
            build_supervised(args)
        elif args.command == "build-source-separated-dataset":
            build_source_separated(args)
        elif args.command == "process-files":
            process_mp3_files(args)
        else:
            parser.print_help()
            print("Unrecognized command")
            exit(1)
    except AssertionError as e:
        print(f"Error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
