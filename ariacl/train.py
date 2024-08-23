import os
import time
import csv
import traceback
import argparse
import logging
import torch
import accelerate

from torch import nn as nn
from torch.utils.data import DataLoader

from accelerate.logging import get_logger
from logging.handlers import RotatingFileHandler
from tqdm import tqdm

from ariacl.model import ModelConfig, MelSpectrogramCNN
from ariacl.config import load_model_config
from ariacl.audio import AudioTransform
from ariacl.data import TrainingDataset

GRADIENT_ACC_STEPS = 1

# ----- USAGE -----
#
# This script is meant to be run using the huggingface accelerate cli, see:
#
# https://huggingface.co/docs/accelerate/basic_tutorials/launch
# https://huggingface.co/docs/accelerate/package_reference/cli
#
# For example usage you could run the training script with:
#
# accelerate launch [arguments] ariacl/train.py \
#   small \
#   data/train.jsonl \
#   data/val.jsonl \
#   -epochs 10 \
#   -bs 4 \
#   -workers 8


def setup_logger(project_dir: str):
    # Get logger and reset all handlers
    logger = logging.getLogger(__name__)
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s: [%(levelname)s] %(message)s",
    )

    fh = RotatingFileHandler(
        os.path.join(project_dir, "logs.txt"), backupCount=5, maxBytes=1024**3
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return get_logger(__name__)  # using accelerate.logging.get_logger()


def setup_project_dir(project_dir: str | None):
    if not project_dir:
        # Create project directory
        if not os.path.isdir("./experiments"):
            os.mkdir("./experiments")

        project_dirs = [
            _dir
            for _dir in os.listdir("./experiments")
            if os.path.isdir(os.path.join("experiments", _dir))
        ]

        ind = 0
        while True:
            if str(ind) not in project_dirs:
                break
            else:
                ind += 1

        project_dir_abs = os.path.abspath(os.path.join("experiments", str(ind)))
        assert not os.path.isdir(project_dir_abs)
        os.mkdir(project_dir_abs)

    elif project_dir:
        # Run checks on project directory
        if os.path.isdir(project_dir):
            assert (
                len(os.listdir(project_dir)) == 0
            ), "Provided project directory is not empty"
            project_dir_abs = os.path.abspath(project_dir)
        elif os.path.isfile(project_dir):
            raise FileExistsError(
                "The provided path points toward an existing file"
            )
        else:
            try:
                os.mkdir(project_dir)
            except Exception as e:
                raise e(f"Failed to create project directory at {project_dir}")
        project_dir_abs = os.path.abspath(project_dir)

    os.mkdir(os.path.join(project_dir_abs, "checkpoints"))

    return project_dir_abs


def _get_optim(
    lr: float,
    model: nn.Module,
    num_epochs: int,
    steps_per_epoch: int,
    warmup: int = 100,
    end_ratio: int = 0.1,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        eps=1e-6,
    )

    warmup_lrs = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1,
        total_iters=warmup,
    )
    linear_decay_lrs = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1,
        end_factor=end_ratio,
        total_iters=(num_epochs * steps_per_epoch) - warmup,
    )

    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_lrs, linear_decay_lrs],
        milestones=[warmup],
    )

    return optimizer, lr_scheduler


def get_optim(
    model: nn.Module,
    num_epochs: int,
    steps_per_epoch: int,
):
    LR = 3e-4
    END_RATIO = 0.1
    WARMUP_STEPS = 1000

    return _get_optim(
        lr=LR,
        model=model,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        warmup=WARMUP_STEPS,
        end_ratio=END_RATIO,
    )


def get_dataloaders(
    train_data_paths: str,
    val_data_path: str,
    batch_size: int,
    num_workers: int,
):
    logger = get_logger(__name__)
    train_dataset = TrainingDataset(load_paths=train_data_paths)
    val_dataset = TrainingDataset(load_paths=val_data_path)
    logger.info(
        f"Loaded datasets with length: train={len(train_dataset)}; val={len(val_dataset)}"
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    return train_dataloader, val_dataloader


def _train(
    epochs: int,
    accelerator: accelerate.Accelerator,
    model: MelSpectrogramCNN,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    audio_transform: AudioTransform,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    steps_per_checkpoint: int | None = None,
    resume_step: int | None = None,
    resume_epoch: int | None = None,
    project_dir: str | None = None,
):
    def make_checkpoint(_accelerator, _epoch: int, _step: int):
        checkpoint_dir = os.path.join(
            project_dir,
            "checkpoints",
            f"epoch{_epoch}_step{_step}",
        )

        logger.info(
            f"EPOCH {_epoch}/{epochs + start_epoch}: Saving checkpoint - {checkpoint_dir}"
        )
        _accelerator.save_state(checkpoint_dir)

    # This is all slightly messy as train_loop and val_loop make use of the
    # variables in the wider scope. Perhaps refactor this at some point.
    def train_loop(
        dataloader: DataLoader,
        _epoch: int,
        _resume_step: int = 0,
    ):
        avg_train_loss = 0
        trailing_loss = 0
        loss_buffer = []

        try:
            lr_for_print = "{:.2e}".format(scheduler.get_last_lr()[0])
        except Exception:
            pass
        else:
            lr_for_print = "{:.2e}".format(optimizer.param_groups[-1]["lr"])

        model.train()
        grad_norm = 0.0
        for __step, batch in (
            pbar := tqdm(
                enumerate(dataloader),
                total=len(dataloader) + _resume_step,
                initial=_resume_step,
                leave=False,
            )
        ):
            with accelerator.accumulate(model):
                step = __step + _resume_step + 1
                wav, tgt = batch

                with torch.no_grad():
                    mel = audio_transform.log_mel(wav)
                    mel = mel.unsqueeze(1)

                logits = model(mel)  # (b_sz, 1)
                loss = loss_fn(logits, tgt.view(-1, 1))

                # Calculate statistics
                loss_buffer.append(accelerator.gather(loss).mean(dim=0).item())
                trailing_loss = sum(loss_buffer[-TRAILING_LOSS_STEPS:]) / len(
                    loss_buffer[-TRAILING_LOSS_STEPS:]
                )
                avg_train_loss = sum(loss_buffer) / len(loss_buffer)

                # Backwards step
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # Need for CNN?
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), 1.0
                    ).item()
                optimizer.step()
                optimizer.zero_grad()

                # Logging
                logger.debug(
                    f"EPOCH {_epoch} STEP {step}: "
                    f"lr={lr_for_print}, "
                    f"loss={round(loss_buffer[-1], 4)}, "
                    f"trailing_loss={round(trailing_loss, 4)}, "
                    f"average_loss={round(avg_train_loss, 4)}, "
                    f"grad_norm={round(grad_norm, 4)}"
                )
                if accelerator.is_main_process:
                    loss_writer.writerow([_epoch, step, loss_buffer[-1]])

                pbar.set_postfix_str(
                    f"lr={lr_for_print}, "
                    f"loss={round(loss_buffer[-1], 4)}, "
                    f"trailing={round(trailing_loss, 4)}, "
                    f"grad_norm={round(grad_norm, 4)}"
                )

                if scheduler:
                    scheduler.step()
                    lr_for_print = "{:.2e}".format(scheduler.get_last_lr()[0])

                if steps_per_checkpoint:
                    if step % steps_per_checkpoint == 0:
                        make_checkpoint(
                            _accelerator=accelerator,
                            _epoch=_epoch,
                            _step=step,
                        )

        logger.info(
            f"EPOCH {_epoch}/{epochs + start_epoch}: Finished training - "
            f"average_loss={round(avg_train_loss, 4)}"
        )

        return avg_train_loss

    @torch.no_grad()
    def val_loop(dataloader, _epoch: int, aug: bool):
        loss_buffer = []
        model.eval()
        for step, batch in (
            pbar := tqdm(
                enumerate(dataloader),
                total=len(dataloader),
                leave=False,
            )
        ):
            wav, tgt = batch

            if aug == False:
                mel = audio_transform.log_mel(wav)
            elif aug == True:
                mel = audio_transform.forward(wav)

            mel = mel.unsqueeze(1)
            logits = model(mel)
            loss = loss_fn(logits, tgt.view(-1, 1))

            # Logging
            loss_buffer.append(accelerator.gather(loss).mean(dim=0).item())
            avg_val_loss = sum(loss_buffer) / len(loss_buffer)
            pbar.set_postfix_str(f"average_loss={round(avg_val_loss, 4)}")

        # EPOCH
        logger.info(
            f"EPOCH {_epoch}/{epochs + start_epoch}: Finished evaluation "
            f"{'(aug)' if aug is True else ''} - "
            f"average_loss={round(avg_val_loss, 4)}"
        )

        return avg_val_loss

    if steps_per_checkpoint:
        assert (
            steps_per_checkpoint > 1
        ), "Invalid checkpoint mode value (too small)"

    TRAILING_LOSS_STEPS = 100
    logger = get_logger(__name__)  # Accelerate logger
    loss_fn = nn.BCEWithLogitsLoss()

    logger.info(
        f"Model has "
        f"{'{:,}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad))} "
        "parameters"
    )

    if accelerator.is_main_process:
        loss_csv = open(os.path.join(project_dir, "loss.csv"), "w")
        loss_writer = csv.writer(loss_csv)
        loss_writer.writerow(["epoch", "step", "loss"])
        epoch_csv = open(os.path.join(project_dir, "epoch.csv"), "w")
        epoch_writer = csv.writer(epoch_csv)
        epoch_writer.writerow(
            ["epoch", "avg_train_loss", "avg_val_loss", "avg_val_loss_aug"]
        )

    if resume_epoch is not None:
        start_epoch = resume_epoch + 1
    else:
        start_epoch = 0

    if resume_step is not None:
        assert resume_epoch is not None, "Must provide resume epoch"
        logger.info(
            f"Resuming training from step {resume_step} - logging as EPOCH {resume_epoch}"
        )
        skipped_dataloader = accelerator.skip_first_batches(
            dataloader=train_dataloader,
            num_batches=resume_step,
        )

        avg_train_loss = train_loop(
            dataloader=skipped_dataloader,
            _epoch=resume_epoch,
            _resume_step=resume_step,
        )
        avg_val_loss = val_loop(
            dataloader=val_dataloader, _epoch=resume_epoch, aug=False
        )
        avg_val_loss_aug = val_loop(
            dataloader=val_dataloader, _epoch=resume_epoch, aug=True
        )
        if accelerator.is_main_process:
            epoch_writer.writerow(
                [resume_epoch, avg_train_loss, avg_val_loss, avg_val_loss_aug]
            )
            epoch_csv.flush()
            make_checkpoint(
                _accelerator=accelerator, _epoch=start_epoch, _step=0
            )

    for epoch in range(start_epoch, epochs + start_epoch):
        try:
            avg_train_loss = train_loop(
                dataloader=train_dataloader, _epoch=epoch
            )
            avg_val_loss = val_loop(
                dataloader=val_dataloader, _epoch=epoch, aug=False
            )
            avg_val_loss_aug = val_loop(
                dataloader=val_dataloader, _epoch=epoch, aug=True
            )
            if accelerator.is_main_process:
                epoch_writer.writerow(
                    [epoch, avg_train_loss, avg_val_loss, avg_val_loss_aug]
                )
                epoch_csv.flush()
                make_checkpoint(
                    _accelerator=accelerator, _epoch=epoch + 1, _step=0
                )
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise e

    logging.shutdown()
    if accelerator.is_main_process:
        loss_csv.close()
        epoch_csv.close()


def train(
    model_name: str,
    train_data_paths: str,
    val_data_path: str,
    num_workers: int,
    batch_size: int,
    epochs: int,
    steps_per_checkpoint: int | None = None,
    project_dir: str = None,
):
    assert 0 < num_workers <= 128, "Too many workers"
    assert epochs > 0, "Invalid number of epochs"
    assert batch_size > 0, "Invalid batch size"
    assert torch.cuda.is_available() is True, "CUDA not available"
    for _path in train_data_paths:
        assert os.path.isfile(_path), f"No file found at {_path}"
    assert os.path.isfile(val_data_path), f"No file found at {val_data_path}"

    accelerator = accelerate.Accelerator(
        project_dir=project_dir, gradient_accumulation_steps=GRADIENT_ACC_STEPS
    )

    if accelerator.is_main_process:
        project_dir = setup_project_dir(project_dir)
        logger = setup_logger(project_dir)

    logger = get_logger(__name__)
    logger.info(f"Using project directory {project_dir}")
    logger.info(
        f"Using training config: "
        f"model_name={model_name}, "
        f"epochs={epochs}, "
        f"num_proc={accelerator.num_processes}, "
        f"batch_size={batch_size}, "
        f"grad_acc_steps={GRADIENT_ACC_STEPS}, "
        f"num_workers={num_workers}"
    )

    # Init model
    model_config = ModelConfig(**load_model_config(model_name))
    model = MelSpectrogramCNN(model_config)
    model = torch.compile(model)
    audio_transform = AudioTransform().to(accelerator.device)
    logger.info(f"Loaded model with config: {load_model_config(model_name)}")
    logger.info(f"Loaded transform with config: {audio_transform.get_params()}")

    train_dataloader, val_dataloader = get_dataloaders(
        train_data_paths=train_data_paths,
        val_data_path=val_data_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    optimizer, scheduler = get_optim(
        model,
        num_epochs=epochs,
        steps_per_epoch=len(train_dataloader) // GRADIENT_ACC_STEPS,
    )

    (
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
    ) = accelerator.prepare(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
    )

    _train(
        epochs=epochs,
        accelerator=accelerator,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        audio_transform=audio_transform,
        optimizer=optimizer,
        scheduler=scheduler,
        steps_per_checkpoint=steps_per_checkpoint,
        project_dir=project_dir,
    )


if __name__ == "__main__":
    # Nested argparse inspired by - https://shorturl.at/kuKW0
    argp = argparse.ArgumentParser(usage="python ariacl/train.py [<args>]")

    argp.add_argument("model", help="name of model config file")
    argp.add_argument("-train_data", nargs="+", help="paths to train data")
    argp.add_argument("-val_data", help="path to val dir")
    argp.add_argument("-epochs", help="train epochs", type=int, required=True)
    argp.add_argument("-bs", help="batch size", type=int, default=32)
    argp.add_argument("-workers", help="number workers", type=int, default=1)
    argp.add_argument("-pdir", help="project dir", type=str, required=False)
    argp.add_argument(
        "-spc", help="steps per checkpoint", type=int, required=False
    )

    args = argp.parse_args()
    train(
        model_name=args.model,
        train_data_paths=args.train_data,
        val_data_path=args.val_data,
        num_workers=args.workers,
        batch_size=args.bs,
        epochs=args.epochs,
        steps_per_checkpoint=args.spc,
        project_dir=args.pdir,
    )
