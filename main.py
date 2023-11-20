import argparse
import os
import yaml
import time
import datetime

from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torchvision.utils import make_grid, save_image
from einops import rearrange
from cleanfid import fid

from utils.checkpoint import Checkpoint
from utils.metrics import inception_score
from utils.dist import init_ddp, worker_init_fn
from data.dataset import (
    create_webdataset,
    create_webdataset_metzler,
    collate_variable_length,
)
from models.palette_single import PaletteViewSynthesis
from models.unet_modified import UNet
from models.diffusion_transformer_old import DiT_models


class LrScheduler:
    """Implements a learning rate schedule with warum up and decay"""

    def __init__(self, peak_lr=4e-4, peak_it=10000, decay_rate=0.5, decay_it=100000):
        self.peak_lr = peak_lr
        self.peak_it = peak_it
        self.decay_rate = decay_rate
        self.decay_it = decay_it

    def get_cur_lr(self, it):
        if it < self.peak_it:  # Warmup period
            return self.peak_lr * (it / self.peak_it)
        it_since_peak = it - self.peak_it
        return self.peak_lr * (self.decay_rate ** (it_since_peak / self.decay_it))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="YAML config file")
    parser.add_argument("-gpus", "--gpu_ids", type=str, default=None)
    parser.add_argument("-t", "--train", action="store_true", default=True)
    parser.add_argument("-i", "--inference", action="store_true", default=False)
    parser.add_argument("-r", "--resume", action="store_true", default=False)
    parser.add_argument("-s", "--src_dir", type=str, default=None)
    parser.add_argument("-m", "--metzler", action="store_true", default=False)
    parser.add_argument(
        "--wandb", action="store_true", help="Log run to Weights and Biases."
    )
    args = parser.parse_args()

    log_dir = "./logs"
    is_metzler = args.metzler

    if args.inference or args.resume:
        if args.src_dir is None:
            raise ValueError("Source directory (-s, --src_dir) must be provided.")
        out_dir = Path(args.src_dir)
        tmp_name = os.path.basename(args.src_dir)
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        with open(os.path.join(args.src_dir, "config.yaml")) as f:
            config = yaml.load(f, Loader=yaml.CLoader)

    else:
        exp_name = os.path.splitext(os.path.basename(args.config))[0]
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        out_dir = os.path.join(log_dir, "-".join((now, exp_name)))
        tmp_name = "-".join((now, exp_name))
        with open(args.config, "r") as f:
            config = yaml.load(f, Loader=yaml.CLoader)

    if args.gpu_ids is not None:
        rank, world_size = init_ddp()
        device = torch.device(f"cuda:{rank}")
    else:
        rank, world_size = init_ddp()
        device = torch.device("cpu")

    args.wandb = args.wandb and rank == 0

    max_it = config["model"].get("max_it", 1000000)
    validate_every = config["model"].get("validate_every", 5000)
    checkpoint_every = config["model"].get("checkpoint_every", 10)
    log_every = config["model"].get("log_every", 10)
    masked_cnt = config["model"].get("masked_cnt", 6)

    tmp_dir = os.path.join("/tmp", tmp_name)
    print(tmp_dir)
    # os.makedirs(tmp_dir)

    if world_size > 0:
        batch_size = config["data"]["params"]["batch_size"] // world_size
    else:
        batch_size = config["data"]["params"]["batch_size"]

    # Initialize datasets
    print("Loading training set...")
    # train_dataset = NMRShardedDataset(**config["data"]["params"]["train"]["params"])
    if is_metzler:
        train_dataset = create_webdataset_metzler("/scratch/work/spieglb1/datasets/")
    else:
        train_dataset = create_webdataset(**config["data"]["params"]["train"]["params"])
    print("Training set loaded.")

    print("Loading validation set...")
    # val_dataset = NMRShardedDataset(**config["data"]["params"]["test"]["params"])
    if is_metzler:
        val_dataset = create_webdataset_metzler("/scratch/work/spieglb1/datasets/")
    else:
        val_dataset = create_webdataset(**config["data"]["params"]["test"]["params"])
    print("Validation set loaded.")

    # Initialize data loaders

    num_workers = config["data"]["params"].get("num_workers", 1)
    print(
        f"Initializing datalaoders, using {num_workers} workers per process for data loading."
    )
    train_sampler = val_sampler = None

    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, drop_last=False
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, drop_last=False
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
        worker_init_fn=worker_init_fn,
        # collate_fn=collate_variable_length,
        persistent_workers=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=1,
        sampler=val_sampler,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        # collate_fn=collate_variable_length,
        persistent_workers=True,
    )

    val_vis_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=12,
        worker_init_fn=worker_init_fn,
        # collate_fn=collate_variable_length,
    )
    train_vis_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=12,
        worker_init_fn=worker_init_fn,
        # collate_fn=collate_variable_length,
    )
    val_vis_data = next(iter(val_vis_loader))
    train_vis_data = next(iter(train_vis_loader))

    # Initialize model

    denoise_net = config["model"].get("denoise_net", "unet")
    if denoise_net == "unet":
        denoise_fn = UNet(**config["model"]["denoise_net_params"])
    elif denoise_net == "dit":
        dit_name = config["model"].get("dit_name", "DiT-L/4")
        denoise_fn = DiT_models[dit_name](**config["model"]["denoise_net_params"])
    else:
        raise ValueError("Provided denoising function is not supported!")
    model = PaletteViewSynthesis(
        denoise_fn,
        config["model"]["palette_params"]["beta_schedule"],
    ).to(device)
    model.set_new_noise_schedule(device=device, phase="train")

    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    if device == torch.device("cpu"):
        model = DistributedDataParallel(model)

    peak_it = config.get("lr_warmup", 2500)
    decay_it = config.get("decay_it", 4000000)

    lr_scheduler = LrScheduler(
        peak_lr=1e-4, peak_it=peak_it, decay_it=decay_it, decay_rate=0.16
    )
    optimizer = optim.Adam(model.parameters(), lr=lr_scheduler.get_cur_lr(0))
    checkpoint = Checkpoint(
        out_dir, device=device, config=config, model=model, optimizer=optimizer
    )

    # Try to resume training

    try:
        if os.path.exists(os.path.join(out_dir, f"model.pt")):
            load_dict = checkpoint.load(f"model.pt")
        else:
            load_dict = checkpoint.load("model.pt")
    except FileNotFoundError:
        load_dict = dict()

    epoch_no = load_dict.get("epoch_no", -1)
    it = load_dict.get("it", -1)
    time_elapsed = load_dict.get("t", 0.0)
    run_id = load_dict.get("run_id", None)
    # metric_val_best = load_dict.get("loss_val_best", -model_selection_sign * np.inf)

    # print(
    #    f"Current best validation metric ({model_selection_metric}): {metric_val_best:.8f}."
    # )

    if args.wandb:
        import wandb

        if run_id is None:
            run_id = wandb.util.generate_id()
            print(f"Sampled new wandb run_id {run_id}.")
            wandb.init(
                project="palette-view-synthesis",
                name=f"{exp_name}-{run_id}",
                id=run_id,
                resume=True,
                config=config,
            )
        else:
            print(f"Resuming wandb with existing run_id {run_id}.")
            wandb.init(
                project="palette-view-synthesis",
                id=run_id,
                resume=True,
                config=config,
            )

    if args.inference:
        log_dict = dict()
        model.eval()

        images_path = os.path.join(tmp_dir, f"images-{it}-{now}")
        ground_truth_path = os.path.join(images_path, "ground-truth")
        generated_path = os.path.join(images_path, "generated")
        os.makedirs(images_path)
        os.makedirs(generated_path)
        os.makedirs(ground_truth_path)
        compute_fid = False
        generate = False
        if compute_fid:
            losses = list()
            generated_batches = list()
            ground_truth_batches = list()

            for val_batch in val_loader:
                images = val_batch["view"].to(device)
                cond = val_batch["cond"].to(device)
                # angle = batch["angle"].to(device)
                with torch.no_grad():
                    *_, generated_samples = model.generate(cond)
                    generated_batches.append(generated_samples)
                    ground_truth_batches.append(images)

            cnt = 0
            for gt_batch, generated_batch in zip(
                ground_truth_batches, generated_batches
            ):
                for gt_sample, generated_sample in zip(gt_batch, generated_batch):
                    # print(gt_sample.shape, generated_sample.shape)
                    save_image(
                        gt_sample,
                        os.path.join(ground_truth_path, f"{cnt}.png"),
                    )
                    save_image(
                        generated_sample,
                        os.path.join(generated_path, f"{cnt}.png"),
                    )
                    cnt += 1

            fid_score = fid.compute_fid(ground_truth_path, generated_path)
            print(fid_score)
            log_dict["fid_score"] = fid_score

        if generate:
            print("Running image generation...")
            images = val_vis_data["view"].to(device)
            cond = val_vis_data["cond"].to(device)
            # angle = val_vis_data["angle"].to(device)

            y_t, generated_batch, *_ = model.generate(cond)

            output = torch.cat(
                (
                    torch.clamp(generated_batch, 0, 1),
                    torch.unsqueeze(images, 1),
                    torch.unsqueeze(cond, 1)[:, :, :3, ...],
                ),
                dim=1,
            )
            log_dict["inference_output"] = wandb.Image(
                make_grid(
                    rearrange(output, "b s c h w -> (b s) c h w"),
                    nrow=output.shape[1],
                    scale_each=True,
                ),
                caption="Denoising steps, Target, Input View",
            )
        wandb.log(log_dict)

        exit(0)

    # Training loop
    if args.train:
        model.set_loss(F.mse_loss)
        metric_val_best = float("inf")

        test_eval = False
        compute_fid = False
        look_back = False

        while True:
            epoch_no += 1
            if train_sampler is not None:
                train_sampler.set_epoch(epoch_no)

            for batch in train_loader:
                it += 1
                log_dict = dict()

                if rank == 0:
                    checkpoint_dict = {
                        "epoch_no": epoch_no,
                        "it": it,
                        "t": time_elapsed,
                        "metric_val_best": metric_val_best,
                        "run_id": run_id,
                    }

                    if (
                        (checkpoint_every > 0)
                        and (it % checkpoint_every) == 0
                        and it > 0
                    ):
                        checkpoint.save("model.pt", **checkpoint_dict)

                # Run validation
                if test_eval or (
                    it > 0 and validate_every > 0 and (it % validate_every) == 0
                ):
                    images_path = os.path.join(tmp_dir, f"images-{it}")
                    ground_truth_path = os.path.join(images_path, "ground-truth")
                    generated_path = os.path.join(images_path, "generated")
                    os.makedirs(images_path)
                    os.makedirs(generated_path)
                    os.makedirs(ground_truth_path)
                    model.eval()
                    print("Running evaluation...")

                    if compute_fid:
                        losses = list()
                        generated_batches = list()
                        ground_truth_batches = list()

                        for val_batch in val_loader:
                            images = batch["cond"].to(device)
                            cond = batch["cond"].to(device)
                            angle = batch["angle"].to(device)
                            with torch.no_grad():
                                loss = model(images, y_cond=cond)
                                *_, generated_samples, ground_truth = model.generate(
                                    cond, angle=angle
                                )
                                generated_batches.append(generated_samples)
                                ground_truth_batches.append(ground_truth)

                            losses.append(loss.item())

                        cnt = 0
                        for gt_batch, generated_batch in zip(
                            ground_truth_batches, generated_batches
                        ):
                            for gt_sample, generated_sample in zip(
                                gt_batch, generated_batch
                            ):
                                # print(gt_sample.shape, generated_sample.shape)
                                save_image(
                                    gt_sample,
                                    os.path.join(ground_truth_path, f"{cnt}.png"),
                                )
                                save_image(
                                    generated_sample,
                                    os.path.join(generated_path, f"{cnt}.png"),
                                )
                                cnt += 1

                        fid_score = fid.compute_fid(ground_truth_path, generated_path)
                        log_dict["fid_score"] = fid_score
                        avg_loss = np.mean(losses)
                        log_dict["val_loss"] = avg_loss
                        if avg_loss < metric_val_best:
                            metric_val_best = avg_loss
                            checkpoint.save("model_best.pt")

                    print("Running image generation...")
                    images = val_vis_data["view"].to(device)
                    cond = val_vis_data["cond"].to(device)
                    # angle = val_vis_data["angle"].to(device)

                    y_t, generated_batch, *_ = model.generate(cond)

                    output = torch.cat(
                        (
                            torch.clamp(generated_batch, 0, 1),
                            torch.unsqueeze(images, 1),
                            cond[:, :, :3, ...]
                            # rearrange(
                            #    cond[:, :3, ...], "b (v c) h w -> b v c h w", c=3
                            # ),
                        ),
                        dim=1,
                    )
                    output = torch.nn.utils.rnn.pad_sequence(output, batch_first=True)
                    log_dict["output"] = wandb.Image(
                        make_grid(
                            rearrange(output, "b s c h w -> (b s) c h w"),
                            nrow=output.shape[1],
                            scale_each=True,
                        ),
                        caption="Denoising steps, Target, Input View",
                    )

                new_lr = lr_scheduler.get_cur_lr(it)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = new_lr

                t0 = time.perf_counter()

                images = batch["view"].to(device)
                cond = batch["cond"].to(device)
                # angle = batch["angle"].to(device)

                model.train()
                optimizer.zero_grad()
                loss = model(images, y_cond=cond)
                loss.backward()
                optimizer.step()
                if look_back:
                    model.eval()
                    *_, generated_samples = model.generate(cond)

                    rev_sin_angle = torch.sin(
                        2 * torch.pi - torch.arcsin(cond[:, -2, ...])
                    )[:, None, ...]
                    rev_cos_angle = torch.cos(
                        2 * torch.pi - torch.arccos(cond[:, -1, ...])
                    )[:, None, ...]
                    rev_cond = torch.cat(
                        (generated_samples.squeeze(), rev_sin_angle, rev_cos_angle),
                        dim=1,
                    )
                    model.train()
                    optimizer.zero_grad()
                    loss = model(
                        cond[:, :3, ...], rev_cond
                    )  # cond as images only, generated_samples with angular data
                    optimizer.step()
                time_elapsed += time.perf_counter() - t0

                if log_every > 0 and it % log_every == 0:
                    log_dict["t"] = time_elapsed
                    log_dict["lr"] = new_lr
                    log_dict["loss"] = loss.item()

                if args.wandb and log_dict:
                    wandb.log(log_dict, step=it)

                if it > max_it:
                    print("Maximum iteration count reached.")
                    if rank == 0:
                        checkpoint.save(
                            "model.pt",
                        )
                    exit(0)
