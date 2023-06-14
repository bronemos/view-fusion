import argparse
import os
import yaml
import time

import numpy as np

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torchvision.utils import make_grid
from einops import rearrange

from utils.trainer import Trainer
from utils.checkpoint import Checkpoint
from utils.metrics import inception_score
from utils.dist import init_ddp, worker_init_fn
from data.dataset import NMRShardedDataset, create_webdataset
from models.palette import PaletteViewSynthesis


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
    parser.add_argument("-s", action="store_true")
    parser.add_argument(
        "--wandb", action="store_true", help="Log run to Weights and Biases."
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    rank, world_size = init_ddp()
    device = torch.device(f"cuda:{rank}")

    args.wandb = args.wandb and rank == 0

    max_it = config["model"].get("max_it", 1000000)
    validate_every = config["model"].get("validate_every", 5000)
    checkpoint_every = config["model"].get("checkpoint_every", 5000)
    log_every = config["model"].get("log_every", 10)

    exp_name = os.path.basename(os.path.dirname(args.config))

    out_dir = os.path.dirname(args.config)

    batch_size = config["data"]["params"]["batch_size"] // world_size

    # Initialize datasets
    print("Loading training set...")
    # train_dataset = NMRShardedDataset(**config["data"]["params"]["train"]["params"])
    train_dataset = create_webdataset(**config["data"]["params"]["train"]["params"])
    print("Training set loaded.")

    print("Loading validation set...")
    # val_dataset = NMRShardedDataset(**config["data"]["params"]["test"]["params"])
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
        persistent_workers=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=max(1, batch_size // 8),
        num_workers=1,
        sampler=val_sampler,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
    )

    val_vis_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=12, worker_init_fn=worker_init_fn
    )
    train_vis_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=12, worker_init_fn=worker_init_fn
    )
    val_vis_data = next(iter(val_vis_loader))
    train_vis_data = next(iter(train_vis_loader))

    # Initialize model

    model = PaletteViewSynthesis(
        config["model"]["unet_params"],
        config["model"]["palette_params"]["beta_schedule"],
    ).to(device)

    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    peak_it = config.get("lr_warmup", 2500)
    decay_it = config.get("decay_it", 4000000)

    lr_scheduler = LrScheduler(
        peak_lr=1e-4, peak_it=peak_it, decay_it=decay_it, decay_rate=0.16
    )
    optimizer = optim.Adam(model.parameters(), lr=lr_scheduler.get_cur_lr(0))
    checkpoint = Checkpoint(out_dir, device=device, model=model, optimizer=optimizer)

    # Try to resume training

    try:
        if os.path.exists(os.path.join(out_dir, f"model_{max_it}.pt")):
            load_dict = checkpoint.load(f"model_{max_it}.pt")
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
        else:
            print(f"Resuming wandb with existing run_id {run_id}.")
        wandb.init(
            project="palette-view-synthesis",
            name=os.path.dirname(args.config),
            id=run_id,
            resume=True,
        )
        wandb.config = config

    # Training loop

    model.set_new_noise_schedule(device=device, phase="train")
    model.set_loss(F.mse_loss)
    metric_val_best = float("inf")

    masked_cnt = 3
    test_eval = False

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

                if (checkpoint_every > 0) and (it % checkpoint_every) == 0 and it > 0:
                    checkpoint.save("model.pt", **checkpoint_dict)

            # Run validation
            if test_eval or (
                it > 0 and validate_every > 0 and (it % validate_every) == 0
            ):
                model.eval()
                print("Running evaluation...")

                losses = list()
                for val_batch in val_loader:
                    images = batch["images"].to(device)
                    loss = model(images)
                    losses.append(loss.item())

                avg_loss = np.mean(losses)
                log_dict["val_loss"] = avg_loss
                if avg_loss < metric_val_best:
                    metric_val_best = avg_loss

                print("Running image generation...")
                images = val_vis_data["images"].to(device)

                y_t, generated_batch = model.generate(images, masked_cnt)
                log_dict["input_batch"] = wandb.Image(
                    make_grid(
                        rearrange(images, "b v c h w -> (v b) c h w"),
                        nrow=images.shape[0],
                    )
                )

                for i, generated_images in enumerate(generated_batch):
                    grid_images = rearrange(
                        generated_images, "s v c h w -> (v s) c h w"
                    )
                    log_dict[f"generated_images_{i}"] = wandb.Image(
                        make_grid(grid_images, nrow=generated_images.shape[0])
                    )

            new_lr = lr_scheduler.get_cur_lr(it)
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr

            t0 = time.perf_counter()

            images = batch["images"].to(device)
            model.train()
            optimizer.zero_grad()
            loss = model(images)
            loss.backward()
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
