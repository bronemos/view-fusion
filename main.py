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
    parser.add_argument("-t", "--train" action="store_true", default=True)
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

    epoch_it = load_dict.get("epoch_it", -1)
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

    masked_cnt = 3
    view_cnt = 24
    in_chann = 3
    test_eval = False

    while True:
        for batch in train_loader:
            it += 1
            if rank == 0:
                pass

            # Run validation
            if test_eval or (
                it > 0 and validate_every > 0 and (it % validate_every) == 0
            ):
                print("Running evaluation...")
                eval_dict = dict()
                images = val_vis_data["images"].to(device)
                print(torch.max(images), torch.min(images))
                masked_ids = torch.randint(24, size=(masked_cnt,))
                mask = rearrange(
                    torch.zeros_like(images),
                    "b (v c) h w -> b v c h w",
                    v=view_cnt,
                    c=in_chann,
                )
                mask[:, masked_ids, :, :, :] = 1
                mask = rearrange(mask, "b v c h w -> b (v c) h w")
                y_t, ret_arr = model.generate(images, mask)

                generated_images = rearrange(
                    ret_arr, "b (v c) h w -> (v b) c h w", v=view_cnt, c=in_chann
                )

                for i, generated_image in enumerate(generated_images):
                    eval_dict[f"generated_images_{i}"] = wandb.Image(
                        make_grid(
                            generated_images,
                            nrow=generated_images.shape[0] // view_cnt,
                        )
                    )

                if args.wandb:
                    wandb.log(eval_dict, step=it)

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
                log_dict = dict()
                log_dict["t"] = time_elapsed
                log_dict["lr"] = new_lr
                log_dict["loss"] = loss.item()

                if args.wandb:
                    wandb.log(log_dict, step=it)

            if it > max_it:
                if rank == 0:
                    pass
                exit(0)
