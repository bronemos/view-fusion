import datetime
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import webdataset as wds
import yaml
from einops import rearrange
from torch.nn.parallel import DistributedDataParallel
from torchvision.utils import make_grid

import wandb
from data.nmr_dataset import create_webdataset
from model.unet import UNet
from model.view_fusion import ViewFusion
from utils.checkpoint import Checkpoint
from utils.dist import init_ddp, reduce_dict, worker_init_fn
from utils.metrics import compute_psnr, compute_ssim
from utils.schedulers import LrScheduler


class Experiment:
    def __init__(self, args):
        # Setup logging directories
        if args.inference or args.resume:
            if args.src is None:
                raise ValueError("Source directory (-s, --src_dir) must be provided.")
            self.out_dir = Path(args.src)
            exp_name = os.path.basename(args.src)
            now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            with open(os.path.join(args.src, "config.yaml")) as f:
                self.config = yaml.load(f, Loader=yaml.CLoader)

        else:
            log_dir = "./logs"
            config_name = os.path.splitext(os.path.basename(args.config))[0]
            now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            exp_name = "-".join((now, config_name))
            self.out_dir = os.path.join(log_dir, exp_name)
            with open(args.config, "r") as f:
                self.config = yaml.load(f, Loader=yaml.CLoader)

        # Initialize ddp (parallelization)
        if args.gpu:
            self.rank, self.world_size = init_ddp()
            self.device = torch.device(f"cuda:{self.rank}")
        else:
            self.rank, self.world_size = init_ddp()
            self.device = torch.device("cpu")

        args.wandb = args.wandb and self.rank == 0

        self.__init_model_train()
        self.__init_dataloaders()

        self.wandb_enabled = False
        if args.wandb:
            if self.run_id is None:
                self.run_id = wandb.util.generate_id()
                print(f"Sampled new wandb run_id {self.run_id}.")
                wandb.init(
                    project="view-fusion",
                    name=f"{exp_name}",
                    id=self.run_id,
                    resume=True,
                    config=self.config,
                )
            else:
                print(f"Resuming wandb with existing run_id {self.run_id}.")
                wandb.init(
                    project="view-fusion",
                    id=self.run_id,
                    resume=True,
                    config=self.config,
                )
            wandb.define_metric("ssim", summary="max")
            wandb.define_metric("psnr", summary="max")

            self.wandb_enabled = True

        self.relative = True

    def __init_model_train(self):
        denoise_net = self.config["model"].get("denoise_net", "unet")
        if denoise_net == "unet":
            denoise_fn = UNet(**self.config["model"]["denoise_net_params"])
        else:
            raise ValueError("Provided denoising function is not supported!")
        self.model = ViewFusion(
            denoise_fn,
            self.config["model"]["view_fusion_params"]["beta_schedule"],
        ).to(self.device)
        self.model.set_new_noise_schedule(device=self.device, phase="train")

        if self.world_size > 1:
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.rank], output_device=self.rank
            )
            model_module = self.model.module
        else:
            model_module = self.model

        peak_it = self.config.get("lr_warmup", 2500)
        decay_it = self.config.get("decay_it", 4000000)

        self.lr_scheduler = LrScheduler(
            peak_lr=1e-4, peak_it=peak_it, decay_it=decay_it, decay_rate=0.16
        )
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr_scheduler.get_cur_lr(0)
        )
        self.checkpoint = Checkpoint(
            self.out_dir,
            device=self.device,
            rank=self.rank,
            config=self.config,
            model=model_module,
            optimizer=self.optimizer,
        )

        # Try loading existing model
        try:
            if os.path.exists(os.path.join(self.out_dir, f"model.pt")):
                load_dict = self.checkpoint.load(f"model.pt")
            else:
                load_dict = self.checkpoint.load("model.pt")
        except FileNotFoundError:
            load_dict = dict()

        self.it = load_dict.get("it", -1)
        self.time_elapsed = load_dict.get("t", 0.0)
        self.run_id = load_dict.get("run_id", None)
        self.max_views = self.config["data"]["params"]["max_views"]

        self.best_metrics = dict()
        self.best_metrics["ssim"] = load_dict.get("ssim", -np.inf)
        self.best_metrics["psnr"] = load_dict.get("psnr", -np.inf)

    def __init_dataloaders(self):
        if self.world_size > 0:
            batch_size = self.config["data"]["params"]["batch_size"] // self.world_size
        else:
            batch_size = self.config["data"]["params"]["batch_size"]

        # Initialize webdatasets
        print("Loading training set...")
        train_dataset = create_webdataset(
            **self.config["data"]["params"]["train"]["params"]
        )
        print("Training set loaded.")

        print("Loading validation set...")
        val_dataset = create_webdataset(
            **self.config["data"]["params"]["test"]["params"]
        )
        print("Validation set loaded.")

        # Initialize dataloaders
        num_workers = self.config["data"]["params"].get("num_workers", 1)
        testset_size = self.config["data"]["params"]["test"]["params"].get("size", 8751)
        epoch_size = testset_size // batch_size
        print(
            f"Initializing datalaoders, using {num_workers} workers per process for data loading."
        )

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            assert num_workers == 1

        self.train_loader = wds.WebLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
        )

        self.val_loader = wds.WebLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=1,
            pin_memory=False,
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
        ).with_epoch(epoch_size)

        val_vis_loader = wds.WebLoader(
            val_dataset,
            batch_size=12,
            worker_init_fn=worker_init_fn,
        )

        self.val_vis_data = next(iter(val_vis_loader))

    def train(self):
        max_it = self.config["model"].get("max_it", 1000000)
        validate_every = self.config["model"].get("validate_every", 5000)
        validate_from = self.config["model"].get("validate_from", 100000)
        checkpoint_every = self.config["model"].get("checkpoint_every", 100)
        log_every = self.config["model"].get("log_every", 100)

        # Overwrite load best metrics from wandb if enabled
        if self.wandb_enabled:
            self.best_metrics["ssim"] = wandb.run.summary.get("ssim", -np.inf)
            self.best_metrics["psnr"] = wandb.run.summary.get("psnr", -np.inf)

        acc_loss = 0

        # Training loop
        while True:
            for batch in self.train_loader:
                self.it += 1
                self.log_dict = dict()

                if self.rank == 0:
                    self.checkpoint_dict = {
                        "it": self.it,
                        "t": self.time_elapsed,
                        "run_id": self.run_id,
                    }
                    self.checkpoint_dict.update(self.best_metrics)

                    if (
                        (checkpoint_every > 0)
                        and (self.it % checkpoint_every) == 0
                        and self.it > 0
                    ):
                        self.checkpoint.save("model.pt", **self.checkpoint_dict)

                # Run validation
                if (
                    self.it >= validate_from
                    and validate_every > 0
                    and ((self.it - validate_from) % validate_every) == 0
                ):
                    self.eval()
                    self.inference()

                new_lr = self.lr_scheduler.get_cur_lr(self.it)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = new_lr

                t0 = time.perf_counter()

                target = batch["target"].to(self.device)
                cond = (
                    batch["cond"].to(self.device)
                    if not self.relative
                    else batch["relative_cond"].to(self.device)
                )
                view_count = torch.randint(
                    1, self.max_views + 1, (target.shape[0],)
                ).to(self.device)
                angle = (
                    batch["angle"].to(self.device)
                    if not self.relative
                    else batch["relative_angle"].to(self.device)
                )

                self.model.train()
                self.optimizer.zero_grad()
                loss = self.model(
                    y_0=target, y_cond=cond, view_count=view_count, angle=angle
                )
                acc_loss += loss.item()
                loss.backward()
                self.optimizer.step()

                self.time_elapsed += time.perf_counter() - t0

                if log_every > 0 and self.it % log_every == 0:
                    self.log_dict["t"] = self.time_elapsed
                    self.log_dict["lr"] = new_lr
                    self.log_dict["loss"] = acc_loss / log_every
                    acc_loss = 0

                if self.wandb_enabled and self.log_dict:
                    wandb.log(self.log_dict, step=self.it)

                if self.it > max_it:
                    print("Maximum iteration count reached.")
                    if self.rank == 0:
                        self.checkpoint.save(
                            "model.pt",
                        )
                    exit(0)

    def eval(self):
        print("Running metric evaluation...")
        self.model.eval()

        generated_batches = list()
        ground_truth_batches = list()
        eval_dict = dict()

        for val_batch in self.val_loader:
            target = val_batch["target"].to(self.device)
            cond = (
                val_batch["cond"].to(self.device)
                if not self.relative
                else val_batch["relative_cond"].to(self.device)
            )
            view_count = torch.randint(1, self.max_views + 1, (target.shape[0],))
            angle = (
                val_batch["angle"].to(self.device)
                if not self.relative
                else val_batch["relative_angle"]
            )

            with torch.no_grad():
                *_, generated_samples = self.model(
                    y_cond=cond,
                    view_count=view_count,
                    angle=angle,
                    generate=True,
                )
                generated_batches.append(generated_samples)
                ground_truth_batches.append(target)

        print("Completed generation.")
        torch.distributed.barrier()

        ssims = list()
        psnrs = list()
        print("Computing metrics.")
        for gt_batch, generated_batch in zip(ground_truth_batches, generated_batches):
            ssims.append(compute_ssim(generated_batch, gt_batch))
            psnrs.append(compute_psnr(generated_batch, gt_batch))

        ssims = torch.cat(ssims)
        psnrs = torch.cat(psnrs)

        eval_dict["ssim"] = torch.mean(ssims)
        eval_dict["psnr"] = torch.mean(psnrs)

        print("Computed metrics.")

        torch.distributed.barrier()
        reduced_dict = reduce_dict(eval_dict)
        torch.distributed.barrier()
        print("Reduced eval dict.")

        self.log_dict["ssim"] = reduced_dict["ssim"]
        self.log_dict["psnr"] = reduced_dict["psnr"]

        # Save best metric models
        best_metric_cnt = 0
        if self.log_dict["ssim"] > self.best_metrics["ssim"]:
            best_metric_cnt += 1
            self.best_metrics["ssim"] = self.log_dict["ssim"]
            if self.rank == 0:
                self.checkpoint.save(f"best_model_ssim.pt", **self.checkpoint_dict)
                print(f"Saved best SSIM modle at iteration {self.it}.")

        if self.log_dict["psnr"] > self.best_metrics["psnr"]:
            best_metric_cnt += 1
            self.best_metrics["psnr"] = self.log_dict["psnr"]
            if self.rank == 0:
                self.checkpoint.save(f"best_model_psnr.pt", **self.checkpoint_dict)
                print(f"Saved best PSNR model at iteration {self.it}.")

        if best_metric_cnt == 2 and self.rank == 0:
            self.checkpoint.save(f"best_model_all.pt", **self.checkpoint_dict)
            print(f"Saved best model at iteration {self.it}.")

    def inference(self):
        if self.wandb_enabled:
            print("Running image generation...")

            target = self.val_vis_data["target"].to(self.device)
            cond = (
                self.val_vis_data["cond"].to(self.device)
                if not self.relative
                else self.val_vis_data["relative_cond"].to(self.device)
            )
            view_count = torch.randint(1, self.max_views + 1, (target.shape[0],)).to(
                self.device
            )
            angle = (
                self.val_vis_data["angle"].to(self.device)
                if not self.relative
                else self.val_vis_data["relative_angle"].to(self.device)
            )

            _, generated_batch, *_ = self.model(
                y_cond=cond,
                view_count=view_count,
                angle=angle,
                generate=True,
            )

            cond_padded = torch.nn.utils.rnn.pad_sequence(
                [cond[i, :view_idx, 3:] for i, view_idx in enumerate(view_count)],
                batch_first=True,
            )

            output = torch.cat(
                (
                    torch.clamp(generated_batch, 0, 1),
                    torch.unsqueeze(target, 1),
                    cond_padded,
                ),
                dim=1,
            )

            self.log_dict["output"] = wandb.Image(
                make_grid(
                    rearrange(output, "b s c h w -> (b s) c h w"),
                    nrow=output.shape[1],
                    scale_each=True,
                ),
                caption="Denoising steps, Target, Input View",
            )
            wandb.log(self.log_dict, step=self.it)
