import datetime
import os
import time
from pathlib import Path

import imageio
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

        self.args = args

        self.log_dict = dict()

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

        self.__init_model()
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

    def __init_model(self):
        denoise_net = self.config["model"].get("denoise_net", "unet")
        if denoise_net == "unet":
            denoise_fn = UNet(**self.config["model"]["denoise_net_params"])
        else:
            raise ValueError("Provided denoising function is not supported!")
        self.model = ViewFusion(
            denoise_fn,
            self.config["model"]["view_fusion_params"]["beta_schedule"],
            self.config["model"]["view_fusion_params"].get("weighting_train", True),
            self.config["model"]["view_fusion_params"].get("weighting_inference", True),
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

        # Try loading existing model - load different model for
        # training / resuming and inference
        if self.args.train or self.args.resume:
            print("Loading latest checkpoint...")
            checkpoint_name = "model.pt"
        elif self.args.inference or self.args.eval:
            print("Loading best checkpoint...")
            checkpoint_name = "best_model_all.pt"

        try:
            if os.path.exists(os.path.join(self.out_dir, checkpoint_name)):
                load_dict = self.checkpoint.load(checkpoint_name)
            else:
                load_dict = self.checkpoint.load(checkpoint_name)
        except FileNotFoundError:
            load_dict = dict()

        self.it = load_dict.get("it", -1)
        self.time_elapsed = load_dict.get("t", 0.0)
        self.run_id = load_dict.get("run_id", None)
        self.max_views = self.config["data"]["params"]["max_views"]
        self.relative = self.config["model"].get("relative", False)
        print("Relative conditioning:", self.relative)

        self.best_metrics = dict()
        self.best_metrics["ssim"] = load_dict.get("ssim", -np.inf)
        self.best_metrics["psnr"] = load_dict.get("psnr", -np.inf)

    def __init_dataloaders(self):
        if self.world_size > 0:
            batch_size = self.config["data"]["params"]["batch_size"] // self.world_size
        else:
            batch_size = self.config["data"]["params"]["batch_size"]

        # Initialize train webdataset and dataloader
        if self.args.train:
            print("Loading training set...")
            train_dataset = create_webdataset(
                **self.config["data"]["params"]["train"]["params"]
            )
            print("Training set loaded.")

            num_workers = self.config["data"]["params"].get("num_workers", 1)
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

        # Initialize val / test webdataset and dataloaders (test dataset is used both
        # for validating and test, as the benchmarks are perofmed solely on the test set
        # throughout)
        print("Loading validation set...")
        val_dataset = create_webdataset(
            **self.config["data"]["params"]["test"]["params"]
        )
        print("Validation set loaded.")

        testset_size = self.config["data"]["params"]["test"]["params"].get("size", 8751)
        epoch_size = testset_size // batch_size

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
            self.best_metrics["ssim"] = wandb.run.summary.get("ssim").get(
                "max", -np.inf
            )
            self.best_metrics["psnr"] = wandb.run.summary.get("psnr").get(
                "max", -np.inf
            )

        acc_loss = 0

        # Training loop
        while True:
            for batch in self.train_loader:
                self.it += 1

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

        # Save best metric models if currently training
        if self.args.train:
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
        variable = False
        plausible = False
        fill_missing = False
        generate = False

        if self.args.train:
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

            if self.relative:
                cond_padded = torch.nn.utils.rnn.pad_sequence(
                    [cond[i, :view_idx, 3:] for i, view_idx in enumerate(view_count)],
                    batch_first=True,
                )
            else:
                cond_padded = torch.nn.utils.rnn.pad_sequence(
                    [cond[i, :view_idx] for i, view_idx in enumerate(view_count)],
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

            if self.wandb_enabled:
                self.log_dict["output"] = wandb.Image(
                    make_grid(
                        rearrange(output, "b s c h w -> (b s) c h w"),
                        nrow=output.shape[1],
                        scale_each=True,
                    ),
                    caption="Denoising steps, Target, Input View",
                )

            else:
                # TODO save locally
                pass

        elif self.args.inference:
            if self.args.extrapolate:
                self.__extrapolate()

            if self.args.autoregressive:
                self.__autoregressive()

            if self.args.generate_gifs:
                self.__generate_gif()

        # save everything to wandb if enabled
        if self.wandb_enabled:
            wandb.log(self.log_dict)  # step=self.it)

    def __extrapolate(self):
        print("Running extrapolate image generation...")
        target = self.val_vis_data["target"].to(self.device)
        cond = self.val_vis_data["cond"].to(self.device)
        # view_count = val_vis_data["view_count"].to(device)
        view_count = torch.randint(self.max_views + 1, 24, (target.shape[0],)).to(
            self.device
        )
        angle = self.val_vis_data["angle"].to(self.device)

        with torch.no_grad():
            _, generated_batch, logit_arr, weight_arr, _ = self.model(
                y_cond=cond,
                view_count=view_count,
                angle=angle,
                generate=True,
            )

        cond_padded = torch.nn.utils.rnn.pad_sequence(
            [cond[i, :view_idx] for i, view_idx in enumerate(view_count)],
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

        self.log_dict["extrapolate"] = make_grid(
            rearrange(output, "b s c h w -> (b s) c h w"),
            nrow=output.shape[1],
            scale_each=True,
        )

        return make_grid(
            rearrange(output, "b s c h w -> (b s) c h w"),
            nrow=output.shape[1],
            scale_each=True,
        )

    def __autoregressive(self):
        # os.makedirs(os.path.join(out_dir, "ar"))
        all_views = self.val_vis_data["all_views"][10:11].to(self.device)
        cond = all_views[:, :1].to(self.device)
        angles_incremental = torch.as_tensor(
            [2 * np.pi / 24 * i for i in range(1, 25)]
        ).to(self.device)
        # torch.save(
        #     all_views,
        #     os.path.join(out_dir, "ar", f"all_views.pt"),
        # )
        # torch.save(
        #     cond,
        #     os.path.join(out_dir, "ar", f"cond_0.pt"),
        # )

        cond_list = list()
        sample_list = list()

        for count, angle in enumerate(angles_incremental, start=1):
            print(f"Conditioning count and sample number: {count}")
            cond_list.append(cond[0])
            view_count = torch.full((cond.shape[0],), count).to(self.device)
            angle = torch.full((cond.shape[0], 1), angle).to(self.device)
            *_, generated_samples = self.model(
                y_cond=cond, view_count=view_count, angle=angle, generate=True
            )
            cond = torch.cat((cond, generated_samples[:, None, ...]), dim=1)
            sample_list.append(generated_samples)

            # torch.save(
            #     cond,
            #     os.path.join(out_dir, "ar", f"cond_{count}.pt"),
            # )
            # torch.save(
            #     generated_samples,
            #     os.path.join(out_dir, "ar", f"samples_{count}.pt"),
            # )

        print(len(cond_list))
        conds_padded = torch.clamp(
            torch.nn.utils.rnn.pad_sequence(
                cond_list, batch_first=True, padding_value=1.0
            ),
            0,
            1,
        )
        samples = torch.clamp(torch.stack(sample_list), 0, 1)

        joint = [
            torch.cat((cond, sample)) for sample, cond in zip(samples, conds_padded)
        ]
        print(joint[0].shape)
        ar_frames = [
            (make_grid(sample, nrow=25).cpu() * 255).to(torch.uint8) for sample in joint
        ]

        if self.wandb_enabled:
            print(ar_frames[0].shape)
            self.log_dict[f"autoregressive_single"] = wandb.Image(ar_frames[0])
            self.log_dict[f"autoregressive_animated"] = wandb.Video(
                np.stack(ar_frames), format="gif"
            )

    def __generate_gif(self):
        print("Running animation sequence generation...")
        i = 10  # class
        n = 24
        views = self.val_vis_data["all_views"].to(self.device)
        angles_incremental = torch.as_tensor([2 * np.pi / n * i for i in range(n)]).to(
            self.device
        )
        target = torch.repeat_interleave(views[i], n // 24, dim=0)
        cond_views = torch.stack([views[i, ::4]] * target.shape[0], dim=0)
        view_counts = torch.as_tensor(
            [cond_views.shape[1] for _ in range(target.shape[0])],
        ).to(self.device)

        _, generated_batch, logit_arr, weight_arr, _ = self.model(
            y_cond=cond_views,
            angle=angles_incremental.unsqueeze(1),
            view_count=view_counts,
            generate=True,
        )

        # mask = (
        #     torch.stack(
        #         [torch.stack([target] * weight_arr.shape[2], dim=1)]
        #         * weight_arr.shape[1],
        #         dim=1,
        #     )
        #     != 1.0
        # )
        # weight_masked = weight_arr * mask

        frames = []
        for i in range(n):
            target_grid = torch.cat(
                [
                    target[i, ...][None, ...],
                ]
                * (cond_views.shape[1] + 1)
            )[None, ...]
            view_weights = torch.cat(
                (weight_arr[i, ...], cond_views[i][None, ...]), dim=0
            )
            view_weights = torch.cat(
                (
                    view_weights,
                    torch.clamp(generated_batch[i, ...][:, None, ...], 0, 1),
                ),
                dim=1,
            )
            view_weights = torch.cat((view_weights, target_grid))
            view_weights = make_grid(
                rearrange(view_weights, "s v c h w -> (v s) c h w"),
                nrow=view_weights.shape[0],
                # scale_each=True,
                pad_value=0.9,
            )
            # view_weights = rearrange(view_weights, "c h w -> h w c")
            frames.append((view_weights.cpu() * 255).to(torch.uint8))

        if self.wandb_enabled:
            self.log_dict[f"weights_animated"] = wandb.Video(
                np.stack(frames), format="gif"
            )

        else:
            imageio.mimsave(
                os.path.join(self.out_dir, "output_fast.gif"),
                frames,
                "GIF",
                duration=0.1,
            )

        # frames = []
        # for i in range(n):
        #     target_grid = torch.cat(
        #         [
        #             target[i, ...][None, ...],
        #         ]
        #         * (cond_views.shape[1] + 1)
        #     )[None, ...]
        #     view_weights = torch.cat(
        #         (weight_masked[i, ...], cond_views[i][None, ...]), dim=0
        #     )
        #     view_weights = torch.cat(
        #         (
        #             view_weights,
        #             torch.clamp(generated_batch[i, ...][:, None, ...], 0, 1),
        #         ),
        #         dim=1,
        #     )
        #     view_weights = torch.cat((view_weights, target_grid))
        #     view_weights = make_grid(
        #         rearrange(view_weights, "s v c h w -> (v s) c h w"),
        #         nrow=view_weights.shape[0],
        #         scale_each=True,
        #     )
        #     # view_weights = rearrange(view_weights, "c h w -> h w c")
        #     frames.append(view_weights.cpu())

        # if self.wandb_enabled:
        #     self.log_dict[f"output_masked_fast"] = wandb.Video(
        #         np.stack(frames), format="gif"
        #     )

        # else:
        #     imageio.mimsave(
        #         os.path.join(self.out_dir, "output_masked_fast.gif"),
        #         frames,
        #         "GIF",
        #         duration=0.1,
        #     )
