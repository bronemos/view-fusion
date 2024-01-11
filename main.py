import argparse
import os
import yaml
import time
import datetime
import random

from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import webdataset as wds
from torch.nn.parallel import DistributedDataParallel
from torchvision.utils import make_grid, save_image
from einops import rearrange
from cleanfid import fid

from utils.checkpoint import Checkpoint
from utils.metrics import compute_ssim, compute_psnr
from utils.dist import init_ddp, worker_init_fn, reduce_dict
from utils.schedulers import LrScheduler
from data.dataset import (
    create_webdataset,
    create_webdataset_metzler,
)
from model.palette import PaletteViewSynthesis
from model.unet import UNet


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="YAML config file")
    parser.add_argument("-gpus", "--gpu_ids", type=str, default=None)
    parser.add_argument("-t", "--train", action="store_true", default=True)
    parser.add_argument("-i", "--inference", action="store_true", default=False)
    parser.add_argument("-e", "--test_eval", action="store_true", default=False)
    parser.add_argument("-r", "--resume", action="store_true", default=False)
    parser.add_argument("-s", "--src_dir", type=str, default=None)
    parser.add_argument("-m", "--metzler", action="store_true", default=False)
    parser.add_argument(
        "--wandb", action="store_true", help="Log run to Weights and Biases."
    )

    return parser


def main(args):
    log_dir = "./logs"
    is_metzler = args.metzler

    if args.inference or args.resume:
        if args.src_dir is None:
            raise ValueError("Source directory (-s, --src_dir) must be provided.")
        out_dir = Path(args.src_dir)
        exp_name = os.path.basename(args.src_dir)
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        with open(os.path.join(args.src_dir, "config.yaml")) as f:
            config = yaml.load(f, Loader=yaml.CLoader)

    else:
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        exp_name = "-".join((now, config_name))
        out_dir = os.path.join(log_dir, exp_name)
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
    validate_from = config["model"].get("validate_from", 50000)
    checkpoint_every = config["model"].get("checkpoint_every", 10)
    log_every = config["model"].get("log_every", 10)

    tmp_dir = os.path.join("/tmp", exp_name)
    print(tmp_dir)

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
    testset_size = config["data"]["params"]["test"]["params"].get("size", 8751)
    epoch_size = testset_size // (batch_size * world_size)
    print(
        f"Initializing datalaoders, using {num_workers} workers per process for data loading."
    )
    train_sampler = val_sampler = None

    if isinstance(train_dataset, torch.utils.data.IterableDataset):
        assert num_workers == 1
    elif world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, drop_last=False
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, drop_last=False
        )

    train_loader = wds.WebLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
    )

    val_loader = wds.WebLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=1,
        sampler=val_sampler,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
    ).with_epoch(epoch_size)

    val_vis_loader = wds.WebLoader(
        val_dataset,
        batch_size=12,
        worker_init_fn=worker_init_fn,
    )
    train_vis_loader = wds.WebLoader(
        train_dataset,
        batch_size=12,
        worker_init_fn=worker_init_fn,
    )
    val_vis_data = next(iter(val_vis_loader))
    train_vis_data = next(iter(train_vis_loader))

    # Initialize model

    denoise_net = config["model"].get("denoise_net", "unet")
    if denoise_net == "unet":
        denoise_fn = UNet(**config["model"]["denoise_net_params"])
    else:
        raise ValueError("Provided denoising function is not supported!")
    model = PaletteViewSynthesis(
        denoise_fn,
        config["model"]["palette_params"]["beta_schedule"],
    ).to(device)
    model.set_new_noise_schedule(device=device, phase="train")

    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)
        model_module = model.module
    else:
        model_module = model

    # if device == torch.device("cpu"):
    #    model = DistributedDataParallel(model)

    peak_it = config.get("lr_warmup", 2500)
    decay_it = config.get("decay_it", 4000000)

    lr_scheduler = LrScheduler(
        peak_lr=1e-4, peak_it=peak_it, decay_it=decay_it, decay_rate=0.16
    )
    optimizer = optim.Adam(model.parameters(), lr=lr_scheduler.get_cur_lr(0))
    checkpoint = Checkpoint(
        out_dir,
        device=device,
        rank=rank,
        config=config,
        model=model_module,
        optimizer=optimizer,
    )

    # try loading existing model

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

    if args.wandb:
        import wandb

        if run_id is None:
            run_id = wandb.util.generate_id()
            print(f"Sampled new wandb run_id {run_id}.")
            wandb.init(
                project="palette-view-synthesis",
                name=f"{exp_name}",
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
        if rank == 0:
            os.makedirs(images_path)
            os.makedirs(generated_path)
            os.makedirs(ground_truth_path)
        compute_fid = False
        generate = True
        if compute_fid:
            generated_batches = list()
            ground_truth_batches = list()

            for val_batch in val_loader:
                target = val_batch["target"].to(device)
                cond = val_batch["cond"].to(device)
                # angle = batch["angle"].to(device)
                with torch.no_grad():
                    *_, generated_samples = model(y_cond=cond, generate=True)
                    generated_batches.append(generated_samples)
                    ground_truth_batches.append(target)

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
            target = val_vis_data["target"].to(device)
            cond = val_vis_data["cond"].to(device)
            spoof_cond = val_vis_data["spoof_cond"].to(device)
            torch.save(target, os.path.join(out_dir, "target.pt"))
            torch.save(cond, os.path.join(out_dir, "cond.pt"))
            torch.save(spoof_cond, os.path.join(out_dir, "spoof_cond.pt"))
            # angle = val_vis_data["angle"].to(device)

            y_t, generated_batch, logit_arr, weight_arr, _ = model(
                y_cond=cond, generate=True
            )
            # print(out_dir)
            torch.save(generated_batch, os.path.join(out_dir, "generated_batch.pt"))
            torch.save(logit_arr, os.path.join(out_dir, "logit_arr.pt"))
            torch.save(weight_arr, os.path.join(out_dir, "weight_arr.pt"))

            output = torch.cat(
                (
                    torch.clamp(generated_batch, 0, 1),
                    torch.unsqueeze(target, 1),
                    cond[:, :, :3, ...],
                ),
                dim=1,
            )

            print("Running target spoof...")
            _, spoof_generated_batch, spoof_logit_arr, spoof_weight_arr, _ = model(
                y_cond=spoof_cond, generate=True
            )
            torch.save(
                spoof_generated_batch, os.path.join(out_dir, "spoof_generated_batch.pt")
            )
            torch.save(spoof_logit_arr, os.path.join(out_dir, "spoof_logit_arr.pt"))
            torch.save(spoof_weight_arr, os.path.join(out_dir, "spoof_weight_arr.pt"))
            spoof_output = torch.cat(
                (
                    torch.clamp(spoof_generated_batch, 0, 1),
                    torch.unsqueeze(target, 1),
                    spoof_cond[:, :, :3, ...],
                ),
                dim=1,
            )

            print("Running variable view count...")
            variable_output = list()
            for i in range(1, cond.shape[0]):
                print(f"Running view count {i}")
                (
                    _,
                    variable_generated_batch,
                    variable_logit_arr,
                    variable_weight_arr,
                    _,
                ) = model(y_cond=cond[:, :i, ...], generate=True)
                torch.save(
                    variable_generated_batch,
                    os.path.join(out_dir, f"variable_generated_batch_{i}.pt"),
                )
                torch.save(
                    variable_logit_arr,
                    os.path.join(out_dir, f"variable_logit_arr_{i}.pt"),
                )
                torch.save(
                    variable_weight_arr,
                    os.path.join(out_dir, f"variable_weight_arr_{i}.pt"),
                )
                variable_output.append(variable_generated_batch)

            # TODO implement arbitrary (inbetween) angle test

            if args.wandb:
                log_dict["inference_output"] = wandb.Image(
                    make_grid(
                        rearrange(output, "b s c h w -> (b s) c h w"),
                        nrow=output.shape[1],
                        scale_each=True,
                    ),
                    caption="Denoising steps, Target, Input View",
                )
                log_dict["spoof_output"] = wandb.Image(
                    make_grid(
                        rearrange(spoof_output, "b s c h w -> (b s) c h w"),
                        nrow=spoof_output.shape[1],
                        scale_each=True,
                    ),
                    caption="Denoising steps, Target, Input View",
                )

                for i, variable_generated_batch in enumerate(variable_output, start=1):
                    log_dict[f"variable_output_{i}"] = wandb.Image(
                        make_grid(
                            rearrange(
                                variable_generated_batch, "b s c h w -> (b s) c h w"
                            ),
                            nrow=output.shape[1],
                            scale_each=True,
                        ),
                        caption="Denoising steps, Target, Input View",
                    )
                wandb.log(log_dict)

        exit(0)

    # Training loop
    if args.train:
        fid_score_best = np.inf
        ssim_best = -np.inf
        psnr_best = -np.inf

        test_eval = args.test_eval
        compute_metrics = False

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
                    it >= validate_from
                    and validate_every > 0
                    and (it % validate_every) == 0
                ):
                    images_path = os.path.join(tmp_dir, f"images-{it}")
                    ground_truth_path = os.path.join(images_path, "ground-truth")
                    generated_path = os.path.join(images_path, "generated")
                    if rank == 0:
                        os.makedirs(images_path)
                        os.makedirs(generated_path)
                        os.makedirs(ground_truth_path)
                    model.eval()
                    print("Running evaluation...")

                    if compute_metrics:
                        generated_batches = list()
                        ground_truth_batches = list()
                        eval_dict = dict()

                        i = 0
                        for val_batch in val_loader:
                            target = val_batch["target"].to(device)
                            cond = val_batch["cond"].to(device)
                            view_count = val_batch["view_count"].to(device)
                            angle = val_batch["angle"].to(device)

                            with torch.no_grad():
                                *_, generated_samples = model(
                                    y_cond=cond,
                                    view_count=view_count,
                                    angle=angle,
                                    generate=True,
                                )
                                generated_batches.append(generated_samples)
                                ground_truth_batches.append(target)
                            print("sample", i)
                            i += 1
                            # if test_eval:
                            #    break
                        print("completed generation")
                        cnt = 0
                        ssims = list()
                        psnrs = list()
                        for gt_batch, generated_batch in zip(
                            ground_truth_batches, generated_batches
                        ):
                            for gt_sample, generated_sample in zip(
                                gt_batch, generated_batch
                            ):
                                # print(gt_sample.shape, generated_sample.shape)
                                save_image(
                                    gt_sample,
                                    os.path.join(
                                        ground_truth_path, f"{rank}-{cnt}.png"
                                    ),
                                )
                                save_image(
                                    generated_sample,
                                    os.path.join(generated_path, f"{rank}-{cnt}.png"),
                                )
                                cnt += 1

                                ssims.append(compute_ssim(generated_batch, gt_batch))
                                psnrs.append(compute_psnr(generated_batch, gt_batch))

                        print("saved images")
                        ssims = torch.cat(ssims)
                        psnrs = torch.cat(psnrs)

                        eval_dict["fid_score"] = torch.tensor(
                            fid.compute_fid(
                                ground_truth_path,
                                generated_path,
                                verbose=False,
                                use_dataparallel=False,
                            ),
                            dtype=torch.float64,
                            device=device,
                        )
                        eval_dict["ssim"] = torch.mean(ssims)
                        eval_dict["psnr"] = torch.mean(psnrs)

                        print("computed metrics")

                        reduced_dict = reduce_dict(eval_dict)
                        print("reduced eval dict")

                        fid_score = reduced_dict["fid_score"]
                        log_dict["fid_score"] = fid_score

                        ssim = reduced_dict["ssim"]
                        log_dict["ssim"] = ssim

                        psnr = reduced_dict["psnr"]
                        log_dict["psnr"] = psnr

                        best_metric_cnt = 0
                        if ssim > ssim_best:
                            best_metric_cnt += 1
                            ssim_best = ssim
                            if rank == 0:
                                checkpoint.save(
                                    f"best_model_ssim.pt", **checkpoint_dict
                                )
                                print(f"Saved best SSIM modle at iteration {it}.")

                        if psnr > psnr_best:
                            best_metric_cnt += 1
                            psnr_best = psnr
                            if rank == 0:
                                checkpoint.save(
                                    f"best_model_psnr.pt", **checkpoint_dict
                                )
                                print(f"Saved best PSNR model at iteration {it}.")

                        if fid_score < fid_score_best:
                            best_metric_cnt += 1
                            fid_score_best = fid_score
                            if rank == 0:
                                checkpoint.save(f"best_model_fid.pt", **checkpoint_dict)
                                print(f"Saved best FID model at iteration {it}.")

                        if best_metric_cnt == 3 and rank == 0:
                            checkpoint.save(f"best_model_all.pt", **checkpoint_dict)
                            print(f"Saved best model at iteration {it}.")

                    if args.wandb:
                        print("Running image generation...")

                        target = val_vis_data["target"].to(device)
                        cond = val_vis_data["cond"].to(device)
                        view_count = val_vis_data["view_count"].to(device)
                        angle = val_vis_data["angle"].to(device)

                        _, generated_batch, *_ = model(
                            y_cond=cond,
                            view_count=view_count,
                            angle=angle,
                            generate=True,
                        )

                        # print(generated_batch.shape, target.shape, cond.shape, sep="\n")
                        output = torch.cat(
                            (
                                torch.clamp(generated_batch, 0, 1),
                                torch.unsqueeze(target, 1),
                                cond[:, :, :3, ...],
                            ),
                            dim=1,
                        )
                        output = torch.nn.utils.rnn.pad_sequence(
                            output, batch_first=True
                        )
                        log_dict["output"] = wandb.Image(
                            make_grid(
                                rearrange(output, "b s c h w -> (b s) c h w"),
                                nrow=output.shape[1],
                                scale_each=True,
                            ),
                            caption="Denoising steps, Target, Input View",
                        )
                        wandb.log(log_dict, step=it)
                        exit(0)

                new_lr = lr_scheduler.get_cur_lr(it)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = new_lr

                t0 = time.perf_counter()

                target = batch["target"].to(device)
                cond = batch["cond"].to(device)
                view_count = batch["view_count"].to(device)
                angle = batch["angle"].to(device)

                model.train()
                optimizer.zero_grad()
                loss = model(
                    y_0=target, y_cond=cond, view_count=view_count, angle=angle
                )
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


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()

    main(args)
