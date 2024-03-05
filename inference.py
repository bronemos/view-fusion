import argparse
import datetime
import os
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.optim as optim
import webdataset as wds
import yaml
from einops import rearrange
from torch.nn.parallel import DistributedDataParallel
from torchvision.utils import make_grid, save_image

from data.nmr_dataset import (
    create_webdataset,
    create_webdataset_metzler,
    create_webdataset_plot,
)
from model.unet import UNet
from model.view_fusion import ViewFusion
from utils.checkpoint import Checkpoint
from utils.dist import init_ddp, reduce_dict, worker_init_fn
from utils.metrics import compute_psnr, compute_ssim
from utils.schedulers import LrScheduler


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
    variable = False
    plausible = False
    fill_missing = False
    autoregressive = False
    generate = False
    animate_generation = True
    extrapolate = False

    return parser


def inference(args):

    if args.src_dir is None:
        raise ValueError("Source directory (-s, --src_dir) must be provided.")
    out_dir = Path(args.src_dir)
    exp_name = os.path.basename(args.src_dir)
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    with open(os.path.join(args.src_dir, "config.yaml")) as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    if args.gpu_ids is not None:
        rank, world_size = init_ddp()
        device = torch.device(f"cuda:{rank}")
    else:
        rank, world_size = init_ddp()
        device = torch.device("cpu")

    args.wandb = args.wandb and rank == 0

    max_views = config["data"]["params"]["max_views"]

    tmp_dir = os.path.join("/tmp", exp_name)

    if world_size > 0:
        batch_size = config["data"]["params"]["batch_size"] // world_size
    else:
        batch_size = config["data"]["params"]["batch_size"]

    # Initialize datasets

    print("Loading validation set...")
    val_dataset = create_webdataset(**config["data"]["params"]["test"]["params"])
    print("Validation set loaded.")

    # Initialize data loaders

    num_workers = config["data"]["params"].get("num_workers", 1)

    print(
        f"Initializing datalaoders, using {num_workers} workers per process for data loading."
    )
    val_sampler = None

    if isinstance(val_dataset, torch.utils.data.IterableDataset):
        assert num_workers == 1
    elif world_size > 1:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, drop_last=False
        )

    val_loader = wds.WebLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=1,
        sampler=val_sampler,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
    )

    val_vis_data = dict()
    for shard_idx in range(13):
        dataset = create_webdataset_plot(
            path="/scratch/work/spieglb1/datasets/NMR_sharded",
            mode="test",
            start_shard=shard_idx,
            end_shard=shard_idx,
        )
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

        visualization_data = next(iter(data_loader))
        for k, v in visualization_data.items():
            if isinstance(v, list):
                continue
            if k in val_vis_data:
                val_vis_data[k] = torch.cat((val_vis_data[k], v))
            else:
                val_vis_data[k] = v

    # Initialize model

    denoise_net = config["model"].get("denoise_net", "unet")
    if denoise_net == "unet":
        denoise_fn = UNet(**config["model"]["denoise_net_params"])
    else:
        raise ValueError("Provided denoising function is not supported!")
    model = ViewFusion(
        denoise_fn,
        config["model"]["view_fusion_params"]["beta_schedule"],
    ).to(device)
    model.set_new_noise_schedule(device=device, phase="train")

    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)
        model_module = model.module
    else:
        model_module = model

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
            load_dict = checkpoint.load(f"best_model_psnr.pt")
        else:
            load_dict = checkpoint.load("model.pt")
    except FileNotFoundError:
        load_dict = dict()

    it = load_dict.get("it", -1)
    run_id = load_dict.get("run_id", None)

    if args.wandb:
        import wandb

        if run_id is None:
            run_id = wandb.util.generate_id()
            print(f"Sampled new wandb run_id {run_id}.")
            wandb.init(
                project="view-fusion",
                name=f"{exp_name}",
                id=run_id,
                resume=True,
                config=config,
            )
        else:
            print(f"Resuming wandb with existing run_id {run_id}.")
            wandb.init(
                project="view-fusion",
                id=run_id,
                resume=True,
                config=config,
            )

    log_dict = dict()
    model.eval()

    variable = False
    plausible = False
    fill_missing = False
    autoregressive = False
    generate = False
    animate_generation = True
    extrapolate = False

    if generate:
        print("Running image generation...")
        target = val_vis_data["target"].to(device)
        cond = val_vis_data["cond"].to(device)
        view_count = torch.randint(1, max_views + 1, (target.shape[0],)).to(device)
        angle = val_vis_data["angle"].to(device)
        torch.save(target, os.path.join(out_dir, "target.pt"))
        torch.save(cond, os.path.join(out_dir, "cond.pt"))
        torch.save(view_count, os.path.join(out_dir, "view_count.pt"))
        # angle = val_vis_data["angle"].to(device)

        print(cond.shape, view_count.shape, angle.shape)
        with torch.no_grad():
            _, generated_batch, logit_arr, weight_arr, _ = model(
                y_cond=cond,
                view_count=view_count,
                angle=angle,
                generate=True,
            )
        # print(out_dir)
        torch.save(generated_batch, os.path.join(out_dir, "generated_batch.pt"))
        torch.save(logit_arr, os.path.join(out_dir, "logit_arr.pt"))
        torch.save(weight_arr, os.path.join(out_dir, "weight_arr.pt"))

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

        # print("Running target spoof...")
        # _, spoof_generated_batch, spoof_logit_arr, spoof_weight_arr, _ = model(
        #     y_cond=spoof_cond, generate=True
        # )
        # torch.save(
        #     spoof_generated_batch, os.path.join(out_dir, "spoof_generated_batch.pt")
        # )
        # torch.save(spoof_logit_arr, os.path.join(out_dir, "spoof_logit_arr.pt"))
        # torch.save(spoof_weight_arr, os.path.join(out_dir, "spoof_weight_arr.pt"))
        # spoof_output = torch.cat(
        #     (
        #         torch.clamp(spoof_generated_batch, 0, 1),
        #         torch.unsqueeze(target, 1),
        #         spoof_cond[:, :, :3, ...],
        #     ),
        #     dim=1,
        # )

    if extrapolate:
        print("Running extrapolate image generation...")
        target = val_vis_data["target"].to(device)
        cond = val_vis_data["cond"].to(device)
        # view_count = val_vis_data["view_count"].to(device)
        view_count = torch.randint(max_views + 1, 24, (target.shape[0],)).to(device)
        angle = val_vis_data["angle"].to(device)
        torch.save(target, os.path.join(out_dir, "target.pt"))
        torch.save(cond, os.path.join(out_dir, "cond.pt"))
        torch.save(view_count, os.path.join(out_dir, "view_count.pt"))
        # angle = val_vis_data["angle"].to(device)

        with torch.no_grad():
            _, generated_batch, logit_arr, weight_arr, _ = model(
                y_cond=cond,
                view_count=view_count,
                angle=angle,
                generate=True,
            )
        # print(out_dir)
        torch.save(generated_batch, os.path.join(out_dir, "generated_batch.pt"))
        torch.save(logit_arr, os.path.join(out_dir, "logit_arr.pt"))
        torch.save(weight_arr, os.path.join(out_dir, "weight_arr.pt"))

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

    if variable:
        print("Running variable view count...")
        variable_output = list()
        for i in range(1, torch.max(view_count)):
            print(f"Running view count {i}")
            variable_count = torch.full_like(view_count, i)
            (
                _,
                variable_generated_batch,
                variable_logit_arr,
                variable_weight_arr,
                _,
            ) = model(
                y_cond=cond,
                view_count=variable_count,
                angle=angle,
                generate=True,
            )
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

    if animate_generation:
        print("Running animation sequence generation...")
        i = 10  # class
        n = 96
        views = val_vis_data["images"].to(device)
        angles_incremental = torch.as_tensor([2 * np.pi / n * i for i in range(n)]).to(
            device
        )
        target = torch.repeat_interleave(views[i], n // 24, dim=0)
        cond_views = torch.stack([views[i, ::4]] * target.shape[0], dim=0)
        view_counts = torch.as_tensor(
            [cond_views.shape[1] for _ in range(target.shape[0])],
        ).to(device)
        print(cond_views.shape)
        print(angles_incremental.shape)
        print(view_counts.shape)
        print(target.shape)

        _, generated_batch, logit_arr, weight_arr, _ = model(
            y_cond=cond_views,
            angle=angles_incremental.unsqueeze(1),
            view_count=view_counts,
            generate=True,
        )

        mask = (
            torch.stack(
                [torch.stack([target] * weight_arr.shape[2], dim=1)]
                * weight_arr.shape[1],
                dim=1,
            )
            != 1.0
        )
        weight_masked = weight_arr * mask

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
            view_weights = rearrange(view_weights, "c h w -> h w c")
            frames.append((view_weights.cpu() * 255).to(torch.uint8))

        imageio.mimsave(
            os.path.join(out_dir, "output_fast.gif"), frames, "GIF", duration=0.1
        )

        frames = []
        for i in range(n):
            target_grid = torch.cat(
                [
                    target[i, ...][None, ...],
                ]
                * (cond_views.shape[1] + 1)
            )[None, ...]
            view_weights = torch.cat(
                (weight_masked[i, ...], cond_views[i][None, ...]), dim=0
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
                scale_each=True,
            )
            view_weights = rearrange(view_weights, "c h w -> h w c")
            frames.append(view_weights.cpu())

        imageio.mimsave(
            os.path.join(out_dir, "output_masked_fast.gif"), frames, "GIF", duration=0.1
        )

    if plausible:
        print("Running plausible image generation...")
        target = val_vis_data["images"][:, 12:13].to(device)
        cond = val_vis_data["images"][:, :1].to(device)
        # view_count = val_vis_data["view_count"].to(device)
        view_count = torch.full((target.shape[0],), 1).to(device)
        # angle = val_vis_data["angle"].to(device)
        angle = torch.full((target.shape[0], 1), 2 * np.pi / 24 * 12).to(device)
        torch.save(target, os.path.join(out_dir, "target_plausible.pt"))
        torch.save(cond, os.path.join(out_dir, "cond_plausible.pt"))
        torch.save(view_count, os.path.join(out_dir, "view_count_plausible.pt"))
        # angle = val_vis_data["angle"].to(device)

        print(cond.shape, view_count.shape, angle.shape)
        for i in range(6):
            with torch.no_grad():
                _, generated_batch, logit_arr, weight_arr, _ = model(
                    y_cond=cond,
                    view_count=view_count,
                    angle=angle,
                    generate=True,
                )
            # print(out_dir)
            torch.save(
                generated_batch,
                os.path.join(out_dir, f"generated_batch_plausible_{i}.pt"),
            )
            torch.save(logit_arr, os.path.join(out_dir, f"logit_arr_plausible_{i}.pt"))
            torch.save(
                weight_arr, os.path.join(out_dir, f"weight_arr_plausible_{i}.pt")
            )

        # cond_padded = torch.nn.utils.rnn.pad_sequence(
        #     [cond[i, :view_idx] for i, view_idx in enumerate(view_count)],
        #     batch_first=True,
        # )

        # output = torch.cat(
        #     (
        #         torch.clamp(generated_batch, 0, 1),
        #         torch.unsqueeze(target, 1),
        #         cond_padded,
        #     ),
        #     dim=1,
        # )

    if fill_missing:
        view_count = torch.randint(4, 9, (target.shape[0],))
        ordering = 0

    if autoregressive:
        os.makedirs(os.path.join(out_dir, "ar"))
        all_views = val_vis_data["images"][10:11].to(device)
        cond = all_views[:, :1].to(device)
        angles_incremental = torch.as_tensor(
            [2 * np.pi / 24 * i for i in range(1, 25)]
        ).to(device)
        torch.save(
            all_views,
            os.path.join(out_dir, "ar", f"all_views.pt"),
        )
        torch.save(
            cond,
            os.path.join(out_dir, "ar", f"cond_0.pt"),
        )

        for count, angle in enumerate(angles_incremental, start=1):
            view_count = torch.full((cond.shape[0],), count).to(device)
            angle = torch.full((cond.shape[0], 1), angle).to(device)
            *_, generated_samples = model(
                y_cond=cond, view_count=view_count, angle=angle, generate=True
            )
            cond = torch.cat((cond, generated_samples[:, None, ...]), dim=1)
            torch.save(
                cond,
                os.path.join(out_dir, "ar", f"cond_{count}.pt"),
            )
            torch.save(
                generated_samples,
                os.path.join(out_dir, "ar", f"samples_{count}.pt"),
            )

    if args.wandb:
        log_dict["inference_output"] = wandb.Image(
            make_grid(
                rearrange(output, "b s c h w -> (b s) c h w"),
                nrow=output.shape[1],
                scale_each=True,
            ),
            caption="Denoising steps, Target, Input View",
        )

        if variable:
            for i, variable_generated_batch in enumerate(variable_output, start=1):
                log_dict[f"variable_output_{i}"] = wandb.Image(
                    make_grid(
                        rearrange(variable_generated_batch, "b s c h w -> (b s) c h w"),
                        nrow=output.shape[1],
                        scale_each=True,
                    ),
                    caption="Denoising steps, Target, Input View",
                )
        wandb.log(log_dict)

    exit(0)


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()

    inference(args)
