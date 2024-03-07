import os
from functools import partial

import numpy as np
import torch
import webdataset as wds
from einops import rearrange


def process_sample(sample, mode="train"):
    images_idx = np.arange(24)
    images = [sample[f"{i:04d}.png"] for i in images_idx]
    images = np.stack(images, 0).astype(np.float32)
    angle = np.asarray(
        [
            2 * np.pi / 24 * images_idx[0],
        ]
    ).astype(np.float32)

    images = rearrange(images, "v h w c -> v c h w")  # 2 * ... -1

    np.random.shuffle(images_idx)
    cond_images = images[images_idx]
    target = cond_images[0]

    # occasionally feed target image as conditioning during training (improves generalization)
    if np.random.random() < 0.1 and mode == "train":
        np.random.shuffle(images_idx)
        cond_images = cond_images[images_idx]

    relative_angle = np.asarray(
        [
            2 * np.pi / 24 * (images_idx[1] - images_idx[0]),
        ]
    ).astype(np.float32)

    refernece_images_stacked = np.repeat(images[1][np.newaxis, ...], 24, axis=0)
    relative_cond_images = np.concatenate((refernece_images_stacked, images), axis=1)

    result = {
        "target": target,
        "cond": cond_images[1:],
        "relative_cond": relative_cond_images[1:],
        "all_views": images,
        "angle": angle,
        "relative_angle": relative_angle,
        "scene_hash": sample["__key__"],
    }

    return result


def nodesplitter(urls):
    rank, world_size = (
        torch.distributed.get_rank(),
        torch.distributed.get_world_size(),
    )

    return urls[rank::world_size]


def create_webdataset(path, mode, start_shard=0, end_shard=3, **kwargs):
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        total_shard_count = end_shard - start_shard + 1
        assert (
            total_shard_count % world_size == 0
        ), "Shard count must be divisible by the number of GPUs!"

    if start_shard == end_shard:
        assert (
            torch.distributed.is_initialized() is False
        ), "Distributed training on a single shard is not supported!"
        webdataset = wds.WebDataset(
            os.path.join(
                path,
                f"NMR-{mode}-{start_shard:02d}.tar",
            ),
            shardshuffle=True,
            resampled=True,
        )

    else:
        webdataset = wds.WebDataset(
            os.path.join(
                path,
                f"NMR-{mode}-{{{start_shard:02d}..{end_shard:02d}}}.tar",
            ),
            shardshuffle=True,
            resampled=True,
            nodesplitter=nodesplitter if torch.distributed.is_initialized() else None,
        )

    return (
        webdataset.shuffle(1000).decode("rgb").map(partial(process_sample, mode=mode))
    )
