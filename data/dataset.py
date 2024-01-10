import random
import numpy as np
import imageio
import yaml
import torch
import webdataset as wds
from io import BytesIO
import cv2

from torch.utils.data import IterableDataset, Dataset
from einops import rearrange


import os


def create_webdataset_metzler(path):
    def process_sample(sample, single_view=True):
        if single_view:
            images_idx = np.sort(np.random.choice(range(34), 2, replace=False))
            images = [
                cv2.resize(
                    sample[f"{i:03d}.png"],
                    dsize=(64, 64),
                    interpolation=cv2.INTER_CUBIC,
                )
                for i in images_idx
            ]
            images = np.stack(images, 0).astype(np.float32)
            angle = 2 * np.pi / 34 * (images_idx[1] - images_idx[0])
            sin_angle = (np.full(images.shape[1:3], np.sin(angle)) + 1) / 2
            cos_angle = (np.full(images.shape[1:3], np.cos(angle)) + 1) / 2

            # images = np.concatenate((images, sin_angle, cos_angle), axis=0)
            angles = np.stack((sin_angle, cos_angle), 0).astype(np.float32)

            images = rearrange(images, "v h w c -> v c h w")  # 2 * ... -1

            cond = np.concatenate((images[1], angles), axis=0).astype(np.float32)

            result = {
                "view": images[0],
                "cond": cond,
                "scene_hash": sample["__key__"],
            }

            return result

        images = [sample[f"{i:03d}.png"] for i in range(34)]
        images = np.stack(images, 0)  # .astype(np.float32)

        images = rearrange(images, "v h w c -> v c h w")  # 2 * ... -1

        result = {
            "images": images,
            "scene_hash": sample["__key__"],
        }

        return result

    webdataset = wds.WebDataset(
        os.path.join(
            path,
            "metzler_dataset.tar",
        ),
        shardshuffle=True,
    )

    return webdataset.shuffle(1000).decode("rgb").map(lambda x: process_sample(x))


def create_webdataset(path, mode, start_shard=0, end_shard=12, **kwargs):
    def process_sample(sample):
        view_cnt = 23
        images_idx = np.arange(24)
        np.random.shuffle(images_idx)
        images = [sample[f"{i:04d}.png"] for i in images_idx]
        images = np.stack(images, 0).astype(np.float32)
        angle = 2 * np.pi / 24 * images_idx[0]
        sin_angle = (np.full(images.shape[1:3], np.sin(angle)) + 1) / 2
        cos_angle = (np.full(images.shape[1:3], np.cos(angle)) + 1) / 2

        # images = np.concatenate((images, sin_angle, cos_angle), axis=0)
        angles = np.stack((sin_angle, cos_angle), 0).astype(np.float32)

        images = rearrange(images, "v h w c -> v c h w")  # 2 * ... -1

        cond = np.concatenate(
            (images[1:], np.repeat(angles[None, ...], view_cnt, axis=0)), axis=1
        ).astype(np.float32)
        spoof_cond = np.concatenate(
            (images[:-1], np.repeat(angles[None, ...], view_cnt, axis=0)), axis=1
        ).astype(np.float32)
        if random.random() < 0.15:
            cond = spoof_cond

        # if random.random() < 0.15:
        #     np.random.shuffle(images_idx)
        all_views = [sample[f"{i:04d}.png"] for i in images_idx]
        all_views = np.stack(all_views, 0).astype(np.float32)
        all_views = rearrange(all_views, "v h w c -> v c h w")
        all_views = np.concatenate(
            (
                all_views,
                np.repeat(
                    angles[
                        None,
                        ...,
                    ],
                    24,
                    axis=0,
                ),
            ),
            axis=1,
        ).astype(np.float32)

        result = {
            "target": images[0],
            "cond": cond,
            "spoof_cond": spoof_cond,
            "all_views": all_views,
            "view_cnt": view_cnt,
            "angle": images_idx[0] / 24,
            "scene_hash": sample["__key__"],
        }

        return result

    if start_shard == end_shard:
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
        )

    return webdataset.shuffle(1000).decode("rgb").map(lambda x: process_sample(x))
