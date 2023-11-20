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


def transform_points(points, transform, translate=True):
    """Apply linear transform to a np array of points.
    Args:
        points (np array [..., 3]): Points to transform.
        transform (np array [3, 4] or [4, 4]): Linear map.
        translate (bool): If false, do not apply translation component of transform.
    Returns:
        transformed points (np array [..., 3])
    """
    # Append ones or zeros to get homogenous coordinates
    if translate:
        constant_term = np.ones_like(points[..., :1])
    else:
        constant_term = np.zeros_like(points[..., :1])
    points = np.concatenate((points, constant_term), axis=-1)

    points = np.einsum("nm,...m->...n", transform, points)
    return points[..., :3]


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

    return webdataset.shuffle(100).decode("rgb").map(lambda x: process_sample(x))


def create_webdataset(path, mode, start_shard=0, end_shard=12, view_cnt=1):
    def process_sample(sample):
        view_cnt = np.random.randint(1, 24)
        images_idx = np.random.choice(range(24), view_cnt + 1, replace=False)
        # if view_cnt > 1:
        #    images_idx = np.sort(images_idx)
        images = [sample[f"{i:04d}.png"] for i in images_idx]
        images = np.stack(images, 0).astype(np.float32)
        # angle = 2 * np.pi / 24 * (images_idx[1] - images_idx[0])
        angle = 2 * np.pi / 24 * images_idx[0]
        sin_angle = (np.full(images.shape[1:3], np.sin(angle)) + 1) / 2
        cos_angle = (np.full(images.shape[1:3], np.cos(angle)) + 1) / 2

        # images = np.concatenate((images, sin_angle, cos_angle), axis=0)
        angles = np.stack((sin_angle, cos_angle), 0).astype(np.float32)

        # angle = images_idx[0] / 24
        # angles = np.full(images.shape[1:3], angle)

        images = rearrange(images, "v h w c -> v c h w")  # 2 * ... -1
        # print(images.shape)
        # avg_weights = 24 - np.abs(images_idx[1:] - images_idx[0])
        # cond_avg = np.average(images[1:], axis=0, weights=avg_weights)
        # print(cond_avg.shape)

        cond = np.concatenate(
            (images[1:], np.repeat(angles[None, ...], view_cnt, axis=0)), axis=1
        ).astype(np.float32)
        padding = np.zeros((23 - view_cnt, cond.shape[1], cond.shape[2], cond.shape[3]))
        cond = np.concatenate((cond, padding), axis=0).astype(np.float32)

        result = {
            "view": images[0],
            "cond": cond,
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
        )

    else:
        webdataset = wds.WebDataset(
            os.path.join(
                path,
                f"NMR-{mode}-{{{start_shard:02d}..{end_shard:02d}}}.tar",
            ),
            shardshuffle=True,
        )

    return webdataset.shuffle(100).decode("rgb").map(lambda x: process_sample(x))


def collate_variable_length(batch):
    print(batch[0].keys())
    padded_cond = torch.nn.utils.rnn.pad_sequence(
        [cond_view for cond_view in batch["cond"]], batch_first=True
    )
    batch["cond"] = padded_cond
    return batch


def create_webdataset_dit(path, mode, start_shard=0, end_shard=12, view_cnt=1):
    def process_sample(sample, canonical=True):
        rot_mat = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

        input_views = np.random.choice(np.arange(24), size=view_cnt, replace=False)
        target_views = np.array(list(set(range(24)) - set(input_views)))

        images = [sample[f"{i:04d}.png"] for i in range(24)]
        images = np.stack(images, 0).astype(np.float32)
        input_images = np.transpose(images[input_views], (0, 3, 1, 2))
        target_images = np.transpose(images[target_views], (0, 3, 1, 2))

        cameras = np.load(BytesIO(sample["cameras"]))
        cameras = {k: v for k, v in cameras.items()}  # Load all matrices into memory

        for i in range(24):  # Apply rotation matrix to rotate coordinate system
            cameras[f"world_mat_inv_{i}"] = rot_mat @ cameras[f"world_mat_inv_{i}"]
            # The transpose here is not technically necessary, since the rotation matrix is symmetric
            cameras[f"world_mat_{i}"] = cameras[f"world_mat_{i}"] @ np.transpose(
                rot_mat
            )

        rays = []
        height = width = 64

        xmap = np.linspace(-1, 1, width)
        ymap = np.linspace(-1, 1, height)
        xmap, ymap = np.meshgrid(xmap, ymap)

        for i in range(24):
            cur_rays = np.stack((xmap, ymap, np.ones_like(xmap)), -1)
            cur_rays = transform_points(
                cur_rays,
                cameras[f"world_mat_inv_{i}"] @ cameras[f"camera_mat_inv_{i}"],
                translate=False,
            )
            cur_rays = cur_rays[..., :3]
            cur_rays = cur_rays / np.linalg.norm(cur_rays, axis=-1, keepdims=True)
            rays.append(cur_rays)

        rays = np.stack(rays, axis=0).astype(np.float32)
        camera_pos = [cameras[f"world_mat_inv_{i}"][:3, -1] for i in range(24)]
        camera_pos = np.stack(camera_pos, axis=0).astype(np.float32)
        # camera_pos and rays are now in world coordinates.

        if canonical:  # Transform to canonical camera coordinates
            canonical_extrinsic = cameras[f"world_mat_{input_views[0]}"].astype(
                np.float32
            )
            camera_pos = transform_points(camera_pos, canonical_extrinsic)
            rays = transform_points(rays, canonical_extrinsic, translate=False)

        result = {
            "input_images": input_images,  # [3, h, w]
            "input_camera_pos": camera_pos[input_views],  # [v, 3]
            "input_rays": rays[input_views],  # [v, h, w, 3]
            "target_images": target_images,
            "target_camera_pos": camera_pos[target_views],  # [24 - v, 3]
            "target_rays": rays[target_views],  # [24 - v, h, w, 3]
            "sceneid": process_sample.idx,  # int
            "scene_hash": sample["__key__"],
        }

        if canonical:
            result["transform"] = canonical_extrinsic  # [3, 4] (optional)

        process_sample.idx += 1

        return result

    process_sample.idx = 0

    if start_shard == end_shard:
        webdataset = wds.WebDataset(
            os.path.join(
                path,
                f"NMR-{mode}-{start_shard:02d}.tar",
            ),
            shardshuffle=True,
        )

    else:
        webdataset = wds.WebDataset(
            os.path.join(
                path,
                f"NMR-{mode}-{{{start_shard:02d}..{end_shard:02d}}}.tar",
            ),
            shardshuffle=True,
        )

    return webdataset.shuffle(100).decode("rgb").map(lambda x: process_sample(x))
