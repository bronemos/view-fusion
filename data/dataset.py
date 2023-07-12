import numpy as np
import imageio
import yaml
import webdataset as wds
from io import BytesIO

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


def create_webdataset(path, mode, start_shard=0, end_shard=12, single_view=False):
    def process_sample(sample, single_view=False):
        if single_view:
            images_idx = np.sort(np.random.choice(range(24), 2, replace=False))
            images = [sample[f"{i:04d}.png"] for i in images_idx]
            images = np.stack(images, 0).astype(np.float32)
            angle = 2 * np.pi / 24 * (images_idx[1] - images_idx[0])
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

        images = [sample[f"{i:04d}.png"] for i in range(24)]
        images = np.stack(images, 0)  # .astype(np.float32)

        images = rearrange(images, "v h w c -> v c h w")  # 2 * ... -1

        result = {
            "images": images,
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

    return (
        webdataset.shuffle(100)
        .decode("rgb")
        .map(lambda x: process_sample(x, single_view))
    )


class NMRShardedDataset(Dataset):
    def __init__(
        self,
        path,
        mode,
        start_shard=0,
        end_shard=12,
        view_count=12,
        max_len=None,
        canonical_view=True,
    ):
        """Loads the NMR dataset as found at
        https://s3.eu-central-1.amazonaws.com/avg-projects/differentiable_volumetric_rendering/data/NMR_Dataset.zip
        Hosted by Niemeyer et al. (https://github.com/autonomousvision/differentiable_volumetric_rendering)
        Args:
            path (str): Path to dataset.
            mode (str): 'train', 'val', or 'test'.
            points_per_item (int): Number of target points per scene.
            max_len (int): Limit to the number of entries in the dataset.
            canonical_view (bool): Return data in canonical camera coordinates (like in SRT), as opposed
                to world coordinates.
            full_scale (bool): Return all available target points, instead of sampling.
        """

        self.path = path
        self.mode = mode
        self.view_count = view_count
        self.max_len = max_len
        self.canonical = canonical_view
        if start_shard == end_shard:
            self.dataset = list(
                (
                    wds.WebDataset(
                        os.path.join(
                            path,
                            f"NMR-{mode}-{start_shard:02d}.tar",
                        ),
                        shardshuffle=True,
                    )
                    .shuffle(100)
                    .decode("rgb")
                )
            )
        else:
            self.dataset = list(
                (
                    wds.WebDataset(
                        os.path.join(
                            path,
                            f"NMR-{mode}-{{{start_shard:02d}..{end_shard:02d}}}.tar",
                        ),
                        shardshuffle=True,
                    )
                    .shuffle(100)
                    .decode("rgb")
                )
            )
        self.num_records = len(self.dataset)

        self.render_kwargs = {"min_dist": 2.0, "max_dist": 4.0}

        # Rotation matrix making z=0 is the ground plane.
        # Ensures that the scenes are layed out in the same way as the other datasets,
        # which is convenient for visualization.
        self.rot_mat = np.array(
            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
        )

    def __len__(self):
        return self.num_records

    def __getitem__(self, idx):
        self.sample = self.dataset[idx]

        images = [self.sample[f"{i:04d}.png"] for i in range(24)]
        images = np.stack(images, 0).astype(np.float32) / 255.0

        images = rearrange(images, "b h w c -> (b c) h w")

        result = {
            "images": images,
            "scene_hash": self.sample["__key__"],
        }

        return result
