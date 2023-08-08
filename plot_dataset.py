import torch

from torchvision.utils import make_grid, save_image
from einops import rearrange

from data.dataset import create_webdataset, create_webdataset_metzler

if __name__ == "__main__":
    for shard_idx in range(13):
        dataset = create_webdataset(
            path="/scratch/work/spieglb1/datasets/NMR_sharded",
            mode="train",
            start_shard=shard_idx,
            end_shard=shard_idx,
        )
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

        visualization_data = next(iter(data_loader))
        views = visualization_data["images"]
        if shard_idx == 0:
            images = views
        else:
            images = torch.cat((images, views), dim=0)

    save_image(
        make_grid(rearrange(images, "b v c h w -> (v b) c h w"), nrow=images.shape[0]),
        f"output.png",
    )
