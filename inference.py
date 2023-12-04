import os
import torch

from torchvision.utils import make_grid, save_image
from einops import rearrange


def run_inference(args, model, val_loader, wandb):
    log_dict = dict()
    model.eval()

    images_path = os.path.join(tmp_dir, f"images-{it}-{now}")
    ground_truth_path = os.path.join(images_path, "ground-truth")
    generated_path = os.path.join(images_path, "generated")
    os.makedirs(images_path)
    os.makedirs(generated_path)
    os.makedirs(ground_truth_path)
    compute_fid = False
    generate = True
    if compute_fid:
        generated_batches = list()
        ground_truth_batches = list()

        for val_batch in val_loader:
            targets = val_batch["target"].to(device)
            cond = val_batch["cond"].to(device)
            # angle = batch["angle"].to(device)
            with torch.no_grad():
                *_, generated_samples = model(y_cond=cond, generate=True)
                generated_batches.append(generated_samples)
                ground_truth_batches.append(targets)

        cnt = 0
        for gt_batch, generated_batch in zip(ground_truth_batches, generated_batches):
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
        targets = val_vis_data["target"].to(device)
        cond = val_vis_data["cond"].to(device)
        # angle = val_vis_data["angle"].to(device)

        y_t, generated_batch, *_ = model(y_cond=cond, generate=True)

        output = torch.cat(
            (
                torch.clamp(generated_batch, 0, 1),
                torch.unsqueeze(targets, 1),
                cond[:, :, :3, ...],
            ),
            dim=1,
        )
        log_dict["inference_output"] = wandb.Image(
            make_grid(
                rearrange(output, "b s c h w -> (b s) c h w"),
                nrow=output.shape[1],
                scale_each=True,
            ),
            caption="Denoising steps, Target, Input View",
        )
    wandb.log(log_dict)

    exit(0)


if __name__ == "__main__":
    pass
