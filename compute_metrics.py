from cleanfid import fid
from utils.metrics import compute_psnr, compute_ssim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import lpips
import os

base_path = "/scratch/work/spieglb1/palette-view-synthesis/logs/2024-01-17T15-13-15-multi-view-composable-variable-small-v100-eval-new/view_count_test_1_2024-02-02T09-22-28"
src = os.path.join(base_path, "images-710001")
dataset = ImageFolder(
    src,
    transform=transforms.ToTensor(),
)
total = len(dataset)
print(total)

dataloader = DataLoader(
    dataset,
    batch_size=2187,
)

gts = list()
generated = list()
for i, (batch, label) in enumerate(dataloader, start=1):
    if i * 2187 > (total // 2):
        gts.append(batch)
        # print(label == 1)
    else:
        generated.append(batch)
        # print(label == 0)


loss_fn_alex = lpips.LPIPS(net="vgg")
psnrs = list()
ssims = list()
lpipss = list()
for gen, target in zip(generated, gts):
    psnrs.append(compute_psnr(gen, target))
    ssims.append(compute_ssim(gen, target))
    lpipss.append(loss_fn_alex(2 * gen - 1, 2 * target - 1))

print(torch.cat(ssims).mean())
print(torch.cat(psnrs).mean())
print(torch.cat(lpipss).mean())
