import torch
import torch.utils.data
from pytorch_msssim import ssim


def compute_psnr(generated, target):
    mse = torch.mean((generated - target) ** 2, dim=(1, 2, 3))
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def compute_ssim(generated, target):
    return ssim(generated, target, data_range=1.0, size_average=False)
