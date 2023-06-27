import math
import torch
import numpy as np

from inspect import isfunction
from einops import rearrange
from functools import partial
from tqdm import tqdm
from torch import nn
from models.openai_unet.unet import UNetModel
from models.unet import UNet

from cleanfid import fid


class PaletteViewSynthesis(nn.Module):
    def __init__(self, unet, beta_schedule, **kwargs):
        super(PaletteViewSynthesis, self).__init__(**kwargs)

        self.unet = UNet(**unet)
        self.beta_schedule = beta_schedule

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, device=torch.device("cuda"), phase="train"):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**self.beta_schedule[phase])
        betas = (
            betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        )
        alphas = 1.0 - betas

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1.0, gammas[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("gammas", to_torch(gammas))
        self.register_buffer("sqrt_recip_gammas", to_torch(np.sqrt(1.0 / gammas)))
        self.register_buffer("sqrt_recipm1_gammas", to_torch(np.sqrt(1.0 / gammas - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - gammas_prev) / (1.0 - gammas)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(gammas_prev) / (1.0 - gammas)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch((1.0 - gammas_prev) * np.sqrt(alphas) / (1.0 - gammas)),
        )

    def predict_start_from_noise(self, y_t, t, noise):
        return (
            extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t
            - extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def q_posterior(self, y_0_hat, y_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat
            + extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, y_t.shape
        )
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool):
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        y_0_hat = self.predict_start_from_noise(
            y_t,
            t=t,
            noise=self.unet(y_t, noise_level),
        )

        if clip_denoised:
            y_0_hat.clamp_(-1.0, 1.0)

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t
        )
        return model_mean, posterior_log_variance

    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_0))
        return sample_gammas.sqrt() * y_0 + (1 - sample_gammas).sqrt() * noise

    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True):
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised
        )
        noise = torch.randn_like(y_t) if any(t > 0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def q_posterior_sample(self, y_0, y_t, t):
        model_mean, model_log_variance = self.q_posterior(y_0, y_t, t)
        noise = torch.randn_like(y_t) if any(t > 0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def generate(self, images, masked_cnt=6, sample_num=8, single=True):
        if single:
            masked_ids = 1
        else:
            masked_ids = torch.randperm(24)[:masked_cnt]
        mask = torch.zeros_like(images)
        mask[:, masked_ids, :, :, :] = 1
        mask = rearrange(mask, "b v c h w -> b (v c) h w")
        y_0 = rearrange(images, "b v c h w -> b (v c) h w")

        b, *_ = y_0.shape

        assert (
            self.num_timesteps > sample_num
        ), "num_timesteps must greater than sample_num"
        sample_inter = self.num_timesteps // sample_num

        y_t = torch.randn_like(y_0)
        y_scaled = torch.clip(y_t, 0, 1)
        ret_arr = y_scaled
        for i in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            t = torch.full((b,), i, device=y_0.device, dtype=torch.long)
            y_t_unk = self.p_sample(y_t, t)
            y_t_known = self.q_posterior_sample(y_0, y_t, t)
            if mask is not None:
                y_t = y_t_known * (1.0 - mask) + mask * y_t_unk
            if i % sample_inter == 0:
                y_scaled = torch.clip(y_t, 0, 1)
                ret_arr = torch.cat([ret_arr, y_scaled], dim=0)

        ret_arr = rearrange(
            ret_arr,
            "(s b) (v c) h w -> b s v c h w",
            b=images.shape[0],
            s=sample_num + 1,
            v=images.shape[1],
            c=images.shape[2],
        )
        generated_samples = ret_arr[:, -1, masked_ids, ...]
        ground_truth = images[:, masked_ids, ...]
        return y_t, ret_arr, generated_samples, ground_truth

    def forward(self, y_0, mask=None, noise=None):
        # sampling from p(gammas)
        y_0 = rearrange(y_0, "b v c h w -> b (v c) h w")

        b, *_ = y_0.shape
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
        gamma_t1 = extract(self.gammas, t - 1, x_shape=(1, 1))
        sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        sample_gammas = (sqrt_gamma_t2 - gamma_t1) * torch.rand(
            (b, 1), device=y_0.device
        ) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(y_0))
        y_noisy = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise
        )

        noise_hat = self.unet(y_noisy, sample_gammas)
        loss = self.loss_fn(noise, noise_hat)
        return loss


# gaussian diffusion trainer class
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape=(1, 1, 1, 1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# beta_schedule function
def _warmup_beta(linear_start, linear_end, num_timesteps, warmup_frac):
    betas = linear_end * np.ones(num_timesteps, dtype=np.float64)
    warmup_time = int(num_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64
    )
    return betas


def make_beta_schedule(
    schedule, num_timesteps, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3
):
    if schedule == "quad":
        betas = (
            np.linspace(
                linear_start**0.5, linear_end**0.5, num_timesteps, dtype=np.float64
            )
            ** 2
        )
    elif schedule == "linear":
        betas = np.linspace(linear_start, linear_end, num_timesteps, dtype=np.float64)
    elif schedule == "warmup10":
        betas = _warmup_beta(linear_start, linear_end, num_timesteps, 0.1)
    elif schedule == "warmup50":
        betas = _warmup_beta(linear_start, linear_end, num_timesteps, 0.5)
    elif schedule == "const":
        betas = linear_end * np.ones(num_timesteps, dtype=np.float64)
    elif schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_timesteps, 1, num_timesteps, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(num_timesteps + 1, dtype=torch.float64) / num_timesteps
            + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas
