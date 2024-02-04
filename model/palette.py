import math
import torch
import numpy as np
import torch.nn.functional as F

from inspect import isfunction
from einops import rearrange
from functools import partial
from tqdm import tqdm
from torch import nn


class PaletteViewSynthesis(nn.Module):
    def __init__(self, denoise_fn, beta_schedule, **kwargs):
        super(PaletteViewSynthesis, self).__init__(**kwargs)

        self.denoise_fn = denoise_fn
        self.beta_schedule = beta_schedule
        self.loss_fn = F.mse_loss

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

    def p_mean_variance(
        self,
        y_t,
        y_cond,
        view_count,
        angle,
        t,
        clip_denoised: bool,
    ):
        view_delimiters = torch.cumsum(view_count, 0).tolist()
        view_delimiters.insert(0, 0)

        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)

        # prepare shapes of conditioning, noise, angles and levels;
        # from b 23 c h w ->  (v_1 + ... + v_b) c h w; where v_n is cond view count for each sample
        y_cond_stacked = torch.concatenate(
            [y_cond[i, :idx] for i, idx in enumerate(view_count)],
            dim=0,
        ).to(y_cond.device)

        y_t_stacked = torch.repeat_interleave(y_t, view_count, dim=0)
        noise_level_stacked = torch.repeat_interleave(noise_level, view_count, dim=0)
        angle_stacked = torch.repeat_interleave(angle, view_count, dim=0)

        denoise_output = self.denoise_fn(
            torch.cat([y_cond_stacked, y_t_stacked], dim=1),
            angle_stacked,
            noise_level_stacked,
        )
        noise_all, logits = denoise_output[:, :3, ...], denoise_output[:, 3:, ...]

        # weights and noise padded; shape b max(v_1, ... , v_b) c h w
        logits_padded = torch.nn.utils.rnn.pad_sequence(
            [
                logits[idx1:idx2]
                for idx1, idx2 in zip(view_delimiters[:-1], view_delimiters[1:])
            ],
            batch_first=True,
            padding_value=float("-inf"),
        )
        weights_softmax = F.softmax(logits_padded, dim=1)
        noise_padded = torch.nn.utils.rnn.pad_sequence(
            [
                noise_all[idx1:idx2]
                for idx1, idx2 in zip(view_delimiters[:-1], view_delimiters[1:])
            ],
            batch_first=True,
        )
        noise_weighted = noise_padded * weights_softmax

        noise = noise_weighted.sum(dim=1)

        y_0_hat = self.predict_start_from_noise(y_t, t=t, noise=noise)

        if clip_denoised:
            y_0_hat.clamp_(-1.0, 1.0)

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t
        )
        return model_mean, posterior_log_variance, logits, weights_softmax

    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_0))
        return sample_gammas.sqrt() * y_0 + (1 - sample_gammas).sqrt() * noise

    @torch.no_grad()
    def p_sample(self, y_t, y_cond, view_count, angle, t, clip_denoised=True):
        model_mean, model_log_variance, logits, weights = self.p_mean_variance(
            y_t,
            y_cond,
            view_count,
            angle,
            t,
            clip_denoised=clip_denoised,
        )
        noise = torch.randn_like(y_t) if any(t > 0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp(), logits, weights

    @torch.no_grad()
    def generate(self, y_cond, view_count, angle, y_t=None, sample_num=8):
        b, *_ = y_cond.shape

        assert (
            self.num_timesteps > sample_num
        ), "num_timesteps must greater than sample_num"
        sample_inter = self.num_timesteps // sample_num

        y_t = default(
            y_t, lambda: torch.randn_like(y_cond[:, :1, :3, ...]).squeeze(dim=1)
        )
        ret_arr = [
            y_t,
        ]
        weight_arr = list()
        logit_arr = list()
        for i in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            # for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
            y_t, logits, weights = self.p_sample(y_t, y_cond, view_count, angle, t)
            if i % sample_inter == 0:
                # ret_arr = torch.cat([ret_arr, y_t], dim=0)
                ret_arr.append(y_t)
                logit_arr.append(logits)
                weight_arr.append(weights)
                # weight_arr = torch.cat([weight_arr, weights], dim=0)

        ret_arr = torch.stack(ret_arr, dim=1)
        logit_arr = torch.stack(logit_arr, dim=1)
        weight_arr = torch.stack(weight_arr, dim=1)
        # ret_arr = rearrange(
        #     ret_arr,
        #     "(b s) c h w -> b s c h w",
        #     b=b,
        # )
        # weight_arr = rearrange(weight_arr, "(s b) c h w -> b s c h w", b=b)
        generated_samples = ret_arr[:, -1, ...]

        return y_t, ret_arr, logit_arr, weight_arr, generated_samples

    def forward(
        self,
        y_0=None,
        y_cond=None,
        view_count=None,
        angle=0.0,
        noise=None,
        generate=False,
    ):
        # generate() wrapped in forward for DDP
        if generate:
            return self.generate(y_cond, view_count, angle)

        # sampling from p(gammas)
        b = y_0.shape[0]
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

        view_delimiters = torch.cumsum(view_count, 0).tolist()
        view_delimiters.insert(0, 0)

        # prepare shapes of conditioning, noise, angles and levels;
        # from b 23 c h w ->  (v_1 + ... + v_b) c h w; where v_n is cond view count for each sample
        y_cond_stacked = torch.concatenate(
            [y_cond[i, :upto] for i, upto in enumerate(view_count)], dim=0
        ).to(y_cond.device)
        y_noisy_stacked = torch.repeat_interleave(y_noisy, view_count, dim=0)
        sample_gammas_stacked = torch.repeat_interleave(
            sample_gammas, view_count, dim=0
        )
        angle_stacked = torch.repeat_interleave(angle, view_count, dim=0)

        # u-net outputs (v_1 + ... + v_b) c h w; where v_n is cond view count for each sample
        denoise_output = self.denoise_fn(
            torch.cat([y_cond_stacked, y_noisy_stacked], dim=1),
            angle_stacked,
            sample_gammas_stacked,
        )
        noise_all, weights = denoise_output[:, :3, ...], denoise_output[:, 3:, ...]

        # weights and noise padded; shape b max(v_1, ... , v_b) c h w
        weights_padded = torch.nn.utils.rnn.pad_sequence(
            [
                weights[idx1:idx2]
                for idx1, idx2 in zip(view_delimiters[:-1], view_delimiters[1:])
            ],
            batch_first=True,
            padding_value=float("-inf"),
        )
        weights_softmax = F.softmax(weights_padded, dim=1)
        noise_padded = torch.nn.utils.rnn.pad_sequence(
            [
                noise_all[idx1:idx2]
                for idx1, idx2 in zip(view_delimiters[:-1], view_delimiters[1:])
            ],
            batch_first=True,
        )
        noise_weighted = noise_padded * weights_softmax

        noise_hat = noise_weighted.sum(dim=1)

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
