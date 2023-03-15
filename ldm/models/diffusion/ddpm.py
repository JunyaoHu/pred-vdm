"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from math import ceil
from torchvision.utils import make_grid

from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler

from models.fvd.calculate_fvd import calculate_fvd1
from models.fvd.calculate_psnr import calculate_psnr1
from models.fvd.calculate_ssim import calculate_ssim1
from models.fvd.calculate_lpips import calculate_lpips1

import torch.nn.functional as F
import mediapy as media
from omegaconf import OmegaConf
import time


__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

# pytorch-lightning has been implemented it by 'reduce'
# def uniform_on_device(r1, r2, shape, device):
#     return (r1 - r2) * torch.rand(*shape, device=device) + r2


class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in [video] space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="video",
                 image_size=256,
                 channels=3,
                 VQ_batch_size=128,
                 log_every_t=100,
                 clip_denoised=False,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 original_elbo_weight=0.2,
                 l_eps_weight=0.8,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                #  use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.VQ_batch_size = VQ_batch_size
        self.channels = channels
        # self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_eps_weight = l_eps_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)


    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            # if context is not None:
            #     rank_zero_info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                # if context is not None:
                #     rank_zero_info(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        # model_out = self.model(x, t)
        # if self.parameterization == "eps":
        #     x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        # elif self.parameterization == "x0":
        #     x_recon = model_out
        # if clip_denoised:
        #     x_recon.clamp_(-1., 1.)

        # model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        # return model_mean, posterior_variance, posterior_log_variance

        # ### it has been rewritten in LatentDiffusion ###
        pass

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=False, repeat_noise=False):
        # b, *_, device = *x.shape, x.device
        # model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        # noise = noise_like(x.shape, device, repeat_noise)
        # # no noise when t == 0
        # nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        # return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        # ### it has been rewritten in LatentDiffusion ###
        pass

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        # device = self.betas.device
        # b = shape[0]
        # img = torch.randn(shape, device=device)
        # intermediates = [img]
        # for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
        #     img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
        #                         clip_denoised=self.clip_denoised)
        #     if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
        #         intermediates.append(img)
        # if return_intermediates:
        #     return img, intermediates
        # return img

        # ### it has been rewritten in LatentDiffusion ###
        pass

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        # image_size = self.image_size
        # channels = self.channels
        # return self.p_sample_loop((batch_size, channels, image_size, image_size),
        #                           return_intermediates=return_intermediates)

        # ### it has been rewritten in LatentDiffusion ###
        pass

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, x_start_pixel, cond, t, noise=None):
        # noise = default(noise, lambda: torch.randn_like(x_start))
        # x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # model_out = self.model(x_noisy, t)

        # loss_dict = {}
        # if self.parameterization == "eps":
        #     target = noise
        # elif self.parameterization == "x0":
        #     target = x_start
        # else:
        #     raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        # loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        # log_prefix = 'train' if self.training else 'val'

        # loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        # loss_simple = loss.mean() * self.l_simple_weight

        # loss_vlb = (self.lvlb_weights[t] * loss).mean()
        # loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        # loss = loss_simple + self.original_elbo_weight * loss_vlb

        # loss_dict.update({f'{log_prefix}/loss': loss})

        # return loss, loss_dict

        # ### it has been rewritten in LatentDiffusion ###
        pass

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        # t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        # return self.p_losses(x, t, *args, **kwargs)

        # ### it has been rewritten in LatentDiffusion ###
        pass
    
    def get_input(self, batch, key, bthwc2btchw=False):
        # batch["video"] wiill get video tensor batch
        # when channel = 1, len(x.shape) == 2, should add a channel dim, but we all 3 channel
        # if len(x.shape) == 3:
        #     x = x[..., None]
        
        x = batch[key]
        if bthwc2btchw:
            x = rearrange(x, 'b t h w c -> b t c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        
        return x

    def shared_step(self, batch):
        # x = self.get_input(batch, self.first_stage_key)
        # loss, loss_dict = self(x)
        # return loss, loss_dict

        # ### it has been rewritten in LatentDiffusion ###
        pass


    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)
        self.log_dict(loss_dict, prog_bar=False,logger=True, on_step=True, on_epoch=False)
        self.log("global_step", self.global_step*1.0, prog_bar=True, logger=False, on_step=True, on_epoch=False)
        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=False, logger=True, on_step=True, on_epoch=False)

        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        rank_zero_info("Validation_step: no EMA")
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope("validation_step: "):
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        pass
    #     rank_zero_info("Test_step: no EMA")
    #     _, loss_dict_no_ema = self.shared_step(batch)
    #     with self.ema_scope("Test_step: "):
    #         _, loss_dict_ema = self.shared_step(batch)
    #         loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
    #     self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
    #     self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_videos(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        # log = dict()
        # x = self.get_input(batch, self.first_stage_key)
        # N = min(x.shape[0], N)
        # n_row = min(x.shape[0], n_row)
        # x = x.to(self.device)[:N]
        # log["inputs"] = x

        # # get diffusion row
        # diffusion_row = list()
        # x_start = x[:n_row]

        # for t in range(self.num_timesteps):
        #     if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
        #         t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
        #         t = t.to(self.device).long()
        #         noise = torch.randn_like(x_start)
        #         x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        #         diffusion_row.append(x_noisy)

        # log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        # if sample:
        #     # get denoise row
        #     with self.ema_scope("Plotting"):
        #         samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

        #     log["samples"] = samples
        #     log["ddim200_row"] = self._get_rows_from_list(denoise_row)

        # if return_keys:
        #     if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
        #         return log
        #     else:
        #         return {key: log[key] for key in return_keys}
        # return log

        # ### it has been rewritten in LatentDiffusion ###
        pass

    def configure_optimizers(self):
        # lr = self.learning_rate
        # """"""
        # params = list(self.model.parameters())
        # if self.learn_logvar:
        #     params = params + [self.logvar]
        # opt = torch.optim.AdamW(params, lr=lr)
        # return opt

        # ### it has been rewritten in LatentDiffusion ###
        pass


class LatentDiffusion(DDPM):
    """main class"""
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="video",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.,
                 scale_by_std=False,
                 frame_num=None,
                 *args, **kwargs
                 ):
        self.frame_num = frame_num
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        # 实现DiffusionWrapper后的向后兼容性
        # cond_stage_config -> conditioning_key -> concat_mode

        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None
        else:
            if conditioning_key is not None:
                pass
            else:
                if concat_mode:
                    conditioning_key = 'concat'
                else:
                    conditioning_key = 'crossattn'
                
        
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        # self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.bbox_tokenizer = None  

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

        # torch.linspace(0,999,1)                                                                                       
        # tensor([0.])
        # self.cond_ids = [0, 999, 999, 999, ..., 999]
                      
    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = []
            for i in range(len(x)):
                encoder_posterior.append(self.encode_first_stage(x[i]))
            encoder_posterior = torch.stack(encoder_posterior)
            encoder_posterior = rearrange(encoder_posterior, 'b t c h w -> b (t c) h w')
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        # false
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        # self.first_stage_model = model.eval() [make VAE has only have eval mode]
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        # this way
        # 条件不可训练
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
            else:
                # TODO: -> this way, make a motion state 相关的 model from x
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def _get_denoise_row(self, samples, desc='', force_no_decoder_quantization=False):
        samples = samples.reshape(samples.shape[0], samples.shape[1], -1, self.channels, self.image_size, self.image_size)
        samples = rearrange(samples, 'diffusion_time b video_time c h w -> b diffusion_time video_time c h w')
        # [256, 7, 30, 3, 16, 16]
        n_row = samples.shape[2]
        
        denoise_grids = []
        for video in samples:
            video = rearrange(video, 'b t c h w -> (b t) c h w')
            denoise_grid = []
            for i in range(ceil(video.shape[0]/self.VQ_batch_size)):
                denoise_grid.append(self.decode_first_stage(video[i*self.VQ_batch_size:(i+1)*self.VQ_batch_size]))
            del video
            denoise_grid = torch.cat(denoise_grid)
            # [210, 3, 64, 64] -> [7, 30, 3, 64, 64]
            denoise_grid = denoise_grid.reshape(samples.shape[1], -1, denoise_grid.shape[1], denoise_grid.shape[2], denoise_grid.shape[3])

            denoise_grid = rearrange(denoise_grid, "diffusion_time video_time c h w -> (diffusion_time video_time) c h w")
            denoise_grid = make_grid(denoise_grid, nrow=n_row)
            denoise_grids.append(denoise_grid)
        denoise_grids = torch.stack(denoise_grids)
        return denoise_grids

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            # print("isinstance(encoder_posterior, DiagonalGaussianDistribution)")
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            # print("isinstance(encoder_posterior, torch.Tensor)")
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            # this way, cond条件模型应该设置一个encode接口
            # ps: c可能是一个字典或者列表，这个时候应该创建encode接口
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def meshgrid(self, h, w):
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch.cat([y, x], dim=-1)
        return arr

    def delta_border(self, h, w):
        """
        :param h: height
        :param w: width
        :return: normalized distance to image border,
         wtith min distance = 0 at border and max dist = 0.5 at image center
        """
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = self.meshgrid(h, w) / lower_right_corner
        dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
        dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
        edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]
        return edge_dist

    def get_weighting(self, h, w, Ly, Lx, device):
        weighting = self.delta_border(h, w)
        weighting = torch.clip(weighting, self.split_input_params["clip_min_weight"],
                               self.split_input_params["clip_max_weight"], )
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

        if self.split_input_params["tie_braker"]:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(L_weighting,
                                     self.split_input_params["clip_min_tie_weight"],
                                     self.split_input_params["clip_max_tie_weight"])

            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
            weighting = weighting * L_weighting
        return weighting

    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):  # todo load once not every time, shorten code
        """
        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """
        bs, nc, h, w = x.shape

        # number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(kernel_size[0], kernel_size[1], Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                                dilation=1, padding=0,
                                stride=(stride[0] * uf, stride[1] * uf))
            fold = torch.nn.Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h * uf, w * uf)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))

        elif df > 1 and uf == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                                dilation=1, padding=0,
                                stride=(stride[0] // df, stride[1] // df))
            fold = torch.nn.Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h // df, w // df)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))

        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting

    @torch.no_grad()
    def get_input(self, batch, key, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):
        # key = "video" to get video batch x
        # batch["video"]:
        #   [b, t, c, h_origin, w_origin]
        # TODO: 改成支持自回归格式的模型
        # x:
        #   [b, self.frame_num.pred, c, h_origin, w_origin]
        # z:
        #   [b, self.frame_num.pred*c, h_latent, w_latent]
        # c:
        #   [b, self.frame_num.cond*c, h_latent, w_latent]

        """
        x: original video batch        
        |
        |   index (b t c h w) -> (b' c h w)
        v
        x[i] 
        |
        |   Encoder (b' c h w) -> (b' c h' w')
        |   append and stack  (b' c h w) -> (b'' b' c h' w')
        v
        z
        |
        |   reshape (b'' b' c h' w') -> (b t c h' w')
        |   rearrange (b t c h' w') -> (b c' h' w')
        |   get_first_stage_encoding
        v
        z   (prepare for Conv2d)
        """

        x = super().get_input(batch, key)
        if bs is not None:
            x = x[:bs]

        # TODO: 为了获取cond frames的encode 和mcvd对齐 暂时注释
        # x = x[:,:self.frame_num.pred].to(self.device)
        if x.shape[2] == 1:
            x = repeat(x, "b t c h w -> b t (3 c) h w")

        # x torch.Size([64, 15, 1->3, 64, 64])

        # it is based on LDM image encode process
        # start = time.time()
        z = []
        tmp_z = rearrange(x, 'b t c h w -> (b t) c h w')
        for i in range(ceil(tmp_z.shape[0]/self.VQ_batch_size)):
            z.append(self.encode_first_stage(tmp_z[i*self.VQ_batch_size:(i+1)*self.VQ_batch_size]))
        del tmp_z
        z = torch.cat(z)
        z = z.reshape(x.shape[0], -1, self.channels, self.image_size, self.image_size)
        z = rearrange(z, 'b t c h w -> b (t c) h w')
        z = self.get_first_stage_encoding(z).detach()
        # rank_zero_info(f"get input encode done         {time.time() - start}")
        # about 0.18 seconds

        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key

            if cond_key != self.first_stage_key:
                if cond_key in ['caption', 'coordinates_bbox']:
                    xc = batch[cond_key]
                elif cond_key == 'class_label':
                    xc = batch
                # condition_frames_motion_state this way
                elif cond_key == "condition_frames_motion_state":
                    xc = x[:,:self.frame_num.cond]
                # condition_frames
                elif cond_key == "condition_frames":
                    xc = x[:,:self.frame_num.cond]
                else:
                    xc = super().get_input(batch, cond_key).to(self.device)
            else:
                xc = x

            #  (条件阶段不可训练) 或者 (强制对条件进行encode)
            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(xc, dict) or isinstance(xc, list):
                    c = self.get_learned_conditioning(xc)
                else:
                    # tensor this way
                    # TODO: 为了直接使用条件帧的办法，临时措施
                    if cond_key == "condition_frames":
                        c = z[:,:self.frame_num.cond*self.channels].to(self.device)
                    else:
                        c = self.get_learned_conditioning(xc.to(self.device))
            else:
                c = xc

            if bs is not None:
                c = c[:bs]

            # if self.use_positional_encodings:
            #     pos_x, pos_y = self.compute_latent_shifts(batch)
            #     ckey = __conditioning_keys__[self.model.conditioning_key]
            #     c = {ckey: c, 'pos_x': pos_x, 'pos_y': pos_y}

        else:
            c = None
            xc = None
            # if self.use_positional_encodings:
            #     pos_x, pos_y = self.compute_latent_shifts(batch)
            #     c = {'pos_x': pos_x, 'pos_y': pos_y}

        # z torch.Size([64, 5*3, 16, 16])
        z = z[:,self.frame_num.cond*self.channels:].to(self.device)
        # x torch.Size([64, 5, 3, 64, 64])
        x = x[:,self.frame_num.cond:].to(self.device)
        
        out = [z, c]

        if return_first_stage_outputs:
            """
            z
            |
            |   reshape  ('b t*c h w -> b t c h w') -> connot go by rearrange
            v
            tmp_z
            |
            |   index ('b t c h w -> t c h w')
            v
            tmp_z[i]
            |
            |   Decoder ('t c h w -> t c h*f w*f')
            |   append and stack  ('t c h w -> b t c h w')
            v
            x_rec (prepare for output)
            """
            # start = time.time()
            tmp_x_rec = z.reshape(z.shape[0], -1, self.channels, self.image_size, self.image_size)
            tmp_x_rec = rearrange(tmp_x_rec, 'b t c h w -> (b t) c h w')
            x_rec = []
            for i in range(ceil(tmp_x_rec.shape[0]/self.VQ_batch_size)):
                x_rec.append(self.decode_first_stage(tmp_x_rec[i*self.VQ_batch_size:(i+1)*self.VQ_batch_size]))
            del tmp_x_rec
            x_rec = torch.cat(x_rec)
            x_rec = x_rec.reshape(x.shape[0], -1, self.channels, x.shape[3], x.shape[4])
            # rank_zero_info(f"x_rec           min max mean {x_rec.min()} {x_rec.max()} {x_rec.mean()}")
            # rank_zero_info(f"get input decode done         {time.time() - start}")
            # about 0.18 seconds
            out.extend([x, x_rec])

        if return_original_cond:
            out.append(xc)
        
        # z: latent [b, self.frame_num.train_valid_total*channels, h_latent, h_latent]
        # x_rec：reconstruction of x (x -- Encode --> z -- Decode --> xrec)
        # xc: original_cond when cond_key (wait for research)
        # out: [z, c] + [x, x_rec] + [xc]
        # [z, None]
        return out 

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        # ignore this we have no this parameter
        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:

                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                result = self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                # result = ((result - result.min()) / (result.max() - result.min()))*2.-1.
                return result
            else:
                result = self.first_stage_model.decode(z)
                # result = ((result - result.min()) / (result.max() - result.min()))*2.-1.
                return result

    @torch.no_grad()
    def encode_first_stage(self, x):
        # ignore this we have no this parameter
        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                df = self.split_input_params["vqf"]
                self.split_input_params['original_image_size'] = x.shape[-2:]
                bs, nc, h, w = x.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(x, ks, stride, df=df)
                z = unfold(x)  # (bn, nc * prod(**ks), L)
                # Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                output_list = [self.first_stage_model.encode(z[:, :, :, :, i])
                               for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # Reverse reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded

            else:
                return self.first_stage_model.encode(x)
        else:
            return self.first_stage_model.encode(x)

    # LatentDiff rewrite DDPM shared_step
    def shared_step(self, batch, **kwargs):
        # batch[key]:
        #   [b, t_dataset, c, h_origin, w_origin]
        z, c = self.get_input(batch, self.first_stage_key)
        loss = self(z, c)
        return loss

    def forward(self, z, c, *args, **kwargs):
        # z: 
        #   latent 
        #   [b, self.frame_num.train_valid_total*self.frame_num.channels, h_latent, h_latent]

        t = torch.randint(0, self.num_timesteps, (z.shape[0],), device=self.device).long()
        
        # self.model.conditioning_key: None
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            # false
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        
        return self.p_losses(z, c, t, *args, **kwargs)

    def _rescale_annotations(self, bboxes, crop_coordinates):  # TODO: move to dataset
        def rescale_bbox(bbox):
            x0 = torch.clamp((bbox[0] - crop_coordinates[0]) / crop_coordinates[2])
            y0 = torch.clamp((bbox[1] - crop_coordinates[1]) / crop_coordinates[3])
            w = min(bbox[2] / crop_coordinates[2], 1 - x0)
            h = min(bbox[3] / crop_coordinates[3], 1 - y0)
            return x0, y0, w, h

        return [rescale_bbox(b) for b in bboxes]

    def apply_model(self, x_noisy, t, cond, return_ids=False):

        if isinstance(cond, dict):
            # TODO: hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'

            # {'c_crossattn': [None]}
            # {'c_crossattn': [tensor] }
            cond = {key: cond}

        # ignore this we have no this parameter
        if hasattr(self, "split_input_params"):
            assert len(cond) == 1  # todo can only deal with one conditioning atm
            assert not return_ids  
            ks = self.split_input_params["ks"]  # eg. (128, 128)
            stride = self.split_input_params["stride"]  # eg. (64, 64)

            h, w = x_noisy.shape[-2:]

            fold, unfold, normalization, weighting = self.get_fold_unfold(x_noisy, ks, stride)

            z = unfold(x_noisy)  # (bn, nc * prod(**ks), L)
            # Reshape to img shape
            z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
            z_list = [z[:, :, :, :, i] for i in range(z.shape[-1])]

            if self.cond_stage_key in ["video", "LR_image", "segmentation", 'bbox_img'] and self.model.conditioning_key:  # todo check for completeness
                c_key = next(iter(cond.keys()))  # get key
                c = next(iter(cond.values()))  # get value
                assert (len(c) == 1)  # todo extend to list with more than one elem
                c = c[0]  # get element

                c = unfold(c)
                c = c.view((c.shape[0], -1, ks[0], ks[1], c.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                cond_list = [{c_key: [c[:, :, :, :, i]]} for i in range(c.shape[-1])]

            elif self.cond_stage_key == 'coordinates_bbox':
                assert 'original_image_size' in self.split_input_params, 'BoudingBoxRescaling is missing original_image_size'

                # assuming padding of unfold is always 0 and its dilation is always 1
                n_patches_per_row = int((w - ks[0]) / stride[0] + 1)
                full_img_h, full_img_w = self.split_input_params['original_image_size']
                # as we are operating on latents, we need the factor from the original image size to the
                # spatial latent size to properly rescale the crops for regenerating the bbox annotations
                num_downs = self.first_stage_model.encoder.num_resolutions - 1
                rescale_latent = 2 ** (num_downs)

                # get top left postions of patches as conforming for the bbbox tokenizer, therefore we
                # need to rescale the tl patch coordinates to be in between (0,1)
                tl_patch_coordinates = [(rescale_latent * stride[0] * (patch_nr % n_patches_per_row) / full_img_w,
                                         rescale_latent * stride[1] * (patch_nr // n_patches_per_row) / full_img_h)
                                        for patch_nr in range(z.shape[-1])]

                # patch_limits are tl_coord, width and height coordinates as (x_tl, y_tl, h, w)
                patch_limits = [(x_tl, y_tl,
                                 rescale_latent * ks[0] / full_img_w,
                                 rescale_latent * ks[1] / full_img_h) for x_tl, y_tl in tl_patch_coordinates]
                # patch_values = [(np.arange(x_tl,min(x_tl+ks, 1.)),np.arange(y_tl,min(y_tl+ks, 1.))) for x_tl, y_tl in tl_patch_coordinates]

                # tokenize crop coordinates for the bounding boxes of the respective patches
                patch_limits_tknzd = [torch.LongTensor(self.bbox_tokenizer._crop_encoder(bbox))[None].to(self.device)
                                      for bbox in patch_limits]  # list of length l with tensors of shape (1, 2)
                print(patch_limits_tknzd[0].shape)
                # cut tknzd crop position from conditioning
                assert isinstance(cond, dict), 'cond must be dict to be fed into model'
                cut_cond = cond['c_crossattn'][0][..., :-2].to(self.device)
                print(cut_cond.shape)

                adapted_cond = torch.stack([torch.cat([cut_cond, p], dim=1) for p in patch_limits_tknzd])
                adapted_cond = rearrange(adapted_cond, 'l b n -> (l b) n')
                print(adapted_cond.shape)
                adapted_cond = self.get_learned_conditioning(adapted_cond)
                print(adapted_cond.shape)
                adapted_cond = rearrange(adapted_cond, '(l b) n d -> l b n d', l=z.shape[-1])
                print(adapted_cond.shape)

                cond_list = [{'c_crossattn': [e]} for e in adapted_cond]

            else:
                cond_list = [cond for i in range(z.shape[-1])]  # Todo make this more efficient

            # apply model by loop over crops
            output_list = [self.model(z_list[i], t, **cond_list[i]) for i in range(z.shape[-1])]
            assert not isinstance(output_list[0],
                                  tuple)  # todo cant deal with multiple model outputs check this never happens

            o = torch.stack(output_list, axis=-1)
            o = o * weighting
            # Reverse reshape to img shape
            o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
            # stitch crops together
            x_recon = fold(o) / normalization

        else:
            # into DiffusionWrapper then into DDPM unet model
            x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
               extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def p_losses(self, z_start, cond, t, noise=None):
        # z_start (z): 
        #   latent 
        #   [b, (total-cond)*c, h_latent, h_latent]
        # cond: 
        #   condition
        #   [b, cond*c, h_latent, h_latent]
        # t: 
        #   difussion forward process (imnoise process)
        #   value randint[0, 1000step] shape [b,]

        # TODO: only support frames_per_sample: first 10 frames
        z_result    = cond # dont set zerolike tensor
        z_noise     = torch.zeros_like(cond) # only for padding
        eps_result  = torch.zeros_like(cond) # only for padding
        
        autogression_num = ceil( z_start.shape[1] / self.channels / self.frame_num.pred)

        for i in range(autogression_num):
            start_idx   = self.frame_num.pred*i*self.channels
            end_idx     = self.frame_num.pred*(i+1)*self.channels
            # TODO: think about how to solve the z_pred_init [(49-10)/5] = 8, we input need padding for it
            z_cond      = z_result[:, -self.frame_num.cond*self.channels:].clamp(-1.,1.)
            z_pred      = z_start[:, start_idx:end_idx]
            tmp_noise   = default(noise, lambda: torch.randn_like(z_pred)).to(self.device)
            z_noisy     = self.q_sample(x_start=z_pred, t=t, noise=tmp_noise)
            # rank_zero_info(f"z_noisy         min max mean {z_noisy.min()} {z_noisy.max()} {z_noisy.mean()}")
            # start_time = time.time()
            tmp_eps     = self.apply_model(z_noisy, t, z_cond)
            # rank_zero_info(f"apply_model done              {time.time() - start_time}")
            tmp_z       = self.predict_start_from_noise(z_noisy, t=t, noise=tmp_eps)
            # rank_zero_info(f"tmp_eps  min max mean {tmp_eps.min()} {tmp_eps.max()} {tmp_eps.mean()}")
            z_noise     = torch.cat([z_noise,    tmp_noise] , dim=1)
            eps_result  = torch.cat([eps_result, tmp_eps]   , dim=1)
            z_result    = torch.cat([z_result,   tmp_z]     , dim=1)

        # rank_zero_info(f"tmp_eps_output  min max mean {eps_result.min()} {eps_result.max()} {eps_result.mean()}")
        
        z_noise     = z_noise   [:, self.frame_num.cond*self.channels : self.frame_num.cond*self.channels+z_start.shape[1]]
        eps_result  = eps_result[:, self.frame_num.cond*self.channels : self.frame_num.cond*self.channels+z_start.shape[1]]
        
        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        # rank_zero_info("get loss")
        # loss_latent (eps) ------------------------------------------------
        loss_eps = self.get_loss(eps_result, z_noise, mean=False).sum([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_eps': loss_eps.mean()})

        # variational lower bound loss --------------------------------------
        loss_vlb = (self.lvlb_weights[t] * loss_eps).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})

        # if self.learn_logvar:
        #     logvar_t = self.logvar[t].to(self.device)
        #     loss_gamma = loss_simple_latent / torch.exp(logvar_t) + logvar_t
        #     # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        #     loss_dict.update({f'{prefix}/loss_gamma': loss_gamma.mean()})
        #     loss_dict.update({'logvar': self.logvar.data.mean()})
        #     loss = self.l_simple_weight * loss_gamma.mean() + self.original_elbo_weight * loss_vlb

        # loss_sum -------------------------------------------------------
        loss = self.l_eps_weight * loss_eps.mean() + self.original_elbo_weight * loss_vlb
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

        

    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,return_x0=False, score_corrector=None, corrector_kwargs=None):
        model_out = self.apply_model(x, t, c, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()
        if clip_denoised:
            x_recon = x_recon.clamp(-1., 1.)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        b, *_, device = *x.shape, x.device

        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, torch.logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, quantize_denoised=False, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            video = torch.randn(shape, device=self.device)
        else:
            video = x_T

        intermediates = []

        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)

        if verbose:
            # rank_zero_info(f"Running progressive_denoising with {timesteps} timesteps")
            iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation', total=timesteps)  
        else:
            iterator = reversed(range(0, timesteps))

        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            # false
            # if self.shorten_cond_schedule:
            #     assert self.model.conditioning_key != 'hybrid'
            #     tc = self.cond_ids[ts].to(cond.device)
            #     cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                video = img_orig * mask + (1. - mask) * video
            
            video, x0_partial = self.p_sample(video, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            
            if i % log_every_t == 0 or i == timesteps - 1:
                rank_zero_info(f"progressive_denoising (DDPM) {i} {log_every_t}")
                # rank_zero_info(f"-------- video-- -- min max mean {video.min()} {video.max()} {video.mean()}")
                intermediates.append(x0_partial)
        return video, intermediates

    @torch.no_grad()
    def p_sample_loop(self, cond, batch_size, shape, return_intermediates=False,
                      x_T=None, verbose=True, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, start_T=None,
                      log_every_t=None):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device

        b = batch_size

        if x_T is None:
            video = torch.randn(shape, device=device)
        else:
            video = x_T

        intermediates = [video]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)

        if verbose:
            iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps)
        else:
            iterator = reversed(range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            
            # false
            # if self.shorten_cond_schedule:
            #     assert self.model.conditioning_key != 'hybrid'
            #     tc = self.cond_ids[ts].to(cond.device)
            #     cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            video = self.p_sample(video, cond, ts, clip_denoised=self.clip_denoised, quantize_denoised=quantize_denoised)
            
            if mask is not None:
                video_orig = self.q_sample(x0, ts)
                video = video_orig * mask + (1. - mask) * video

            if i % log_every_t == 0 or i == timesteps - 1:
                rank_zero_info(f"sample_log p_sample (DDPM) {i} {log_every_t}")
                # rank_zero_info(f"---------- video -- min max mean {video.min()} {video.max()} {video.mean()}")
                intermediates.append(video)
        if return_intermediates:
            return video, intermediates
        return video

    @torch.no_grad()
    def sample(self, cond, shape, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None):
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  batch_size,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0)

    @torch.no_grad()
    def sample_log(self, cond, ddim, ddim_steps, x_T, shape, **kwargs):
        # rank_zero_info("sample_log autogression")
        # x_T [bs, total*ch, h_latent, w_latent]

        batch_size = x_T.shape[0]

        z_cond = cond
        z_pred = x_T
        
        z_intermediates_real = z_pred.unsqueeze(0)
        z_result = z_cond

        z_intermediates = []

        autogression_num = ceil( z_pred.shape[1] / self.channels / self.frame_num.pred)

        for i in range(autogression_num):
            z_cond      = z_result[:, -self.frame_num.cond*self.channels:]
            z_noisy     = torch.randn(z_pred.shape[0], self.frame_num.pred*self.channels, self.image_size, self.image_size).to(self.device)
            if ddim:
                ddim_sampler = DDIMSampler(self)
                tmp_samples, tmp_intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, conditioning=z_cond, verbose=False, x_T=z_noisy, log_every_t=50, clip_denoised=self.clip_denoised, **kwargs)
                z_result = torch.cat([z_result, tmp_samples], dim=1)
                z_intermediates.append(torch.stack(tmp_intermediates['x_inter'])) # 'pred_x0' is DDIM's predicted x0
            else:
                tmp_samples, tmp_intermediates = self.sample(z_cond, shape, batch_size=batch_size, return_intermediates=True, x_T=z_noisy, verbose=False)
                z_result = torch.cat([z_result, tmp_samples], dim=1)
                z_intermediates.append(torch.stack(tmp_intermediates))
        
        z_intermediates = torch.cat(z_intermediates, dim=2)
        z_intermediates = z_intermediates[:,:,:z_intermediates_real.shape[2]]
        z_intermediates = torch.cat([z_intermediates_real, z_intermediates], dim=0)

        samples         = z_result
        intermediates   = z_intermediates

        # rank_zero_info("sample_log autogression done")
        # print("samples", samples.shape)
        # print("intermediates", intermediates.shape)
        # samples torch.Size([8, 150, 16, 16])
        # intermediates torch.Size([4, 8, 150, 16, 16])

        return samples, intermediates
    
    @torch.no_grad()
    def progressive_denoising_log(self, cond, batch_size, x_T, shape):
        # x_T [bs, total*ch, h_latent, w_latent]
        z_start = x_T

        z_cond = z_start[:, :self.frame_num.cond*self.channels]
        z_pred = z_start[:, self.frame_num.cond*self.channels:]
        
        z_intermediates_real = z_pred.unsqueeze(0)
        z_result = z_cond

        z_intermediates = []

        autogression_num = ceil( (z_start.shape[1]/self.channels - self.frame_num.cond) / self.frame_num.pred)

        # FIXME: should fix 现在没用到这个
        for i in range(autogression_num):
            z_cond      = z_result[:, -self.frame_num.cond*self.channels:]
            z_noisy     = torch.randn(z_start.shape[0], self.frame_num.pred*self.channels, self.image_size, self.image_size).to(self.device)
            x_input     = torch.cat([z_cond, z_noisy], dim=1)
            tmp_samples, tmp_intermediates = self.progressive_denoising(x_T=x_input, cond=cond, shape=shape, batch_size=batch_size, verbose=False)
            z_result = torch.cat([z_result, tmp_samples[:,-self.frame_num.pred*self.channels:]], dim=1)
            z_intermediates.append(torch.stack(tmp_intermediates)[:,:,-self.frame_num.pred*self.channels:])
        
        z_intermediates = torch.cat(z_intermediates, dim=2)
        z_intermediates = z_intermediates[:,:,:z_intermediates_real.shape[2]]
        z_intermediates = torch.cat([z_intermediates_real, z_intermediates], dim=0)
 
        samples         = z_result
        intermediates   = z_intermediates

        # rank_zero_info("progressive_denoising_log done")
        # print("samples", samples.shape)
        # print("intermediates", intermediates.shape)
        # # samples torch.Size([8, 150, 16, 16])
        # # intermediates torch.Size([4, 8, 150, 16, 16])

        return samples, intermediates


    @torch.no_grad()
    def log_videos(self, batch, split, save_video_num, calculate_video_num, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,use_ddim = True,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=True, plot_progressive_rows=True,
                   plot_diffusion_rows=True, **kwargs):
        
        log_metrics = dict()
        log = dict()
        z, c, x, x_rec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=calculate_video_num)
        
        # TODO: a little problem
        shape = (self.frame_num.pred*self.channels, self.image_size, self.image_size)

        log["input"] = x[:save_video_num]
        # rank_zero_info(f"-------------------- x -- min max mean {x.min()} {x.max()} {x.mean()}")
        log["recon"] = x_rec[:save_video_num]
        # rank_zero_info(f"---------------- x_rec -- min max mean {x_rec.min()} {x_rec.max()} {x_rec.mean()}")

        # print(f"z    {z.shape}")
        # print(f"c    {c.shape}")
        # print(f"x    {x.shape}")
        # print(f"xrec {x_rec.shape}")
        # print(f"xc   {xc.shape}")

        """
        z    torch.Size([64, 60, 16, 16])
        c    torch.Size([64, 30, 16, 16])
        x    torch.Size([64, 20, 3, 64, 64])
        xrec torch.Size([64, 20, 3, 64, 64])
        xc   torch.Size([64, 10, 3, 64, 64])
        """

        # TODO: do it
        if self.model.conditioning_key is not None:
            log["condition"] = xc[:save_video_num]
            # if hasattr(self.cond_stage_model, "decode"):
            #     xc = self.cond_stage_model.decode(c)
            #     log["conditioning"] = xc
            # elif self.cond_stage_key in ["caption"]:
            #     xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["caption"])
            #     log["conditioning"] = xc
            # elif self.cond_stage_key == 'class_label':
            #     xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"])
            #     log['conditioning'] = xc
            # elif isimage(xc):
            #     log["conditioning"] = xc
            # if ismap(xc):
            #     log["original_conditioning"] = self.to_rgb(xc)

        # plot diffusion add noise process
        # if plot_diffusion_rows:
        #     log["diffusion_row"] = []
        #     for video_idx in range(save_video_num):
        #         # a latent video: [60[:10], 3, 16, 16]
        #         latent_video = z.reshape(z.shape[0], -1, 3, z.shape[-2], z.shape[-1])[video_idx]
        #         # print(latent_video.shape)
        #         n_row = latent_video.shape[0]
        #         # get diffusion row
        #         diffusion_grid = []
        #         for t in range(self.num_timesteps):
        #             if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
        #                 t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
        #                 t = t.to(self.device).long()
        #                 noise = torch.randn_like(latent_video)
        #                 z_noisy = self.q_sample(x_start=latent_video, t=t, noise=noise)
        #                 diffusion_grid.append(self.decode_first_stage(z_noisy))

        #         # n_log_step, n_row, C, H, W
        #         diffusion_grid = torch.stack(diffusion_grid)
        #         diffusion_grid = rearrange(diffusion_grid, 'n b c h w -> (n b) c h w')
        #         diffusion_grid = make_grid(diffusion_grid, nrow=n_row)
        #         log["diffusion_row"].append(diffusion_grid)

        #     log["diffusion_row"] = torch.stack(log["diffusion_row"])
            # print('log["diffusion_row"]', log["diffusion_row"].shape)

        # get denoise row
        if sample:
            # rank_zero_info(f"-------------------- z -- min max mean {z.min()} {z.max()} {z.mean()}")
            start_time = time.time()
            with self.ema_scope("Plotting denoise"):
                tmp_samples, tmp_z_denoise_row = self.sample_log(x_T=z, cond=c, ddim=use_ddim, ddim_steps=ddim_steps, eta=ddim_eta, shape=shape)
            tmp_samples = tmp_samples.reshape(z.shape[0], -1, 3, z.shape[-2], z.shape[-1])

            rank_zero_info(f"---------- tmp_samples -- min max mean {tmp_samples.min()} {tmp_samples.max()} {tmp_samples.mean()}")
            log["z_sample"] = tmp_samples[:save_video_num]
            log["z_origin"] = z.reshape(z.shape[0], -1, 3, z.shape[-2], z.shape[-1])[:save_video_num]

            log_metrics[f'{split}/z0_min'] = tmp_samples.min() 
            log_metrics[f'{split}/z0_max'] = tmp_samples.max() 

            tmp_samples = rearrange(tmp_samples, 'b t c h w -> (b t) c h w')[:]
            x_samples = []
            for i in range(ceil(tmp_samples.shape[0]/self.VQ_batch_size)):
                x_samples.append(self.decode_first_stage(tmp_samples[i*self.VQ_batch_size:(i+1)*self.VQ_batch_size]))
            x_samples = torch.cat(x_samples)
            x_samples = x_samples.reshape(x.shape[0], -1, self.channels, x.shape[3], x.shape[4])

            log_metrics[f'{split}/x0_min'] = x_samples.min() 
            log_metrics[f'{split}/x0_max'] = x_samples.max() 

            rank_zero_info(f"------- ddim x_samples -- min max mean {x_samples.min()} {x_samples.max()} {x_samples.mean()}")
            log["ddim200"] = x_samples[:save_video_num]

            # x_samples 包括条件帧
            autogression_num = ceil( (x_samples.shape[1] - self.frame_num.cond) / self.frame_num.pred)
            log_metrics[f"{split}/used_time_per_autogression"] = (time.time() - start_time) / autogression_num

            videos1 = x_samples[:,self.frame_num.cond:].clamp(-1.,1.)
            videos2 = x.clamp(-1.,1.)
            videos1 = (videos1+1.)/2.
            videos2 = (videos2+1.)/2.

            log_metrics[f'{split}/pixel mse'] = self.get_loss(videos1, videos2, mean=False).sum([1, 2, 3, 4]).mean()

            if videos1.shape[1] >= 10: 
                fvd = calculate_fvd1(videos1, videos2, self.device)
                log_metrics[f'{split}/fvd']  = fvd

            ssim = calculate_ssim1(videos1, videos2)
            psnr = calculate_psnr1(videos1, videos2)
            lpips = calculate_lpips1(videos1, videos2, self.device)

            log_metrics[f'{split}/avg/ssim']  = ssim[0]
            log_metrics[f'{split}/std/ssim']  = ssim[1]
            log_metrics[f'{split}/avg/psnr']  = psnr[0]
            log_metrics[f'{split}/std/psnr']  = psnr[1]
            log_metrics[f'{split}/avg/lpips'] = lpips[0]
            log_metrics[f'{split}/std/lpips'] = lpips[1]
            
            if plot_denoise_rows:
                denoise_grids = self._get_denoise_row(tmp_z_denoise_row[:,:save_video_num])
                log["ddim200_row"] = denoise_grids

            # if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(self.first_stage_model, IdentityFirstStage):
            #     # also display when quantizing x0 while sampling
            #     with self.ema_scope("Plotting Quantized Denoised"):
            #         samples, _ = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,ddim_steps=ddim_steps,eta=ddim_eta,quantize_denoised=True, x_T=z, shape=shape)
            #     samples = samples.reshape(N, -1, 3, z.shape[-2], z.shape[-1])

            #     x_samples = []
            #     for i in range(N):
            #         x_samples.append(self.decode_first_stage(samples[i]))
            #     x_samples = torch.stack(x_samples) 
            #     # rank_zero_info(f"----- q ddim x_samples -- min max mean {x_samples.min()} {x_samples.max()} {x_samples.mean()}")
            #     log["ddim200_quantized"] = x_samples

        # if plot_progressive_rows:
        #     with self.ema_scope("Plotting Progressives ddpm"):
        #         tmp_samples, progressives = self.progressive_denoising_log(cond=c, batch_size=N, x_T=z, shape=shape)
        #     tmp_samples = tmp_samples.reshape(N, -1, 3, z.shape[-2], z.shape[-1])
        #     prog_row = self._get_denoise_row(progressives, desc="Progressive Generation")
        #     log["ddpm1000_row"] = prog_row

        #     x_samples = []
        #     for i in range(N):
        #         x_samples.append(self.decode_first_stage(tmp_samples[i]))
        #     x_samples = torch.stack(x_samples) 
        #     log["ddpm1000"] = x_samples

            # rank_zero_info(f"------- ddpm x_samples -- min max mean {x_samples.min()} {x_samples.max()} {x_samples.mean()}")
            

        # return_keys = None
        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
            
        return log, log_metrics

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            default_scheduler_cfg = {
                "target": "ldm.lr_scheduler.LambdaLinearScheduler",
                "params": {
                    "warm_up_steps": [10000],
                    "cycle_lengths": [10000000000000],
                    "f_start": [1.e-6],
                    "f_max": [1.],
                    "f_min": [1.],
                    "verbosity_interval": -1,
                }
            }
            assert 'target' in self.scheduler_config
            scheduler_cfg = OmegaConf.merge(default_scheduler_cfg, self.scheduler_config)
            # instantiate 
            scheduler = instantiate_from_config(scheduler_cfg)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x

# DiffusionWrapper(unet_config, conditioning_key(crossattn))
class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        # -> this way, 把crossattn的条件全加，通道数增加，所以如果有多个条件，所有条件的维度和形状要相同
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out


# class Layout2ImgDiffusion(LatentDiffusion):
#     # TODO: move all layout-specific hacks to this class
#     def __init__(self, cond_stage_key, *args, **kwargs):
#         assert cond_stage_key == 'coordinates_bbox', 'Layout2ImgDiffusion only for cond_stage_key="coordinates_bbox"'
#         super().__init__(cond_stage_key=cond_stage_key, *args, **kwargs)

#     def log_videos(self, batch, N=8, *args, **kwargs):
#         logs = super().log_videos(batch=batch, N=N, *args, **kwargs)

#         key = 'train' if self.training else 'validation'
#         dset = self.trainer.datamodule.datasets[key]
#         mapper = dset.conditional_builders[self.cond_stage_key]

#         bbox_imgs = []
#         map_fn = lambda catno: dset.get_textual_label(dset.get_category_id(catno))
#         for tknzd_bbox in batch[self.cond_stage_key][:N]:
#             bboximg = mapper.plot(tknzd_bbox.detach().cpu(), map_fn, (256, 256))
#             bbox_imgs.append(bboximg)

#         cond_img = torch.stack(bbox_imgs, dim=0)
#         logs['bbox_image'] = cond_img
#         return logs
