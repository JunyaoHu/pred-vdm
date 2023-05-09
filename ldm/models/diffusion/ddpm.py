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
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

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
                 monitor="val/loss_ema",
                 use_ema=True,
                 ema_rate=0.999,
                 first_stage_key="video",
                 image_size=256,
                 channels=3,
                 VQ_batch_size=128,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 original_elbo_weight=0.0,
                 l_eps_weight=1.0,
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
        self.ema_rate = ema_rate
        if self.use_ema:
            self.model_ema = LitEma(self.model, decay=self.ema_rate)
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
        # ### it has been rewritten in LatentDiffusion ###
        pass

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=False, repeat_noise=False):
        # ### it has been rewritten in LatentDiffusion ###
        pass

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        # ### it has been rewritten in LatentDiffusion ###
        pass

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
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

    def p_losses(self, x_start, cond, t, noise=None):
        # ### it has been rewritten in LatentDiffusion ###
        pass

    def forward(self, x, *args, **kwargs):
        # ### it has been rewritten in LatentDiffusion ###
        pass
    
    def get_input(self, batch, key):
        x = batch[key]
        x = x.to(memory_format=torch.contiguous_format).float()
        # garyscale to rgb channels
        if self.origin_channels == 1:
            x = repeat(x, 'b t c h w -> b t (n c) h w', n = 3)
        return x

    def shared_step(self, batch):
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
        # ### it has been rewritten in LatentDiffusion ###
        pass

    def configure_optimizers(self):
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
                 origin_channels=1,
                 *args, **kwargs
                 ):
        self.frame_num = frame_num
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        self.origin_channels = origin_channels
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
        self.clip_denoised = False
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
            z = self.video_batch_encode(x)
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
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
            else:
                # TODO: -> this way, make a motion state 且不训练的 model from x
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
    def video_batch_encode(self, x):
        # x:
        #   [b, t, c=1 -> 3, H, H]
        # z:
        #   [b, t, c=3, h, w]
        bs = x.shape[0]
        z = []
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        for i in range(ceil(x.shape[0]/self.VQ_batch_size)):
            z.append(self.encode_first_stage(x[i*self.VQ_batch_size:(i+1)*self.VQ_batch_size]))
        z = torch.cat(z)
        # channels must be self.channels = 3
        z = z.reshape(bs, -1, self.channels, self.image_size, self.image_size)
        z = self.get_first_stage_encoding(z).detach()
        return z

    @torch.no_grad()
    def video_batch_decode(self, z):
        # z:
        #   [b, t, c=3, h, w]
        # x_rec:
        #   [b, t, c=1 or 3, H, H]
        bs = z.shape[0]
        z = rearrange(z, 'b t c h w -> (b t) c h w')
        x_rec = []
        for i in range(ceil(z.shape[0]/self.VQ_batch_size)):
            x_rec.append(self.decode_first_stage(z[i*self.VQ_batch_size:(i+1)*self.VQ_batch_size]))
        x_rec = torch.cat(x_rec)
        x_rec = x_rec.reshape(bs, -1, self.channels, x_rec.shape[2], x_rec.shape[3])
        if self.origin_channels == 1:
            x_rec.mean(dim=2).unsqueeze(2)
        return x_rec

    @torch.no_grad()
    def get_input(self, batch, key, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):
        #   KTH:            c=1, HW=64,  cond=10, pred=20 or 40
        #   BAIR:           c=3, HW=64,  cond=2,  pred=28
        #   Cityscapes:     c=3, HW=128, cond=2,  pred=28
        #   latent factor:  f=4, means (HW=64 -> hw=16) or (HW=128 -> hw=132)
        #   LDM encoder and decoder only support c=3, so if video is grayscale, it must repeat to 3 channels.
        # 
        # let key = "video" to get video batch tesnor
        # 
        # batch["video"]: 
        #   total video frames
        #   [b, t(cond+pred), c(1 -> 3), H(64 or 128), W(64 or 128)]
        # x:
        #   predicted frames tensor, ground truch, model will predict it finally, it will be encoded by LDM encoder to z
        #   [b, self.frame_num.pred, c(1 -> 3), H, W]
        # z:
        #   predicted latent tensor, input tensor, model will predict it directedly, it comes from x encoded by LDM encoder
        #   [b, self.frame_num.pred, c(only 3), h, w]
        # c:
        #   conditional tensor, model will use it to predict, it will be encoded by condition encoder
        #   [shape depends on contional type]
        # out:
        #  [z, c] or [z, c, x, x_rec, xc]
        # 
        # TODO: 改成支持自回归格式的模型

        total_video = super().get_input(batch, key)

        if bs is not None:
            total_video = total_video[:bs]

        x = total_video[:, self.frame_num.cond:].to(self.device)

        # rank_zero_info(f"get input encode")
        # start = time.time()
        z = self.video_batch_encode(x)
        # rank_zero_info(f"get input encode done         {time.time() - start}")  # about 0.18 seconds

        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key

            if cond_key != self.first_stage_key:
                # TODO: how to input other condition, we only consider the type of "condition_frames_*" condition now
                if cond_key in [
                    'class_label',
                ]:
                    xc = batch
                elif cond_key in [
                    'caption', 
                    'coordinates_bbox',
                ]:
                    xc = batch[cond_key]
                elif cond_key in [
                    "condition_frames",                         # origin video
                    "condition_frames_latent",                  # latent video
                    "condition_frames_optical_flow",            # origin video's optical flow
                    "condition_frames_motion_state",            # origin video's motion state (optical flow, trajectory and others)
                    "condition_frames_latent_and_optical_flow", # latent video + origin video's optical flow
                ]:
                    xc = total_video[:,:self.frame_num.cond]
                else:
                    xc = super().get_input(batch, cond_key).to(self.device)
            else:
                xc = x

            #  (条件不可训练) 或者 (强制对条件进行encode) -> 直接获取条件进行encode之后的编码
            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(xc, dict) or isinstance(xc, list):
                    # force encode
                    c = self.get_learned_conditioning(xc)
                else:
                    if cond_key == "condition_frames_latent":
                        # untrainable
                        c = self.video_batch_encode(xc)
                    elif cond_key == "condition_frames_optical_flow":
                        # untrainable
                        from models.optical_flow.optical_flow import optical_flow
                        c = optical_flow(xc).to(self.device)
                    elif cond_key == "condition_frames_motion_state":
                        # untrainable
                        from models.motion_state.motion_state import motion_state
                        c = motion_state(xc).to(self.device)
                    
                    # force encode
                    else:
                        if self.cond_stage_key == "condition_frames_latent_and_optical_flow":
                            c1 = self.video_batch_encode(xc) # no grad
                            c2 = self.get_learned_conditioning(xc)
                            c = [c1, c2]
                        else:
                            c = self.get_learned_conditioning(xc)
            else:
                # original contidional tensor, like "condition_frames" get condition tensor when forward process
                c = xc

            if bs is not None:
                c = c[:bs]

            # TODO: whats this?
            # if self.use_positional_encodings:
            #     pos_x, pos_y = self.compute_latent_shifts(batch)
            #     ckey = __conditioning_keys__[self.model.conditioning_key]
            #     c = {ckey: c, 'pos_x': pos_x, 'pos_y': pos_y}

        else:
            c = None
            xc = None

            # TODO: whats this?
            # if self.use_positional_encodings:
            #     pos_x, pos_y = self.compute_latent_shifts(batch)
            #     c = {'pos_x': pos_x, 'pos_y': pos_y}

        out = [z, c]

        if return_first_stage_outputs:
            # start = time.time()
            # rank_zero_info(f"get input decode")
            x_rec = self.video_batch_decode(z)
            # rank_zero_info(f"x_rec           min max mean {x_rec.min()} {x_rec.max()} {x_rec.mean()}")
            # rank_zero_info(f"get input decode done         {time.time() - start}") # about 0.18 seconds
            out.extend([x, x_rec])

        if return_original_cond:
            out.append(xc)
        
        return out 

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        # now we ignore this we have no this parameter
        if hasattr(self, "split_input_params"):
            # if self.split_input_params["patch_distributed_vq"]:
            #     ks = self.split_input_params["ks"]  # eg. (128, 128)
            #     stride = self.split_input_params["stride"]  # eg. (64, 64)
            #     uf = self.split_input_params["vqf"]
            #     bs, nc, h, w = z.shape
            #     if ks[0] > h or ks[1] > w:
            #         ks = (min(ks[0], h), min(ks[1], w))
            #         print("reducing Kernel")

            #     if stride[0] > h or stride[1] > w:
            #         stride = (min(stride[0], h), min(stride[1], w))
            #         print("reducing stride")

            #     fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

            #     z = unfold(z)  # (bn, nc * prod(**ks), L)
            #     # 1. Reshape to img shape
            #     z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

            #     # 2. apply model loop over last dim
            #     if isinstance(self.first_stage_model, VQModelInterface):
            #         output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
            #                                                      force_not_quantize=predict_cids or force_not_quantize)
            #                        for i in range(z.shape[-1])]
            #     else:

            #         output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
            #                        for i in range(z.shape[-1])]

            #     o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
            #     o = o * weighting
            #     # Reverse 1. reshape to img shape
            #     o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
            #     # stitch crops together
            #     decoded = fold(o)
            #     decoded = decoded / normalization  # norm is shape (1, 1, h, w)
            #     return decoded
            # else:
            #     if isinstance(self.first_stage_model, VQModelInterface):
            #         return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            #     else:
            #         return self.first_stage_model.decode(z)
            NotImplementedError()
        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                result = self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                return result
            else:
                result = self.first_stage_model.decode(z)
                return result

    @torch.no_grad()
    def encode_first_stage(self, x):
        # now we ignore this we have no this parameter
        if hasattr(self, "split_input_params"):
            # if self.split_input_params["patch_distributed_vq"]:
            #     ks = self.split_input_params["ks"]  # eg. (128, 128)
            #     stride = self.split_input_params["stride"]  # eg. (64, 64)
            #     df = self.split_input_params["vqf"]
            #     self.split_input_params['original_image_size'] = x.shape[-2:]
            #     bs, nc, h, w = x.shape
            #     if ks[0] > h or ks[1] > w:
            #         ks = (min(ks[0], h), min(ks[1], w))
            #         print("reducing Kernel")

            #     if stride[0] > h or stride[1] > w:
            #         stride = (min(stride[0], h), min(stride[1], w))
            #         print("reducing stride")

            #     fold, unfold, normalization, weighting = self.get_fold_unfold(x, ks, stride, df=df)
            #     z = unfold(x)  # (bn, nc * prod(**ks), L)
            #     # Reshape to img shape
            #     z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

            #     output_list = [self.first_stage_model.encode(z[:, :, :, :, i])
            #                    for i in range(z.shape[-1])]

            #     o = torch.stack(output_list, axis=-1)
            #     o = o * weighting

            #     # Reverse reshape to img shape
            #     o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
            #     # stitch crops together
            #     decoded = fold(o)
            #     decoded = decoded / normalization
            #     return decoded
            # else:
            #     return self.first_stage_model.encode(x)
            NotImplementedError()
        else:
            return self.first_stage_model.encode(x)

    # LatentDiff rewrite DDPM shared_step, this step will used in training and validation step
    def shared_step(self, batch, **kwargs):
        # get input tensor and conditional tensor
        z, c = self.get_input(batch, self.first_stage_key)
        # forward, use conditional tensor predict input tensor, and get loss
        return self(z, c)

    def forward(self, z, c, *args, **kwargs):
        # z: 
        #   latent input tensor
        #   [b, t=pred, 3, h, w]
        # c: 
        #   conditional tensor
        #   [unsure shape]

        t = torch.randint(0, self.num_timesteps, (z.shape[0],), device=self.device).long()
        
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                if self.cond_stage_key == "condition_frames_latent_and_optical_flow":
                    c1 = self.video_batch_encode(c) # no grad
                    # print("video_batch_encode", c1.shape)
                    c2 = self.get_learned_conditioning(c)
                    # print("optical_flow_token", c2.shape)
                    c = [c1, c2]
                else:
                    c = self.get_learned_conditioning(c)
            # TODO: drop this option
            # if self.shorten_cond_schedule:  
            #     tc = self.cond_ids[t].to(self.device)
            #     c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        
        return self.p_losses(z, c, t, *args, **kwargs)

    # def _rescale_annotations(self, bboxes, crop_coordinates):  # TODO: move to dataset
    #     def rescale_bbox(bbox):
    #         x0 = torch.clamp((bbox[0] - crop_coordinates[0]) / crop_coordinates[2])
    #         y0 = torch.clamp((bbox[1] - crop_coordinates[1]) / crop_coordinates[3])
    #         w = min(bbox[2] / crop_coordinates[2], 1 - x0)
    #         h = min(bbox[3] / crop_coordinates[3], 1 - y0)
    #         return x0, y0, w, h

    #     return [rescale_bbox(b) for b in bboxes]

    def apply_model(self, x_noisy, t, cond, return_ids=False):

        if isinstance(cond, dict):
            # TODO: hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        # ignore this we have no this parameter
        if hasattr(self, "split_input_params"):
            NotImplementedError()
            # assert len(cond) == 1  # todo can only deal with one conditioning atm
            # assert not return_ids  
            # ks = self.split_input_params["ks"]  # eg. (128, 128)
            # stride = self.split_input_params["stride"]  # eg. (64, 64)

            # h, w = x_noisy.shape[-2:]

            # fold, unfold, normalization, weighting = self.get_fold_unfold(x_noisy, ks, stride)

            # z = unfold(x_noisy)  # (bn, nc * prod(**ks), L)
            # # Reshape to img shape
            # z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
            # z_list = [z[:, :, :, :, i] for i in range(z.shape[-1])]

            # if self.cond_stage_key in ["video", "LR_image", "segmentation", 'bbox_img'] and self.model.conditioning_key:  # todo check for completeness
            #     c_key = next(iter(cond.keys()))  # get key
            #     c = next(iter(cond.values()))  # get value
            #     assert (len(c) == 1)  # todo extend to list with more than one elem
            #     c = c[0]  # get element

            #     c = unfold(c)
            #     c = c.view((c.shape[0], -1, ks[0], ks[1], c.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

            #     cond_list = [{c_key: [c[:, :, :, :, i]]} for i in range(c.shape[-1])]

            # elif self.cond_stage_key == 'coordinates_bbox':
            #     assert 'original_image_size' in self.split_input_params, 'BoudingBoxRescaling is missing original_image_size'

            #     # assuming padding of unfold is always 0 and its dilation is always 1
            #     n_patches_per_row = int((w - ks[0]) / stride[0] + 1)
            #     full_img_h, full_img_w = self.split_input_params['original_image_size']
            #     # as we are operating on latents, we need the factor from the original image size to the
            #     # spatial latent size to properly rescale the crops for regenerating the bbox annotations
            #     num_downs = self.first_stage_model.encoder.num_resolutions - 1
            #     rescale_latent = 2 ** (num_downs)

            #     # get top left postions of patches as conforming for the bbbox tokenizer, therefore we
            #     # need to rescale the tl patch coordinates to be in between (0,1)
            #     tl_patch_coordinates = [(rescale_latent * stride[0] * (patch_nr % n_patches_per_row) / full_img_w,
            #                              rescale_latent * stride[1] * (patch_nr // n_patches_per_row) / full_img_h)
            #                             for patch_nr in range(z.shape[-1])]

            #     # patch_limits are tl_coord, width and height coordinates as (x_tl, y_tl, h, w)
            #     patch_limits = [(x_tl, y_tl,
            #                      rescale_latent * ks[0] / full_img_w,
            #                      rescale_latent * ks[1] / full_img_h) for x_tl, y_tl in tl_patch_coordinates]
            #     # patch_values = [(np.arange(x_tl,min(x_tl+ks, 1.)),np.arange(y_tl,min(y_tl+ks, 1.))) for x_tl, y_tl in tl_patch_coordinates]

            #     # tokenize crop coordinates for the bounding boxes of the respective patches
            #     patch_limits_tknzd = [torch.LongTensor(self.bbox_tokenizer._crop_encoder(bbox))[None].to(self.device)
            #                           for bbox in patch_limits]  # list of length l with tensors of shape (1, 2)
            #     print(patch_limits_tknzd[0].shape)
            #     # cut tknzd crop position from conditioning
            #     assert isinstance(cond, dict), 'cond must be dict to be fed into model'
            #     cut_cond = cond['c_crossattn'][0][..., :-2].to(self.device)
            #     print(cut_cond.shape)

            #     adapted_cond = torch.stack([torch.cat([cut_cond, p], dim=1) for p in patch_limits_tknzd])
            #     adapted_cond = rearrange(adapted_cond, 'l b n -> (l b) n')
            #     print(adapted_cond.shape)
            #     adapted_cond = self.get_learned_conditioning(adapted_cond)
            #     print(adapted_cond.shape)
            #     adapted_cond = rearrange(adapted_cond, '(l b) n d -> l b n d', l=z.shape[-1])
            #     print(adapted_cond.shape)

            #     cond_list = [{'c_crossattn': [e]} for e in adapted_cond]

            # else:
            #     cond_list = [cond for i in range(z.shape[-1])]  # Todo make this more efficient

            # # apply model by loop over crops
            # output_list = [self.model(z_list[i], t, **cond_list[i]) for i in range(z.shape[-1])]
            # assert not isinstance(output_list[0],
            #                       tuple)  # todo cant deal with multiple model outputs check this never happens

            # o = torch.stack(output_list, axis=-1)
            # o = o * weighting
            # # Reverse reshape to img shape
            # o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
            # # stitch crops together
            # x_recon = fold(o) / normalization

        else:
            # into DiffusionWrapper then into DDPM unet model
            x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    # def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
    #     return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
    #            extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    # def _prior_bpd(self, x_start):
    #     """
    #     Get the prior KL term for the variational lower-bound, measured in
    #     bits-per-dim.
    #     This term can't be optimized, as it only depends on the encoder.
    #     :param x_start: the [N x C x ...] tensor of inputs.
    #     :return: a batch of [N] KL values (in bits), one per batch element.
    #     """
    #     batch_size = x_start.shape[0]
    #     t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
    #     qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
    #     kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
    #     return mean_flat(kl_prior) / np.log(2.0)

    def p_losses(self, z_start, cond, t, noise=None):
        # z_start: 
        #   predicted latent tensor 
        #   [b, t=pred, 3, h, w]
        # cond: 
        #   conditional tensor
        #   [unsure shape]
        # t: 
        #   difussion forward process (add noise process)
        #   value randint[0, 1000step]
        #   shape [b,]

        # TODO: in loss calculation, only support (first 10 frames + 5 frames) or (first 2 frames + 5 frames)

        z_pred      = z_start[:,:self.frame_num.k]
        z_noise     = default(noise, lambda: torch.randn_like(z_pred)).to(self.device)
        z_noisy     = self.q_sample(x_start=z_pred, t=t, noise=z_noise)
        # rank_zero_info(f"z_noisy         min max mean {z_noisy.min()} {z_noisy.max()} {z_noisy.mean()}")
        # start_time = time.time()
        eps_result  = self.apply_model(z_noisy, t, cond)
        # print("!", z_noise.shape, eps_result.shape)
        # rank_zero_info(f"apply_model done              {time.time() - start_time}")
        # rank_zero_info(f"tmp_eps  min max mean {tmp_eps.min()} {tmp_eps.max()} {tmp_eps.mean()}")
        # rank_zero_info(f"tmp_eps_output  min max mean {eps_result.min()} {eps_result.max()} {eps_result.mean()}")
        
        #####################################################################################
        # z_result    = cond # dont set zerolike tensor
        # z_noise     = torch.zeros_like(z_result) # only for padding
        # eps_result  = torch.zeros_like(z_result) # only for padding
        
        # autogression_num = ceil( z_start.shape[1] / self.channels / self.frame_num.k)

        # for i in range(autogression_num):
        #     start_idx   = self.frame_num.k*i*self.channels
        #     end_idx     = self.frame_num.k*(i+1)*self.channels
        #     # TODO: think about how to solve the z_pred_init [(49-10)/5] = 8, we input need padding for it

        #     # if self.cond_stage_key == "condition_frames_optical_flow":
        #     #     z_cond  = z_result[:, -self.frame_num.cond*self.channels:]
        #     #     from models.optical_flow.optical_flow import optical_flow
        #     #     c = optical_flow(z_result).to(self.device)
        #     #     c = self.get_learned_conditioning(c)
        #     #     print(c.shape)
        #     if self.clip_denoised:
        #         z_cond  = z_result[:, -self.frame_num.cond*self.channels:].clamp(-1.,1.)
        #     else:
        #         z_cond  = z_result[:, -self.frame_num.cond*self.channels:]
        #     z_pred      = z_start[:, start_idx:end_idx]
        #     tmp_noise   = default(noise, lambda: torch.randn_like(z_pred)).to(self.device)
        #     z_noisy     = self.q_sample(x_start=z_pred, t=t, noise=tmp_noise)
        #     # rank_zero_info(f"z_noisy         min max mean {z_noisy.min()} {z_noisy.max()} {z_noisy.mean()}")
        #     # start_time = time.time()
        #     # print("!", z_noisy.shape ,z_cond.shape)
        #     tmp_eps     = self.apply_model(z_noisy, t, z_cond)
        #     # rank_zero_info(f"apply_model done              {time.time() - start_time}")
        #     tmp_z       = self.predict_start_from_noise(z_noisy, t=t, noise=tmp_eps)
        #     # rank_zero_info(f"tmp_eps  min max mean {tmp_eps.min()} {tmp_eps.max()} {tmp_eps.mean()}")
        #     z_noise     = torch.cat([z_noise,    tmp_noise] , dim=1)
        #     eps_result  = torch.cat([eps_result, tmp_eps]   , dim=1)
        #     z_result    = torch.cat([z_result,   tmp_z]     , dim=1)

        # # rank_zero_info(f"tmp_eps_output  min max mean {eps_result.min()} {eps_result.max()} {eps_result.mean()}")
        
        # z_noise     = z_noise   [:, self.frame_num.cond*self.channels : self.frame_num.cond*self.channels+z_start.shape[1]]
        # eps_result  = eps_result[:, self.frame_num.cond*self.channels : self.frame_num.cond*self.channels+z_start.shape[1]]
        
        loss_dict = {}
        prefix = 'train' if self.training else 'val'
        # rank_zero_info("get loss")
        # loss_latent (eps) ------------------------------------------------
        loss_eps = self.get_loss(eps_result, z_noise, mean=False).sum([1, 2, 3, 4])
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
    def sample_log(self, batch, sampler_type, steps, log_every_t, shape, **kwargs):

        print("sampler_type, steps, log_every_t:",sampler_type, steps, log_every_t)

        # x:
        #   predicted frames tensor, ground truch, model will predict it finally, it will be encoded by LDM encoder to z
        #   [b, self.frame_num.pred, c(1 or 3), H, W]
        # z:
        #   predicted latent tensor, input tensor, model will predict it directedly, it comes from x encoded by LDM encoder
        #   [b, self.frame_num.pred, c(only 3), h, w]
        # c:
        #   conditional tensor, model will use it to predict, it will be encoded by condition encoder
        #   [shape depends on contional type]

        import time
        start_time = time.time()        

        # z     c            x       x_rec  xc
        z_pred, z_cond_init, x_pred, x_rec, x_cond_init = self.get_input(batch, self.first_stage_key,
                            return_first_stage_outputs=True,
                            force_c_encode=True,
                            return_original_cond=True)
        
        # print(z_pred.shape, z_cond_init.shape, x_pred.shape, x_rec.shape, x_cond_init.shape)

        # x_cond_init   [64, cond, 3, 64, 64]
        # z_cond_init   [64, cond, 3, 16, 16]
        # x_pred        [64, pred, 3, 64, 64] 
        # z_pred        [64, pred, 3, 16, 16] 
        # x_rec         [64, pred, 3, 64, 64] 

        batch_size = z_pred.shape[0]
        pred_length = z_pred.shape[1]

        x_cond          = x_cond_init
        z_cond          = z_cond_init

        z_sample = z_cond_init
        x_sample = x_cond_init

        z_intermediates = []

        z_intermediates_real = z_pred.unsqueeze(0)

        autoregression = ceil( pred_length / self.frame_num.k)
        
        now_cond  = z_cond_init

        for _ in range(autoregression):
            now_noisy = torch.randn(batch_size, self.frame_num.k, self.channels, self.image_size, self.image_size).to(self.device)
            if sampler_type == 'ddim':
                ddim_sampler = DDIMSampler(self)
                tmp_z_sample, tmp_z_intermediates = ddim_sampler.sample(steps, batch_size, shape, conditioning=now_cond, verbose=False, x_T=now_noisy, log_every_t=log_every_t, **kwargs)
                z_intermediates.append(torch.stack(tmp_z_intermediates['x_inter'])) # 'pred_x0' is DDIM's predicted x0
            elif sampler_type == 'ddpm':
                tmp_z_sample, tmp_z_intermediates = self.sample(now_cond, shape, batch_size=batch_size, return_intermediates=True, x_T=now_noisy, verbose=False)
                z_intermediates.append(torch.stack(tmp_z_intermediates))
            elif sampler_type == 'dpmsolver':             
                dpmsolver_sampler = DPMSolverSampler(self)
                tmp_z_sample, tmp_z_intermediates = dpmsolver_sampler.sample(steps, batch_size, shape, conditioning=now_cond, verbose=False, x_T=now_noisy, log_every_t=log_every_t, **kwargs)
                z_intermediates.append(torch.stack(tmp_z_intermediates))
            else:
                NotImplementedError()

            z_sample = torch.cat([z_sample, tmp_z_sample], dim=1)

            # print("tmp_z_sample", tmp_z_sample.shape) # [64, 5, 3, 16, 16]
            tmp_x_sample = self.video_batch_decode(tmp_z_sample).clamp(-1.,1.)
            # print("tmp_x_samples", tmp_x_sample.shape) #[64, 5, 3, 64, 64]
            x_sample =  torch.cat([x_sample, tmp_x_sample], dim=1)
            
            x_cond = torch.cat([x_cond, x_sample], dim=1)[:,-self.frame_num.cond:]
            # print("x_cond", x_cond.shape) # [64, 10, 3, 64, 64]

            if self.cond_stage_key == "condition_frames_latent":
                tmp_z_cond          = tmp_z_sample
                z_cond              = torch.cat([z_cond,         tmp_z_cond],         dim=1)[:,-self.frame_num.cond:]
                now_cond = z_cond
            else:
                NotImplementedError()
            

        # rank_zero_info(f"get input encode")
        # start = time.time()
        # rank_zero_info(f"get input encode done         {time.time() - start}")  # about 0.18 seconds

        z_intermediates = torch.cat(z_intermediates, dim=2)
        z_intermediates = z_intermediates[:,:,:z_intermediates_real.shape[2]]
        z_intermediates = torch.cat([z_intermediates_real, z_intermediates], dim=0)

        z_sample = z_sample[:,self.frame_num.cond:self.frame_num.cond+pred_length]
        x_sample = x_sample[:,self.frame_num.cond:self.frame_num.cond+pred_length]

        # rank_zero_info("sample_log autogression done")
        # print("samples", samples.shape)
        # print("intermediates", intermediates.shape)
        # samples torch.Size([8, 150, 16, 16])
        # intermediates torch.Size([4, 8, 150, 16, 16])

        print(f"fps: {autoregression * self.frame_num.k / (time.time() - start_time)}")

        return {
            'x_rec': x_rec,
            'x_origin': x_pred,
            'z_origin': z_pred,
            'condition': x_cond_init,
            'x_sample': x_sample,
            'z_sample': z_sample,
            'z_intermediates': z_intermediates,
        }
        
        # batch_size = z_pred.shape[0]
        # pred_length = z_pred.shape[1]

        # x_cond = x_cond_init
        # z_cond = z_cond_init

        # z_sample = z_cond_init
        # x_sample = x_cond_init

        # z_intermediates = []

        # z_intermediates_real = z_pred.unsqueeze(0)

        # autoregression = ceil( pred_length / self.frame_num.k)

        # for _ in range(autoregression):
        #     now_noisy = torch.randn(batch_size, self.frame_num.k, self.channels, self.image_size, self.image_size).to(self.device)
        #     now_cond  = z_cond[:,-self.frame_num.cond:]
        #     if sampler_type == 'ddim':
        #         ddim_sampler = DDIMSampler(self)
        #         tmp_z_sample, tmp_z_intermediates = ddim_sampler.sample(steps, batch_size, shape, conditioning=now_cond, verbose=False, x_T=now_noisy, log_every_t=log_every_t, **kwargs)
        #         z_intermediates.append(torch.stack(tmp_z_intermediates['x_inter'])) # 'pred_x0' is DDIM's predicted x0
        #     elif sampler_type == 'ddpm':
        #         tmp_z_sample, tmp_z_intermediates = self.sample(now_cond, shape, batch_size=batch_size, return_intermediates=True, x_T=now_noisy, verbose=False)
        #         z_intermediates.append(torch.stack(tmp_z_intermediates))
        #     elif sampler_type == 'dpmsolver':             
        #         dpmsolver_sampler = DPMSolverSampler(self)
        #         tmp_z_sample, tmp_z_intermediates = dpmsolver_sampler.sample(steps, batch_size, shape, conditioning=now_cond, verbose=False, x_T=now_noisy, log_every_t=log_every_t, **kwargs)
        #         z_intermediates.append(torch.stack(tmp_z_intermediates))
        #     else:
        #         NotImplementedError()

        #     z_sample = torch.cat([z_sample, tmp_z_sample], dim=1)

        #     # print("tmp_z_sample", tmp_z_sample.shape) # [64, 5, 3, 16, 16]
        #     tmp_x_sample = self.video_batch_decode(tmp_z_sample).clamp(-1.,1.)
        #     # print("tmp_x_samples", tmp_x_sample.shape) #[64, 5, 3, 64, 64]
        #     x_sample =  torch.cat([x_sample, tmp_x_sample], dim=1)
            
        #     x_cond = torch.cat([x_cond, x_sample], dim=1)[:,-self.frame_num.cond:]
        #     # print("x_cond", x_cond.shape) # [64, 15, 3, 64, 64]

        #     if self.cond_stage_key == "condition_frames_latent":
        #         # option 1: predicted z -> predicted x -> cond z
        #         # tmp_z_cond= self.video_batch_encode(x_cond)
        #         # option 2 [better performance]: predicted z -> cond z
        #         tmp_z_cond = tmp_z_sample
        #     elif self.cond_stage_key == "condition_frames_optical_flow":
        #         from models.optical_flow.optical_flow import optical_flow
        #         tmp_z_cond = optical_flow(x_cond).to(self.device)
        #     elif self.cond_stage_key == "condition_frames_motion_state":
        #         from models.motion_state.motion_state import motion_state
        #         tmp_z_cond = motion_state(x_cond).to(self.device)
        #     else:
        #         tmp_z_cond = self.get_learned_conditioning(x_cond.to(self.device))
            
        #     z_cond = torch.cat([z_cond, tmp_z_cond], dim=1)[:,-self.frame_num.cond:]
        #     # print("z_cond.shape",z_cond.shape)

        # # rank_zero_info(f"get input encode")
        # # start = time.time()
        # # rank_zero_info(f"get input encode done         {time.time() - start}")  # about 0.18 seconds

        # z_intermediates = torch.cat(z_intermediates, dim=2)
        # z_intermediates = z_intermediates[:,:,:z_intermediates_real.shape[2]]
        # z_intermediates = torch.cat([z_intermediates_real, z_intermediates], dim=0)

        # z_sample = z_sample[:,self.frame_num.cond:self.frame_num.cond+pred_length]
        # x_sample = x_sample[:,self.frame_num.cond:self.frame_num.cond+pred_length]

        # # rank_zero_info("sample_log autogression done")
        # # print("samples", samples.shape)
        # # print("intermediates", intermediates.shape)
        # # samples torch.Size([8, 150, 16, 16])
        # # intermediates torch.Size([4, 8, 150, 16, 16])

        # print(f"fps: {autoregression * self.frame_num.k / (time.time() - start_time)}")

        # return {
        #     'x_rec': x_rec,
        #     'x_origin': x_pred,
        #     'z_origin': z_pred,
        #     'condition': x_cond_init,
        #     'x_sample': x_sample,
        #     'z_sample': z_sample,
        #     'z_intermediates': z_intermediates,
        # }
    
    # @torch.no_grad()
    # def progressive_denoising_log(self, cond, batch_size, x_T, shape):
    #     # x_T [bs, total*ch, h_latent, w_latent]
    #     z_start = x_T

    #     z_cond = z_start[:, :self.frame_num.cond*self.channels]
    #     z_pred = z_start[:, self.frame_num.cond*self.channels:]
        
    #     z_intermediates_real = z_pred.unsqueeze(0)
    #     z_result = z_cond

    #     z_intermediates = []

    #     autogression_num = ceil( (z_start.shape[1]/self.channels - self.frame_num.cond) / self.frame_num.k)

    #     # FIXME: should fix 现在没用到这个
    #     for i in range(autogression_num):
    #         z_cond      = z_result[:, -self.frame_num.cond*self.channels:]
    #         z_noisy     = torch.randn(z_start.shape[0], self.frame_num.k*self.channels, self.image_size, self.image_size).to(self.device)
    #         x_input     = torch.cat([z_cond, z_noisy], dim=1)
    #         tmp_samples, tmp_intermediates = self.progressive_denoising(x_T=x_input, cond=cond, shape=shape, batch_size=batch_size, verbose=False)
    #         z_result = torch.cat([z_result, tmp_samples[:,-self.frame_num.k*self.channels:]], dim=1)
    #         z_intermediates.append(torch.stack(tmp_intermediates)[:,:,-self.frame_num.k*self.channels:])
        
    #     z_intermediates = torch.cat(z_intermediates, dim=2)
    #     z_intermediates = z_intermediates[:,:,:z_intermediates_real.shape[2]]
    #     z_intermediates = torch.cat([z_intermediates_real, z_intermediates], dim=0)
 
    #     samples         = z_result
    #     intermediates   = z_intermediates

    #     # rank_zero_info("progressive_denoising_log done")
    #     # print("samples", samples.shape)
    #     # print("intermediates", intermediates.shape)
    #     # # samples torch.Size([8, 150, 16, 16])
    #     # # intermediates torch.Size([4, 8, 150, 16, 16])

    #     return samples, intermediates


    @torch.no_grad()
    def log_videos(self, batch, split, save_video_num, calculate_video_num, steps=50, log_every_t=10, eta=1., sampler_type = "dpmsolver", **kwargs):
        log_metrics = dict()
        log = dict()

        shape = (self.frame_num.k, self.channels, self.image_size, self.image_size)

        with self.ema_scope("Plotting denoise"):
            result = self.sample_log(batch, sampler_type=sampler_type, steps=steps, log_every_t=log_every_t, eta=eta, shape=shape)

        log["recon"] = result['x_rec'][:save_video_num]
        log["x_origin"] = result['x_origin'][:save_video_num]
        log["z_origin"] = result['z_origin'][:save_video_num]
        if self.model.conditioning_key is not None:
            log["condition"] = result['condition'][:save_video_num]
        log["z_sample"] = result['z_sample'][:save_video_num]
        log["x_sample"] = result['x_sample'][:save_video_num]
        log["x_denoise_row"] = self._get_denoise_row(result['z_intermediates'][:,:save_video_num])

        log_metrics[f'{split}/z0_min'] = result['z_sample'].min() 
        log_metrics[f'{split}/z0_max'] = result['z_sample'].max() 
        log_metrics[f'{split}/x0_min'] = result['x_sample'].min() 
        log_metrics[f'{split}/x0_max'] = result['x_sample'].max() 

        # rank_zero_info(f"-------------------- x -- min max mean {x.min()} {x.max()} {x.mean()}")
        # rank_zero_info(f"---------------- x_rec -- min max mean {x_rec.min()} {x_rec.max()} {x_rec.mean()}")
        # rank_zero_info(f"-------------------- z -- min max mean {z.min()} {z.max()} {z.mean()}")
        # rank_zero_info(f"---------- tmp_samples -- min max mean {tmp_samples.min()} {tmp_samples.max()} {tmp_samples.mean()}")
        # rank_zero_info(f"------- ddim x_samples -- min max mean {x_samples.min()} {x_samples.max()} {x_samples.mean()}")
        
        videos1 = result['x_sample'][:calculate_video_num].clamp(-1.,1.)
        videos2 = result['x_origin'][:calculate_video_num].clamp(-1.,1.)
        videos1 = (videos1+1.)/2.
        videos2 = (videos2+1.)/2.

        print(videos1.shape, videos2.shape)

        ssim = calculate_ssim1(videos1, videos2)
        psnr = calculate_psnr1(videos1, videos2)
        lpips = calculate_lpips1(videos1, videos2, self.device)

        pred_length = videos1.shape[1]

        log_metrics[f'{split}/pixel_mse/pred{pred_length}'] = self.get_loss(videos1, videos2, mean=False).sum([1, 2, 3, 4]).mean()
        log_metrics[f'{split}/avg/ssim/pred{pred_length}']  = ssim[0]
        log_metrics[f'{split}/std/ssim/pred{pred_length}']  = ssim[1]
        log_metrics[f'{split}/avg/psnr/pred{pred_length}']  = psnr[0]
        log_metrics[f'{split}/std/psnr/pred{pred_length}']  = psnr[1]
        log_metrics[f'{split}/avg/lpips/pred{pred_length}'] = lpips[0]
        log_metrics[f'{split}/std/lpips/pred{pred_length}'] = lpips[1]

        videos1 = torch.cat([result['condition'][:calculate_video_num], videos1],dim=1).clamp(-1.,1.)
        videos2 = torch.cat([result['condition'][:calculate_video_num], videos2],dim=1).clamp(-1.,1.)

        print(videos1.shape, videos2.shape)

        if pred_length >= 10: 
            fvd = calculate_fvd1(videos1, videos2, self.device)
            log_metrics[f'{split}/fvd/cond{self.frame_num.cond}pred{pred_length}']  = fvd
                    
        return log, log_metrics


    # @torch.no_grad()
    # def log_videos(self, batch, split, save_video_num, calculate_video_num, sample=True, steps=50, log_every_t=10, eta=1., return_keys=None, sampler_type = "dpmsolver",
    #                quantize_denoised=True, inpaint=True, plot_denoise_rows=True, plot_progressive_rows=True,
    #                plot_diffusion_rows=True, **kwargs):
    #     log_metrics = dict()
    #     log = dict()

    #     import time
    #     start_time = time.time()
        
    #     z, c, x, x_rec, xc = self.get_input(batch, self.first_stage_key,
    #                                        return_first_stage_outputs=True,
    #                                        force_c_encode=True,
    #                                        return_original_cond=True,
    #                                        bs=calculate_video_num)
        
    #     # TODO: a little problem
    #     shape = (self.frame_num.k, self.channels, self.image_size, self.image_size)

    #     log["input"] = x[:save_video_num]
    #     # rank_zero_info(f"-------------------- x -- min max mean {x.min()} {x.max()} {x.mean()}")
    #     log["recon"] = x_rec[:save_video_num]
    #     # rank_zero_info(f"---------------- x_rec -- min max mean {x_rec.min()} {x_rec.max()} {x_rec.mean()}")

    #     # print(f"z    {z.shape}")
    #     # print(f"c    {c.shape}")
    #     # print(f"x    {x.shape}")
    #     # print(f"xrec {x_rec.shape}")
    #     # print(f"xc   {xc.shape}")

    #     # TODO: do it
    #     if self.model.conditioning_key is not None:
    #         log["condition"] = xc[:save_video_num]
    #         # if hasattr(self.cond_stage_model, "decode"):
    #         #     xc = self.cond_stage_model.decode(c)
    #         #     log["conditioning"] = xc
    #         # elif self.cond_stage_key in ["caption"]:
    #         #     xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["caption"])
    #         #     log["conditioning"] = xc
    #         # elif self.cond_stage_key == 'class_label':
    #         #     xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"])
    #         #     log['conditioning'] = xc
    #         # elif isimage(xc):
    #         #     log["conditioning"] = xc
    #         # if ismap(xc):
    #         #     log["original_conditioning"] = self.to_rgb(xc)

    #     # plot diffusion add noise process
    #     # if plot_diffusion_rows:
    #     #     log["diffusion_row"] = []
    #     #     for video_idx in range(save_video_num):
    #     #         # a latent video: [60[:10], 3, 16, 16]
    #     #         latent_video = z.reshape(z.shape[0], -1, 3, z.shape[-2], z.shape[-1])[video_idx]
    #     #         # print(latent_video.shape)
    #     #         n_row = latent_video.shape[0]
    #     #         # get diffusion row
    #     #         diffusion_grid = []
    #     #         for t in range(self.num_timesteps):
    #     #             if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
    #     #                 t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
    #     #                 t = t.to(self.device).long()
    #     #                 noise = torch.randn_like(latent_video)
    #     #                 z_noisy = self.q_sample(x_start=latent_video, t=t, noise=noise)
    #     #                 diffusion_grid.append(self.decode_first_stage(z_noisy))

    #     #         # n_log_step, n_row, C, H, W
    #     #         diffusion_grid = torch.stack(diffusion_grid)
    #     #         diffusion_grid = rearrange(diffusion_grid, 'n b c h w -> (n b) c h w')
    #     #         diffusion_grid = make_grid(diffusion_grid, nrow=n_row)
    #     #         log["diffusion_row"].append(diffusion_grid)

    #     #     log["diffusion_row"] = torch.stack(log["diffusion_row"])
    #         # print('log["diffusion_row"]', log["diffusion_row"].shape)

    #     # get denoise row
    #     if sample:
    #         # rank_zero_info(f"-------------------- z -- min max mean {z.min()} {z.max()} {z.mean()}")
            
    #         with self.ema_scope("Plotting denoise"):
    #             tmp_samples, tmp_z_denoise_row = self.sample_log(z_T=z, cond=c, sampler_type=sampler_type, steps=steps, log_every_t=log_every_t, eta=eta, shape=shape)
    #         tmp_samples = tmp_samples.reshape(z.shape[0], -1, 3, z.shape[-2], z.shape[-1])

    #         rank_zero_info(f"---------- tmp_samples -- min max mean {tmp_samples.min()} {tmp_samples.max()} {tmp_samples.mean()}")
    #         log["z_sample"] = tmp_samples[:save_video_num]
    #         log["z_origin"] = z.reshape(z.shape[0], -1, 3, z.shape[-2], z.shape[-1])[:save_video_num]

    #         log_metrics[f'{split}/z0_min'] = tmp_samples.min() 
    #         log_metrics[f'{split}/z0_max'] = tmp_samples.max() 

    #         x_samples = self.video_batch_decode(tmp_samples)

    #         log_metrics[f'{split}/x0_min'] = x_samples.min() 
    #         log_metrics[f'{split}/x0_max'] = x_samples.max() 

    #         rank_zero_info(f"------- ddim x_samples -- min max mean {x_samples.min()} {x_samples.max()} {x_samples.mean()}")
    #         log["ddim200"] = x_samples[:save_video_num]

    #         pred_length = x_samples.shape[1]
    #         print("pred_length", pred_length)
            
    #         print(f"{split}/fps/{steps}-sample: {pred_length / (time.time() - start_time)}")

    #         videos1 = x_samples.clamp(-1.,1.) # [:,self.frame_num.cond:]
    #         videos2 = x.clamp(-1.,1.)
    #         videos1 = (videos1+1.)/2.
    #         videos2 = (videos2+1.)/2.

    #         print(videos1.shape, videos2.shape)

    #         ssim = calculate_ssim1(videos1, videos2)
    #         psnr = calculate_psnr1(videos1, videos2)
    #         lpips = calculate_lpips1(videos1, videos2, self.device)

    #         log_metrics[f'{split}/pixel_mse/pred{pred_length}'] = self.get_loss(videos1, videos2, mean=False).sum([1, 2, 3, 4]).mean()
    #         log_metrics[f'{split}/avg/ssim/pred{pred_length}']  = ssim[0]
    #         log_metrics[f'{split}/std/ssim/pred{pred_length}']  = ssim[1]
    #         log_metrics[f'{split}/avg/psnr/pred{pred_length}']  = psnr[0]
    #         log_metrics[f'{split}/std/psnr/pred{pred_length}']  = psnr[1]
    #         log_metrics[f'{split}/avg/lpips/pred{pred_length}'] = lpips[0]
    #         log_metrics[f'{split}/std/lpips/pred{pred_length}'] = lpips[1]

    #         videos1 = torch.cat([xc,videos1],dim=1).clamp(-1.,1.)
    #         videos2 = torch.cat([xc,videos2],dim=1).clamp(-1.,1.)

    #         print(videos1.shape, videos2.shape)

    #         if videos1.shape[1] >= 10: 
    #             fvd = calculate_fvd1(videos1, videos2, self.device)
    #             log_metrics[f'{split}/fvd/cond{self.frame_num.cond}pred{pred_length}']  = fvd
                        
    #         if plot_denoise_rows:
    #             denoise_grids = self._get_denoise_row(tmp_z_denoise_row[:,:save_video_num])
    #             log["ddim200_row"] = denoise_grids

    #         # if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(self.first_stage_model, IdentityFirstStage):
    #         #     # also display when quantizing x0 while sampling
    #         #     with self.ema_scope("Plotting Quantized Denoised"):
    #         #         samples, _ = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,ddim_steps=ddim_steps,eta=ddim_eta,quantize_denoised=True, x_T=z, shape=shape)
    #         #     samples = samples.reshape(N, -1, 3, z.shape[-2], z.shape[-1])

    #         #     x_samples = []
    #         #     for i in range(N):
    #         #         x_samples.append(self.decode_first_stage(samples[i]))
    #         #     x_samples = torch.stack(x_samples) 
    #         #     # rank_zero_info(f"----- q ddim x_samples -- min max mean {x_samples.min()} {x_samples.max()} {x_samples.mean()}")
    #         #     log["ddim200_quantized"] = x_samples

    #     # if plot_progressive_rows:
    #     #     with self.ema_scope("Plotting Progressives ddpm"):
    #     #         tmp_samples, progressives = self.progressive_denoising_log(cond=c, batch_size=N, x_T=z, shape=shape)
    #     #     tmp_samples = tmp_samples.reshape(N, -1, 3, z.shape[-2], z.shape[-1])
    #     #     prog_row = self._get_denoise_row(progressives, desc="Progressive Generation")
    #     #     log["ddpm1000_row"] = prog_row

    #     #     x_samples = []
    #     #     for i in range(N):
    #     #         x_samples.append(self.decode_first_stage(tmp_samples[i]))
    #     #     x_samples = torch.stack(x_samples) 
    #     #     log["ddpm1000"] = x_samples

    #         # rank_zero_info(f"------- ddpm x_samples -- min max mean {x_samples.min()} {x_samples.max()} {x_samples.mean()}")
            

    #     # return_keys = None
    #     if return_keys:
    #         if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
    #             return log
    #         else:
    #             return {key: log[key] for key in return_keys}
            
    #     return log, log_metrics

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(
            params, 
            lr=lr,
            betas=(0.9, 0.999),
            eps=1.0e-08,
            weight_decay=0.0,
            amsgrad=False
        )
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
        # x:
        #   [b, t, c, h, w] -> [b ,t*c, h, w]
        # c_concat:
        #   list -> [b, c', h, w] -> [b, c', h, w]
        # c_crossattn
        #   [b, length, dim_length]

        assert x.shape[2] == 3, "video channels should be 3"

        import einops
        x = einops.rearrange(x, "b t c h w -> b (t c) h w")

        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)

        elif self.conditioning_key == 'concat':

            # FIXME: Dont use c_concat[i] = rearrange c_concat[i] ......
            # list index is dangerous here, you can easily forget list is not deep copy here!!!!!
            cc = []
            for i in range(len(c_concat)):
                cc.append(einops.rearrange(c_concat[i], "b t c h w -> b (t c) h w"))
            x_with_c = torch.cat([x] + cc, dim=1)
            out = self.diffusion_model(x_with_c, t)

        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc)

        elif self.conditioning_key == 'hybrid':
            x_with_c = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x_with_c, t, context=cc)

        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)

        else:
            raise NotImplementedError()
        
        out = einops.rearrange(out, "b (t c) h w -> b t c h w", c = 3)

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