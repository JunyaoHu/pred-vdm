import argparse, os, sys, glob, datetime, yaml
import torch
import time
import numpy as np
from tqdm import trange

from omegaconf import OmegaConf
from PIL import Image
import mediapy as media

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.util import instantiate_from_config

rescale = lambda x: (x + 1.) / 2.

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def custom_to_video(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(0, 2, 3, 1).numpy()
    return x

def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample

def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


# @torch.no_grad()
# def convsample(model, shape, return_intermediates=True,verbose=True,make_prog_row=False):
#     if not make_prog_row:
#         return model.p_sample_loop(None, shape, return_intermediates=return_intermediates, verbose=verbose)
#     else:
#         return model.progressive_denoising(None, shape, verbose=True)


# @torch.no_grad()
# def convsample_ddim(model, steps, shape, eta=1.0):
#     ddim_sampler = DDIMSampler(model)
#     bs = shape[0]
#     shape = shape[1:]
#     samples, intermediates = ddim_sampler.sample(steps, batch_size=bs, shape=shape, conditioning=now_cond, eta=eta, log_every_t=log_every_t, verbose=False,)
#     return samples, intermediates



@torch.no_grad()
def sample_log(model, batch, frame_cond, frame_pred, is_valid, sampler_type, steps, log_every_t, shape, **kwargs):

    if is_valid:
        batch = batch[:,:frame_cond+frame_pred]

        print("sampler_type, steps, log_every_t:",sampler_type, steps, log_every_t)
        
        # print(z_pred.shape, z_cond_init.shape, x_pred.shape, x_rec.shape, x_cond_init.shape)

        # x_cond_init   [64, cond, 3, 64, 64]
        # z_cond_init   [64, cond, 3, 16, 16], [64, cond-1, 1, 16, 16]
        # x_pred        [64, pred, 3, 64, 64] 
        # z_pred        [64, pred, 3, 16, 16] 
        # x_rec         [64, pred, 3, 64, 64] 
        
        if model.frame_num.cond != frame_cond:
            import gradio as gr
            raise gr.Error(f'模型条件帧数 ({model.frame_num.cond}) 和设置条件帧数 ({frame_cond}) 不一致')
            
        import time 
        
        start_time = time.time()
        
        x_origin = batch.cuda()
        z_origin = model.video_batch_encode(x_origin)
        
        x_cond_init = x_origin[:, :model.frame_num.cond]
        print("x_cond_init", x_cond_init.shape)
        z_cond_init = z_origin[:, :model.frame_num.cond]
        print("z_cond_init", z_cond_init.shape)
        
        z_pred = z_origin[:, model.frame_num.cond:]

        batch_size      = batch.shape[0]
        pred_length     = frame_pred

        x_cond          = x_cond_init
        
        z_cond          = z_cond_init
        z_sample        = z_cond_init
        
        x_sample = x_cond_init

        z_intermediates = []

        z_intermediates_real = z_pred.unsqueeze(0)

        import math
        autoregression = math.ceil( pred_length / model.frame_num.k)
        
        now_cond  = z_cond_init

        for _ in range(autoregression):
            now_noisy = torch.randn(batch_size, model.frame_num.k, model.channels, model.image_size, model.image_size).cuda()
            print(sampler_type, "sampler_type")
            if sampler_type == 'DDIM':
                ddim_sampler = DDIMSampler(model)
                print(now_cond.shape, now_noisy.shape)
                tmp_z_sample, tmp_z_intermediates = ddim_sampler.sample(steps, batch_size, shape, conditioning=now_cond, verbose=False, x_T=now_noisy, log_every_t=log_every_t, **kwargs)
                z_intermediates.append(torch.stack(tmp_z_intermediates['x_inter'])) # 'pred_x0' is DDIM's predicted x0
            elif sampler_type == 'DDPM':
                tmp_z_sample, tmp_z_intermediates = model.sample(now_cond, shape, batch_size=batch_size, return_intermediates=True, x_T=now_noisy, verbose=False)
                z_intermediates.append(torch.stack(tmp_z_intermediates))
            elif sampler_type == 'DPM Solver++':             
                dpmsolver_sampler = DPMSolverSampler(model)
                tmp_z_sample, tmp_z_intermediates = dpmsolver_sampler.sample(steps, batch_size, shape, conditioning=now_cond, verbose=False, x_T=now_noisy, log_every_t=log_every_t, **kwargs)
                z_intermediates.append(torch.stack(tmp_z_intermediates))
            else:
                NotImplementedError()

            z_sample = torch.cat([z_sample, tmp_z_sample], dim=1)

            # print("tmp_z_sample", tmp_z_sample.shape) # [64, 5, 3, 16, 16]
            tmp_x_sample = model.video_batch_decode(tmp_z_sample).clamp(-1.,1.)
            # print("tmp_x_samples", tmp_x_sample.shape) #[64, 5, 3, 64, 64]
            x_sample =  torch.cat([x_sample, tmp_x_sample], dim=1)
            
            x_cond = torch.cat([x_cond, x_sample], dim=1)[:,-model.frame_num.cond:]
            # print("x_cond", x_cond.shape) # [64, 10, 3, 64, 64]

            if model.cond_stage_key == "condition_frames_latent":
                tmp_z_cond          = tmp_z_sample
                z_cond              = torch.cat([z_cond,         tmp_z_cond],         dim=1)[:,-model.frame_num.cond:]
                now_cond =  z_cond
            else:
                NotImplementedError()
            
        z_intermediates = torch.cat(z_intermediates, dim=2)
        z_intermediates = z_intermediates[:,:,:z_intermediates_real.shape[2]]
        z_intermediates = torch.cat([z_intermediates_real, z_intermediates], dim=0)

        z_sample = z_sample[:,model.frame_num.cond:model.frame_num.cond+pred_length]
        x_sample = x_sample[:,model.frame_num.cond:model.frame_num.cond+pred_length]

        print(f"fps: {autoregression * model.frame_num.k / (time.time() - start_time)}")

        return {
            'x_origin': x_origin,
            'z_origin': z_pred,
            'condition': x_cond_init,
            'x_sample': x_sample,
            'z_sample': z_sample,
            'z_intermediates': z_intermediates,
        }
    else:
        batch = batch[:,:frame_cond]
        print("sampler_type, steps, log_every_t:",sampler_type, steps, log_every_t)
        # print(z_pred.shape, z_cond_init.shape, x_pred.shape, x_rec.shape, x_cond_init.shape)
        # x_cond_init   [64, cond, 3, 64, 64]
        # z_cond_init   [64, cond, 3, 16, 16], [64, cond-1, 1, 16, 16]
        # x_pred        [64, pred, 3, 64, 64] 
        # z_pred        [64, pred, 3, 16, 16] 
        # x_rec         [64, pred, 3, 64, 64] 
        if model.frame_num.cond != frame_cond:
            import gradio as gr
            raise gr.Error(f'模型条件帧数 ({model.frame_num.cond}) 和设置条件帧数 ({frame_cond}) 不一致')
        import time 
        start_time = time.time()
        x_origin = batch.cuda()
        z_origin = model.video_batch_encode(x_origin)
        x_cond_init = x_origin[:, :model.frame_num.cond]
        print("x_cond_init", x_cond_init.shape)
        z_cond_init = z_origin[:, :model.frame_num.cond]
        print("z_cond_init", z_cond_init.shape)

        batch_size      = batch.shape[0]
        pred_length     = frame_pred

        x_cond          = x_cond_init
        
        z_cond          = z_cond_init
        z_sample        = z_cond_init
        
        x_sample = x_cond_init

        z_intermediates = []

        import math
        autoregression = math.ceil( pred_length / model.frame_num.k)
        
        now_cond  = z_cond_init

        for _ in range(autoregression):
            now_noisy = torch.randn(batch_size, model.frame_num.k, model.channels, model.image_size, model.image_size).cuda()
            print(sampler_type, "sampler_type")
            if sampler_type == 'DDIM':
                ddim_sampler = DDIMSampler(model)
                print(now_cond.shape, now_noisy.shape)
                tmp_z_sample, tmp_z_intermediates = ddim_sampler.sample(steps, batch_size, shape, conditioning=now_cond, verbose=False, x_T=now_noisy, log_every_t=log_every_t, **kwargs)
                z_intermediates.append(torch.stack(tmp_z_intermediates['x_inter'])) # 'pred_x0' is DDIM's predicted x0
            elif sampler_type == 'DDPM':
                tmp_z_sample, tmp_z_intermediates = model.sample(now_cond, shape, batch_size=batch_size, return_intermediates=True, x_T=now_noisy, verbose=False)
                z_intermediates.append(torch.stack(tmp_z_intermediates))
            elif sampler_type == 'DPM Solver++':             
                dpmsolver_sampler = DPMSolverSampler(model)
                tmp_z_sample, tmp_z_intermediates = dpmsolver_sampler.sample(steps, batch_size, shape, conditioning=now_cond, verbose=False, x_T=now_noisy, log_every_t=log_every_t, **kwargs)
                z_intermediates.append(torch.stack(tmp_z_intermediates))
            else:
                NotImplementedError()

            z_sample = torch.cat([z_sample, tmp_z_sample], dim=1)

            # print("tmp_z_sample", tmp_z_sample.shape) # [64, 5, 3, 16, 16]
            tmp_x_sample = model.video_batch_decode(tmp_z_sample).clamp(-1.,1.)
            # print("tmp_x_samples", tmp_x_sample.shape) #[64, 5, 3, 64, 64]
            x_sample =  torch.cat([x_sample, tmp_x_sample], dim=1)
            
            x_cond = torch.cat([x_cond, x_sample], dim=1)[:,-model.frame_num.cond:]
            # print("x_cond", x_cond.shape) # [64, 10, 3, 64, 64]

            if model.cond_stage_key == "condition_frames_latent":
                tmp_z_cond          = tmp_z_sample
                z_cond              = torch.cat([z_cond,         tmp_z_cond],         dim=1)[:,-model.frame_num.cond:]
                now_cond =  z_cond
            else:
                NotImplementedError()
                
        z_intermediates = torch.cat(z_intermediates, dim=2)
        z_intermediates = z_intermediates[:,:,:frame_pred]

        z_sample = z_sample[:,model.frame_num.cond:model.frame_num.cond+pred_length]
        x_sample = x_sample[:,model.frame_num.cond:model.frame_num.cond+pred_length]

        print(f"fps: {autoregression * model.frame_num.k / (time.time() - start_time)}")

        return {
            'x_origin': x_origin,
            'condition': x_cond_init,
            'x_sample': x_sample,
            'z_sample': z_sample,
            'z_intermediates': z_intermediates,
        }

@torch.no_grad()
def make_convolutional_sample(model, batch, frame_cond, frame_pred, is_valid=True, eta=1.0, sampler_type = "DDIM", log_every_t=50, steps=200):
    # 验证，有cond和pred。
    if is_valid:
        log = dict()
        # (10, 180, 16, 16)
        shape = [model.frame_num.k,
                model.channels,
                model.image_size,
                model.image_size]
        with model.ema_scope("Plotting"):
            t0 = time.time()
            result = sample_log(model, batch, frame_cond, frame_pred, is_valid=is_valid, sampler_type=sampler_type, steps=steps, log_every_t=log_every_t, eta=eta, shape=shape)
            t1 = time.time()
        # output [-1,1]
        print(torch.min(result['x_sample']), torch.min(result['z_sample']))
        print(torch.max(result['x_sample']), torch.max(result['z_sample']))
        log = result
        log["time"] = t1 - t0
        return log
    # 测试，只有cond。
    else:
        log = dict()
        # (10, 180, 16, 16)
        shape = [model.frame_num.k,
                model.channels,
                model.image_size,
                model.image_size]
        with model.ema_scope("Plotting"):
            t0 = time.time()
            result = sample_log(model, batch, frame_cond, frame_pred, is_valid=is_valid, sampler_type=sampler_type, steps=steps, log_every_t=log_every_t, eta=eta, shape=shape)
            t1 = time.time()
        # output [-1,1]
        print(torch.min(result['x_sample']), torch.min(result['z_sample']))
        print(torch.max(result['x_sample']), torch.max(result['z_sample']))
        log = result
        log["time"] = t1 - t0
        return log
    
def run(model, logdir, frame_cond, frame_pred, test_video_path='', is_valid=True, sampler_type='DDIM', steps=200, log_every_t=50, comparison_step=1, eta=None, nplog=None):
    print(f'Using {sampler_type} sampling with {steps} sampling steps.')

    start = time.time()
    
    if model.cond_stage_model is None:
        # # all_videos = []

        # print(f"Running unconditional sampling for {n_samples} samples")
        # for _ in trange(n_samples // batch_size, desc="Sampling Batches (unconditional)"):
        #     logs = make_convolutional_sample(model, batch_size=batch_size,
        #                                      vanilla=vanilla, custom_steps=custom_steps,
        #                                      eta=eta)
        #     n_saved += 1
        #     print(f"sampling of {n_saved} videos finished in {(time.time() - tstart):.2} seconds.")
        #     n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample")
        #     # all_videos.extend([custom_to_np(logs["sample"])])
        #     if n_saved >= n_samples:
        #         print(f'Finish after generating {n_saved} samples')
        #         break
        # # all_img = np.concatenate(all_videos, axis=0)
        # # all_img = all_img[:n_samples]
        # # shape_str = "x".join([str(x) for x in all_img.shape])
        # # nppath = os.path.join(nplog, f"{shape_str}-samples.npz")
        # # np.savez(nppath, all_img)
        pass
    else:
        print(f"Running conditional sampling for a sample")
        batch = media.read_video(test_video_path)
        batch = torch.tensor(batch).unsqueeze(0).float()/125.5 - 1
        import einops
        batch = einops.rearrange(batch, 'b t h w c -> b t c h w')

        print(batch.shape,torch.min(batch),torch.max(batch))
        # B T H W C
        logs = make_convolutional_sample(model, batch=batch, frame_cond=frame_cond, frame_pred=frame_pred, is_valid=is_valid, sampler_type=sampler_type, steps=steps,log_every_t=log_every_t, eta=eta)
        
        runtime = (time.time() - start)
        
        print(f"sampling videos finished in {runtime:.2} seconds.")
        
        print(logs['x_origin'].shape)
        print(logs['condition'].shape)
        print(logs['x_sample'].shape)
        print(logs['z_intermediates'].shape)
        
        origin = logs['x_origin'].cpu()
        result = torch.cat([logs['condition'].cpu(), logs['x_sample'].cpu()],dim=1)
        
        origin = (origin +1) /2
        result = (result +1) /2

        if is_valid:
            origin = origin
        else:
            origin = torch.cat([origin, torch.ones_like(logs['x_sample'].cpu())],dim=1)
        
        print(origin.shape, result.shape)
        
        from visualize import visualize
        
        decomposition, ground_truth, prediction, comparsion = visualize(
            save_path=logdir,
            origin=origin,
            result=result,
            save_pic_num=1,
            grid_nrow=1,
            save_pic_row=True,
            save_gif=True,
            cond_frame_num=frame_cond,  
            skip_pic_num=comparison_step
        )
        
        # torch.Size([13, 1, 10, 3, 16, 16])
        step_output_path = os.path.join(logdir, "intermediates.png")
        step_image = model._get_denoise_row(logs['z_intermediates']).cpu()
        step_image = einops.rearrange(step_image.squeeze(), "c h w -> h w c")
        step_image = np.array((step_image + 1.0) / 2.0)  # -1,1 -> 0,1; c,h,w
        media.write_image(step_output_path, step_image)
        
        return decomposition, ground_truth, prediction, comparsion, step_output_path, runtime
        
        # save_logs(logs, logdir, key="x_sample")
        # return {
        #     'x_origin': x_origin,
        #     'z_origin': z_pred,
        #     'condition': x_cond_init,
        #     'x_sample': x_sample,
        #     'z_sample': z_sample,
        #     'z_intermediates': z_intermediates,
        # }
    # raise NotImplementedError('Currently only sampling for unconditional models supported.')

def save_logs(logs, path, key="x_sample", np_path=None):
    for k in logs:
        if k == key:
            # [1, t, 3, 64, 64]
            batch = logs[key]
            for x in batch:
                x = x.squeeze()
                video = custom_to_video(x)
                video_path = os.path.join(path, f"{key}.gif")
                media.write_video(video_path, video, fps=20, codec='gif')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "-type",
        "--sampler_type",
        default='DDIM',
        help="DDPM / DDIM / DPM Solver++",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default=""
    )
    parser.add_argument(
        "-c",
        "--steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=50
    )
    parser.add_argument(
        "--test_video_path",
        type=str,
        nargs="?",
        help="the test_video_path",
        default=""
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="?",
        help="dataset dir",
    )
    return parser


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval()
    return model

def load_model(config, ckpt):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        print(f"no Loading model")
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model, pl_sd["state_dict"])
    return model, global_step

def main():
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    ckpt = None

    if not os.path.exists(opt.resume):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.resume):
        # xxx/yy/checkpoints/checkpointsA.ckpt -> 
        # logdir:   xxx/yy
        # ckpt:     xxx/yy/checkpoints/checkpointsA.ckpt
        paths = opt.resume.split("/")
        logdir = "/".join(paths[:-2])
        ckpt = opt.resume
    else:
        # xxx/yy/ -> 
        # logdir:   xxx/yy
        # ckpt:     xxx/yy/checkpoints/last.ckpt
        assert os.path.isdir(opt.resume), opt.resume
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

    # ["xxx/yy/configs/aaa.yaml"]
    # base_configs = sorted(glob.glob(os.path.join(logdir, "config.yaml")))
    base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
    opt.base = base_configs

    # config management by OmegaConf
    configs = [OmegaConf.load(cfg) for cfg in opt.base]

    # merge unknown parameter when input command
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    # change output log dir  
    print("opt.logdir", opt.logdir)
    if opt.logdir != "":
        locallog = logdir.split(os.sep)[-1] # path/to/checkpoints
        if locallog == "": 
            locallog = logdir.split(os.sep)[-2]
        print(f"Switching logdir from '{logdir}' to '{os.path.join(opt.logdir, locallog)}'")
        logdir = os.path.join(opt.logdir, locallog)
    else:
        logdir = logdir

    print(config)

    model, global_step = load_model(config, ckpt)
    # print(75 * "=")
    # print(model)
    print(f"global step: {global_step}")
    print(75 * "=")

    print("logging to:")
    logdir = os.path.join(logdir, "samples", f"{global_step:08}")
    video_logdir = os.path.join(logdir, "videos")
    # numpy_logdir = os.path.join(logdir, "numpy")

    os.makedirs(video_logdir, exist_ok=True)
    # os.makedirs(numpy_logdir)
    print(logdir)
    print(video_logdir)
    # print(numpy_logdir)
    print(75 * "=")

    # write config out
    sampling_file = os.path.join(logdir, "sampling_config.yaml")
    sampling_conf = vars(opt)
    with open(sampling_file, 'w') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    print(sampling_conf)
    
    run(model, video_logdir, eta=opt.eta,
        vanilla=opt.vanilla_sample, steps=opt.steps,
        test_video_path=opt.test_video_path)
   
def do_predict(root, resume, logdir, test_video_path, frame_cond, frame_pred, comparison_step=1, is_valid=True, sampler_type='DDIM', steps=200, log_every_t=50, eta=1.0):
    if not os.path.exists(resume):
        raise ValueError("Cannot find {}".format(resume))
    if os.path.isfile(resume):
        # xxx/yy/checkpoints/checkpointsA.ckpt -> 
        # logdir:   xxx/yy
        # ckpt:     xxx/yy/checkpoints/checkpointsA.ckpt
        paths = resume.split("/")
        logdir = "/".join(paths[:-2])
        ckpt = resume
    else:
        # xxx/yy/ -> 
        # logdir:   xxx/yy
        # ckpt:     xxx/yy/checkpoints/last.ckpt
        assert os.path.isdir(resume), resume
        logdir = resume.rstrip("/")
        ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

    # ["xxx/yy/configs/aaa.yaml"]
    # base_configs = sorted(glob.glob(os.path.join(logdir, "config.yaml")))
    base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
    base = base_configs

    # config management by OmegaConf
    configs = [OmegaConf.load(cfg) for cfg in base]
    cli = OmegaConf.from_dotlist([])
    configs = OmegaConf.merge(*configs, cli)

    print("opt.logdir", logdir)

    print(75 * "=")
    print(configs)
    print(75 * "=")
    
    # print(configs["model"])
    # print(configs["model"]['params']["first_stage_config"])
    # print(configs["model"]['params']["first_stage_config"]['params']["ckpt_path"])
    
    configs["model"]['params']["first_stage_config"]['params']["ckpt_path"] = os.path.join(root, configs["model"]['params']["first_stage_config"]['params']["ckpt_path"])

    model, global_step = load_model(configs, ckpt)
    # print(75 * "=")
    # print(model)
    print(f"global step: {global_step}")
    print(75 * "=")

    print("logging to:")
    logdir = os.path.join(logdir, "samples", f"{global_step:08}")
    video_logdir = os.path.join(logdir, "videos")
    # numpy_logdir = os.path.join(logdir, "numpy")

    os.makedirs(video_logdir, exist_ok=True)
    # os.makedirs(numpy_logdir)
    print(logdir)
    print(video_logdir)
    # print(numpy_logdir)
    print(75 * "=")
    
    result = run(
        model, 
        video_logdir, 
        frame_cond, 
        frame_pred, 
        test_video_path,
        is_valid=is_valid,
        sampler_type=sampler_type, 
        eta=eta, 
        steps=steps,
        log_every_t=log_every_t,
        comparison_step=comparison_step,
    )
    
    return result
    
# if __name__ == "__main__":
#     main()

    # CUDA_VISIBLE_DEVICES=0 python scripts/sample_diffusion.py -r logs_training/20230501-102105_kth-ldm-vq-f4 -l ./logs_sampling --custom_steps 200 --test_video_path ./video_input.gif