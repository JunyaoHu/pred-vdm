import argparse, os, sys, glob, datetime, yaml
import torch
import time
import numpy as np
from tqdm import trange

from omegaconf import OmegaConf
from PIL import Image
import mediapy as media

from ldm.models.diffusion.ddim import DDIMSampler
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


@torch.no_grad()
def convsample(model, shape, return_intermediates=True,verbose=True,make_prog_row=False):
    if not make_prog_row:
        return model.p_sample_loop(None, shape, return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(None, shape, verbose=True)


@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(model, batch_size, vanilla=False, custom_steps=None, eta=1.0,):
    log = dict()

    # (10, 180, 16, 16)
    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        if vanilla:
            sample, progrow = convsample(model, shape,
                                         make_prog_row=True)
        else:
            sample, intermediates = convsample_ddim(model,  steps=custom_steps, shape=shape,
                                                    eta=eta)

        t1 = time.time()

    """
    sample
    |
    |   reshape  ('b t*c h w -> b t c h w') -> connot go by rearrange
    v
    tmp_sample
    |
    |   index ('b t c h w -> t c h w')
    v
    tmp_sample[i]
    |
    |   Decoder ('t c h w -> t c h*f w*f')
    |   append and stack  ('t c h w -> b t c h w')
    v
    x_sample (prepare for output)
    """

    tmp_sample = sample.reshape(batch_size, -1, model.channels, sample.shape[-2], sample.shape[-1])
    x_sample = []
    print("decode")
    for i in range(batch_size):
        x_sample.append(model.decode_first_stage(tmp_sample[i]))
    print("decode done")
    x_sample = torch.stack(x_sample)
    del tmp_sample
    print("x_rec", x_sample.shape)

    # output [-1,1]
    print(torch.min(sample), torch.min(x_sample))
    print(torch.max(sample), torch.max(x_sample))

    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    print(f'Throughput for this batch: {log["throughput"]}')
    return log

def run(model, logdir, batch_size=50, vanilla=False, custom_steps=None, eta=None, n_samples=50000, nplog=None):
    if vanilla:
        print(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        print(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')

    tstart = time.time()
    # n_saved = len(glob.glob(os.path.join(logdir,'*.png')))-1
    n_saved = 0
    if model.cond_stage_model is None:
        # all_videos = []

        print(f"Running unconditional sampling for {n_samples} samples")
        for _ in trange(n_samples // batch_size, desc="Sampling Batches (unconditional)"):
            logs = make_convolutional_sample(model, batch_size=batch_size,
                                             vanilla=vanilla, custom_steps=custom_steps,
                                             eta=eta)
            n_saved += 1
            print(f"sampling of {n_saved} videos finished in {(time.time() - tstart):.2} seconds.")
            n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample")
            # all_videos.extend([custom_to_np(logs["sample"])])
            if n_saved >= n_samples:
                print(f'Finish after generating {n_saved} samples')
                break
        # all_img = np.concatenate(all_videos, axis=0)
        # all_img = all_img[:n_samples]
        # shape_str = "x".join([str(x) for x in all_img.shape])
        # nppath = os.path.join(nplog, f"{shape_str}-samples.npz")
        # np.savez(nppath, all_img)

    else:
       raise NotImplementedError('Currently only sampling for unconditional models supported.')

    print(f"sampling of {n_saved} videos finished in {(time.time() - tstart):.2} seconds.")


def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    for k in logs:
        # if k == "sample" to get x_sample
        if k == key:
            # [bs, 60, 3, 64, 64]
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    video = custom_to_video(x)
                    video_path = os.path.join(path, f"{key}_{n_saved:06}.gif")
                    media.write_video(video_path, video, fps=10, codec='gif')
                    n_saved += 1
            # else:
            #     npbatch = custom_to_np(batch)
            #     shape_str = "x".join([str(x) for x in npbatch.shape])
            #     nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
            #     np.savez(nppath, npbatch)
            #     n_saved += npbatch.shape[0]
    return n_saved


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
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=20
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
        "-v",
        "--vanilla_sample",
        default=False,
        action='store_true',
        help="vanilla sampling (default option is DDIM sampling)?",
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
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=50
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=10
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


if __name__ == "__main__":
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
    base_configs = sorted(glob.glob(os.path.join(logdir, "config.yaml")))
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
    # print(model)
    print(f"global step: {global_step}")
    print(75 * "=")

    print("logging to:")
    logdir = os.path.join(logdir, "samples", f"{global_step:08}")
    video_logdir = os.path.join(logdir, "video")
    numpy_logdir = os.path.join(logdir, "numpy")

    os.makedirs(video_logdir)
    os.makedirs(numpy_logdir)
    print(logdir)
    print(video_logdir)
    print(numpy_logdir)
    print(75 * "=")

    # write config out
    sampling_file = os.path.join(logdir, "sampling_config.yaml")
    sampling_conf = vars(opt)
    with open(sampling_file, 'w') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    print(sampling_conf)
    
    run(model, video_logdir, eta=opt.eta,
        vanilla=opt.vanilla_sample,  n_samples=opt.n_samples, custom_steps=opt.custom_steps,
        batch_size=opt.batch_size, nplog=numpy_logdir)

    # import cv2
    # x1 = cv2.imread('icon.jpg', 1)
    # x1 = cv2.resize(x1, (64, 64),  interpolation= cv2.INTER_LINEAR)
    # x1 = torch.from_numpy(x1).permute(2,0,1).unsqueeze(0)
    # x1 = x1.to("cuda:0")
    # print(x1.shape) # (w,h,c)

    # x2 = model.encode_first_stage(x1)
    # x3 = model.decode_first_stage(x2)

    # print(x1.shape, x2.shape, x3.shape)

    # print("done.")

    # CUDA_VISIBLE_DEVICES=0,1 python scripts/sample_diffusion.py -r models/ldm/celeba256/model.ckpt -l ./logs_sampling -n 20
    # CUDA_VISIBLE_DEVICES=0 python scripts/sample_diffusion.py -r models/ldm/kth_64/checkpoints/last.ckpt -l ./logs_sampling -n 20
    # CUDA_VISIBLE_DEVICES=0 python scripts/sample_diffusion.py -r models/ldm/kth_64/checkpoints/last.ckpt -l ./logs_sampling -n 256 --batch_size 256 --custom_steps 200