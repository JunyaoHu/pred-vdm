import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image
import mediapy as media
import einops
import wandb

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.strategies import DDPStrategy 
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_parser(**parser_kwargs):

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)

    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="./logs_training",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    return parser

def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        # 实现数据集的准备如下载等，只有 cuda:0 会执行该函数
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        # 实现数据集的定义，每张GPU都会执行该函数, stage 用于标记是用于什么阶段 如'fit(train + validate)', 'validate', 'test', or 'predict'
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = not is_iterable_dataset

        return DataLoader(self.datasets["train"], 
                          batch_size=self.batch_size.train,
                          num_workers=self.num_workers, 
                          worker_init_fn=init_fn,
                          shuffle=shuffle,
                          persistent_workers=True
                          )

    def _val_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        # shuffle = not is_iterable_dataset

        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size.validation,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle,
                          persistent_workers=True
                          )

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        # shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], 
                          batch_size=self.batch_size.test,
                          num_workers=self.num_workers, 
                          worker_init_fn=init_fn, 
                          shuffle=shuffle,
                          persistent_workers=True
                          )

    # def _predict_dataloader(self, shuffle=False):
    #     is_iterable_dataset = isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset)
    #     if is_iterable_dataset or self.use_worker_init_fn:
    #         init_fn = worker_init_fn
    #     else:
    #         init_fn = None
    #     return DataLoader(self.datasets["predict"], 
    #                       batch_size=self.batch_size.predict,
    #                       num_workers=self.num_workers, 
    #                       worker_init_fn=init_fn,
    #                       persistent_workers=True
    #                       )


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    # The `Callback.on_pretrain_routine_start` hook was removed in v1.8. Please use `Callback.on_fit_start` instead.
    # def on_pretrain_routine_start(self, trainer, pl_module):
    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("#### Project config ####")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config, os.path.join(self.cfgdir, "project.yaml"))

            print("#### Lightning config ####")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}), os.path.join(self.cfgdir, "lightning.yaml"))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class VideoLogger(Callback):
    def __init__(self, batch_frequency, max_videos, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_videos_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_frequency = batch_frequency
        self.max_videos = max_videos
        self.logger_log_videos = {
            # pl.loggers.TestTubeLogger: self._testtube,
            pl.loggers.WandbLogger: self._wandb,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_frequency)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_frequency]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_videos_kwargs = log_videos_kwargs if log_videos_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def _testtube(self, pl_module, videos, batch_idx, split):
        # print("_testtube")
        # for k in videos:
        #     if k == "diffusion_row":
        #         for i in range(videos[k].shape[0]):
        #             grid = videos[k][i]
        #             grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
        #             tag = f"{split}/{k}/{i}"
        #             pl_module.logger.experiment.add_image(tag, grid, global_step=pl_module.global_step)
        # for k in images:
        #     grid = torchvision.utils.make_grid(images[k])
        pass

    @rank_zero_only
    def _wandb(self, pl_module, videos, metrics, batch_idx, split):
        # for pytorch-lightning > 1.6.0
        tag = f"{split}"
        # columns = ["input","recon","z_origin","z_sample","ddim200","ddim200_quantized", "ddpm1000", "diffusion_row", "ddim200_row", "ddpm1000_row"]
        columns = ["input", "recon", "z_origin", "z_sample", "condition", "ddim200", "diffusion_row", "ddim200_row"]
        data = []
        rank_zero_info("upload wandb")
        for k in columns:
            if k in ["diffusion_row", "ddim200_row"]:
                grids = einops.rearrange(videos[k], "b c h w -> b h w c")
                if self.rescale:
                    grids = np.array(((grids + 1.0) * 127.5)).astype(np.uint8)  # -1,1 -> 0,255; h,w,c -> uint8
                grids = list([wandb.Image(i) for i in grids])
                data.append(grids)

            elif k in ["input","recon","z_origin","z_sample","ddim200", "condition"]:
                grids = videos[k]
                if self.rescale:
                    grids = np.array(((grids + 1.0) * 127.5)).astype(np.uint8) 
                data.append(wandb.Video(grids, fps=20)) #  (batch, time, channel, height width)
        
        data = [data]
        pl_module.logger.log_table(key=tag, columns=columns, data=data)
        pl_module.logger.log_metrics(metrics)
        rank_zero_info("upload wandb done")
            

    @rank_zero_only
    def log_local(self, save_dir, split, videos, metrics,
                  global_step, current_epoch, batch_idx):
        rank_zero_info("log local")

        metrics_path = os.path.join(save_dir, f"{split}_metrics.txt")
        with open(metrics_path, "a") as f:
            for k in metrics:
                line = "step-{:06}(epoch-{:06})/batch-{:06}/{}: {}\n".format( global_step, current_epoch, batch_idx, k, metrics[k])
                f.write(line)
            f.write("\n")
        
        root = os.path.join(save_dir, "videos", split)
        for k in videos:
            # if k in ["diffusion_row", "ddim200_row", "ddpm1000_row"]:
            if k in ["diffusion_row", "ddim200_row"]:
                for i in range(videos[k].shape[0]):
                    grid = einops.rearrange(videos[k][i], "c h w -> h w c")
                    if self.rescale:
                        grid = np.array((grid + 1.0) / 2.0)  # -1,1 -> 0,1; c,h,w
                    filename = "step-{:06}(epoch-{:06})/batch-{:06}/{}/video-{:06}.png".format( global_step, current_epoch, batch_idx, k, i)
                    path = os.path.join(root, filename)
                    os.makedirs(os.path.split(path)[0], exist_ok=True)
                    media.write_image(path, grid)
            # elif k in ["input","recon","z_origin","z_sample","ddim200","ddim200_quantized", "ddpm1000"]:
            elif k in ["input","recon","z_origin","z_sample","ddim200", "condition"]:
                for i in range(videos[k].shape[0]):
                    video = einops.rearrange(videos[k][i], "t c h w -> t h w c")
                    if self.rescale:
                        video = np.array((video + 1.0) / 2.0) # -1,1 -> 0,1; c,h,w
                    filename = "step-{:06}(epoch-{:06})/batch-{:06}/{}/video-{:06}.gif".format(global_step, current_epoch, batch_idx, k, i)
                    path = os.path.join(root, filename)
                    os.makedirs(os.path.split(path)[0], exist_ok=True)
                    media.write_video(path, video, fps=20, codec='gif')
        
        rank_zero_info("log local done")

    def log_video(self, pl_module, batch, batch_idx, split="train"):
        
        if self.log_on_batch_idx:
            check_idx = batch_idx  
        else:
            check_idx = pl_module.global_step
        
        # batch_idx % self.batch_freq == 0
        if (self.check_frequency(check_idx)
            and hasattr(pl_module, "log_videos") 
            and callable(pl_module.log_videos) 
            and self.max_videos > 0):
            rank_zero_info("log video")
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                videos, metrics = pl_module.log_videos(batch, split=split, **self.log_videos_kwargs)

            # key = ["input","recon","z_origin","z_sample","ddim200", "diffusion_row", "ddim200_row"]
            # now, we not have  "samples_inpainting" "mask" "samples_outpainting" ,"ddim200_quantized", "ddpm1000","ddpm1000_row"
            for k in videos:
                N = min(videos[k].shape[0], self.max_videos)
                videos[k] = videos[k][:N]
                if isinstance(videos[k], torch.Tensor):
                    videos[k] = videos[k].detach().cpu()
                    if self.clamp:
                        videos[k] = videos[k].clamp(-1., 1.)

            self.log_local(pl_module.logger.save_dir, split, videos, metrics, 
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_videos = self.logger_log_videos.get(logger, lambda *args, **kwargs: None)
            logger_log_videos(pl_module, videos, metrics, pl_module.global_step, split)

            if is_train:
                pl_module.train()
                
            rank_zero_info("log video done")

    def check_frequency(self, check_idx):
        if (((check_idx % self.batch_frequency) == 0 or (check_idx in self.log_steps)) 
            and (check_idx > 0 or self.log_first_step)):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                # print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
        #     self.log_video(pl_module, batch, batch_idx, split="train")
        pass

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and pl_module.global_step > 0:
            self.log_video(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and pl_module.global_step > 0:
            self.log_video(pl_module, batch, batch_idx, split="test")


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    #  `Trainer.root_gpu` was deprecated in v1.6 and is no longer accessible as of v1.8. Please use `Trainer.strategy.root_device.index` instead.
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device.index)
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        self.start_time = time.time()
        

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        epoch_time = time.time() - self.start_time
        max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device.index) / 2 ** 20
        mem_free, mem_total = torch.cuda.mem_get_info()
        current_memory = (mem_total - mem_free) / 2 ** 20
        try:
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            current_memory = trainer.training_type_plugin.reduce(current_memory)
            rank_zero_info(f"")
            rank_zero_info(f"Average Epoch time     : {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory    : {max_memory:.2f} MB")
            rank_zero_info(f"Average Current memory : {current_memory:.2f} MB")
        except AttributeError:
            pass


if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
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

        opt.resume_from_checkpoint = ckpt
        # ["xxx/yy/configs/aaa-lightning.yaml",
        #  "xxx/yy/configs/aaa-project.yaml"]
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        # nowname: yy
        nowname = logdir.split("/")[-1]

    # else means it is not resume
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            # [configs/latent-diffusion/kth-ldm-vq-f4.yaml] ->
            # cfg_fname: kth-ldm-vq-f4.yaml
            # cfg_name:  kth-ldm-vq-f4
            # name:      _kth-ldm-vq-f4
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        if opt.postfix != "":
            nowname = now + name + "_" + opt.postfix
        else:
            nowname = now + name
        # nowname: eg. 20230209-160159_kth-ldm-vq-f4
        nowname = now + name + opt.postfix
        # logdir: eg. ./logs/20230209-160159_kth-ldm-vq-f4
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    wandbdir = os.path.join(logdir, "wandb")

    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp
        trainer_config["accelerator"] = "gpu"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if not "gpus" in trainer_config:
            trainer_config["accelerator"] = "cpu"
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model instantiate
        # import model first
        # and then instantiate with config.model 
        model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": False,
                    "id": nowname,
                }
            },
            # 最新版本的pytorch-lightning已经不支持testtube
            # "testtube": {
            #     "target": "pytorch_lightning.loggers.TestTubeLogger",
            #     "params": {
            #         "name": "testtube",
            #         "save_dir": logdir,
            #     }
            # },
        }
        default_logger_cfg = default_logger_cfgs["wandb"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()

        # ps: lightning_config：kth-ldm-vq-f4.yaml 有 lightning, 但是 lightning 下面没有 key="logger" so create() new

        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)

        # instantiate WandbLogger
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # TrainResult/EvalResult 在 1.2.0 已经被移除
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "epoch-{epoch:06}",
                "verbose": True,
                "save_last": True,
                "auto_insert_metric_name": False,
                "save_top_k": 3,
            }
        }
        if hasattr(model, "monitor"):
            # Monitoring val/loss_simple_ema as checkpoint metric.
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg =  OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)

        print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")

        if version.parse(pl.__version__) < version.parse('1.4.0'):
            trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "video_logger": {
                "target": "main.VideoLogger",
                "params": {
                    "batch_frequency": 1000,
                    "max_videos": 4,
                    "clamp": True,
                    "rescale": True,
                }
            },
            "learning_rate_logger": {
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                }
            },
            "cuda_callback": {
                "target": "main.CUDACallback"
            },
        }
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
            print(
                'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
            default_metrics_over_trainsteps_ckpt_dict = {
                'metrics_over_trainsteps_checkpoint':{
                    "target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                    'params': {
                        "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                        "filename": "{epoch:06}-{step:09}",
                        "verbose": True,
                        'save_top_k': -1,
                        'every_n_train_steps': 10000,
                        'save_weights_only': True
                    }
                }
            }
            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
        elif 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']

        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        trainer = Trainer.from_argparse_args(trainer_opt, strategy=DDPStrategy(find_unused_parameters=False), **trainer_kwargs)
        trainer.logdir = logdir

        # data
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup()
        print("#### Data #####")
        for k in data.datasets:
            print(f"{k:10}\t{data.datasets[k].__class__.__name__:10}\t{len(data.datasets[k])}")

        # configure learning rate
        bs, base_lr = config.data.params.batch_size.train, config.model.base_learning_rate
        if not cpu:
            ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
        else:
            ngpu = 1
        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")


        # allow checkpointing via USR1
        # def melk(*args, **kwargs):
        #     # run all checkpoint hooks
        #     if trainer.global_rank == 0:
        #         print("Summoning checkpoint.")
        #         ckpt_path = os.path.join(ckptdir, "last.ckpt")
        #         trainer.save_checkpoint(ckpt_path)


        # def divein(*args, **kwargs):
        #     if trainer.global_rank == 0:
        #         import pudb;
        #         pudb.set_trace()


        # import signal

        # signal.signal(signal.SIGUSR1, melk)
        # signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                # melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            # trainer.test(model, data)
            # trainer.validate(model, data)
            pass
    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer.global_rank == 0:
            print(trainer.profiler.summary())

# conda activate hjy

# CUDA_VISIBLE_DEVICES=0 python main.py --base configs/latent-diffusion/celebahq-ldm-vq-f4.yaml --train --gpus 0,
# CUDA_VISIBLE_DEVICES=0 python main.py --base configs/latent-diffusion/kth-ldm-vq-f4.yaml --train --gpus 0,
# CUDA_VISIBLE_DEVICES=0,1 python main.py --base configs/latent-diffusion/kth-ldm-vq-f4.yaml --train --gpus 0,1

# [for training like]
# CUDA_VISIBLE_DEVICES=0,1 python main.py --base configs/latent-diffusion/kth-ldm-vq-f4.yaml --train --gpus 0,1
# CUDA_VISIBLE_DEVICES=0 python main.py --base configs/latent-diffusion/kth-ldm-vq-f4.yaml --train --gpus 0,
# python main.py --base configs/latent-diffusion/kth-ldm-vq-f4.yaml --train

# [for resume from a checkpoint like]
# CUDA_VISIBLE_DEVICES=0,1 python main.py --resume logs_training/20230220-213917_kth-ldm-vq-f4 --train --gpus 0,1

# [for evaluation like] wait for edit
# CUDA_VISIBLE_DEVICES=0,1 python main.py --resume logs_training/20230220-213917_kth-ldm-vq-f4 --gpus 0,1

# 主函数main.py
# 训练和推理进入到./ldm/models/diffusion/ddpm.py