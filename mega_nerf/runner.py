import datetime
import faulthandler
import math
import os
import random
import shutil
import signal
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Union

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DistributedSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm

from mega_nerf.datasets.filesystem_dataset import FilesystemDataset
from mega_nerf.datasets.memory_dataset import MemoryDataset
from image_metadata import ImageMetadata
from metrics import psnr, ssim, lpips
from misc_utils import main_print, main_tqdm
from models.model_utils import get_nerf, get_bg_nerf
from ray_utils import get_rays, get_ray_directions, get_rays_od
from rendering import render_rays



class Runner:
    def __init__(self, hparams: Namespace, set_experiment_path: bool = True):
        faulthandler.register(signal.SIGUSR1)

        if hparams.ckpt_path is not None:
            # checkpoint = torch.load(hparams.ckpt_path, map_location='cpu')
            # np.random.set_state(checkpoint['np_random_state'])
            # torch.set_rng_state(checkpoint['torch_random_state'])
            # random.setstate(checkpoint['random_state'])
            pass
        else:
            np.random.seed(hparams.random_seed)
            torch.manual_seed(hparams.random_seed)
            random.seed(hparams.random_seed)

        self.hparams = hparams

        if 'RANK' in os.environ:
            dist.init_process_group(backend='nccl', timeout=datetime.timedelta(0, hours=24))
            torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
            self.is_master = (int(os.environ['RANK']) == 0)
        else:
            self.is_master = True # True

        self.is_local_master = ('RANK' not in os.environ) or int(os.environ['LOCAL_RANK']) == 0 # True
        main_print(hparams)

        if set_experiment_path:
            self.experiment_path = self._get_experiment_path() if self.is_master else None #
            self.model_path = self.experiment_path / 'models' if self.is_master else None

        self.writer = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        coordinate_info = torch.load(Path(hparams.dataset_path) / 'coordinates.pt', map_location='cpu') # 导入数据集原点tensor (3) float32
        self.origin_drb = coordinate_info['origin_drb']
        self.pose_scale_factor = coordinate_info['pose_scale_factor'] # 225
        main_print('Origin: {}, scale factor: {}'.format(self.origin_drb, self.pose_scale_factor))

        self.near = hparams.near / self.pose_scale_factor # hparams.near=1默认值，尺度因子为数据集的coordinates.pt中获取

        if self.hparams.far is not None: # skip
            self.far = hparams.far / self.pose_scale_factor
        elif hparams.bg_nerf: # true
            self.far = 1e5
        else:
            self.far = 2

        main_print('Ray bounds: {}, {}'.format(self.near, self.far))

        self.ray_altitude_range = [(x - self.origin_drb[0]) / self.pose_scale_factor for x in # 
                                   hparams.ray_altitude_range] if hparams.ray_altitude_range is not None else None
        main_print('Ray altitude range in [-1, 1] space: {}'.format(self.ray_altitude_range))
        main_print('Ray altitude range in metric space: {}'.format(hparams.ray_altitude_range))

        if self.ray_altitude_range is not None:
            assert self.ray_altitude_range[0] < self.ray_altitude_range[1]

        if self.hparams.cluster_mask_path is not None: # skip
            cluster_params = torch.load(Path(self.hparams.cluster_mask_path).parent / 'params.pt', map_location='cpu')
            assert cluster_params['near'] == self.near
            assert (torch.allclose(cluster_params['origin_drb'], self.origin_drb))
            assert cluster_params['pose_scale_factor'] == self.pose_scale_factor

            if self.ray_altitude_range is not None:
                assert (torch.allclose(torch.FloatTensor(cluster_params['ray_altitude_range']),
                                       torch.FloatTensor(self.ray_altitude_range))), \
                    '{} {}'.format(self.ray_altitude_range, cluster_params['ray_altitude_range'])

        self.train_items, self.val_items = self._get_image_metadata() # 获取图像数据存放位置、位姿、内参、宽高等
        main_print('Using {} train images and {} val images'.format(len(self.train_items), len(self.val_items)))

        camera_positions = torch.cat([x.c2w[:3, 3].unsqueeze(0) for x in self.train_items + self.val_items]) # n * 3 float32
        min_position = camera_positions.min(dim=0)[0]
        max_position = camera_positions.max(dim=0)[0]

        main_print('Camera range in metric space: {} {}'.format(min_position * self.pose_scale_factor + self.origin_drb,
                                                                max_position * self.pose_scale_factor + self.origin_drb))

        main_print('Camera range in [-1, 1] space: {} {}'.format(min_position, max_position))
        # a. 初始化前景 nerf
        self.nerf = get_nerf(hparams, len(self.train_items)).to(self.device) # len(self.train_items)=50(train+val)
        if 'RANK' in os.environ:
            self.nerf = torch.nn.parallel.DistributedDataParallel(self.nerf, device_ids=[int(os.environ['LOCAL_RANK'])],
                                                                  output_device=int(os.environ['LOCAL_RANK']))
        # b. 初始化背景 nerf
        if hparams.bg_nerf: 
            self.bg_nerf = get_bg_nerf(hparams, len(self.train_items)).to(self.device)
            if 'RANK' in os.environ:
                self.bg_nerf = torch.nn.parallel.DistributedDataParallel(self.bg_nerf,
                                                                         device_ids=[int(os.environ['LOCAL_RANK'])],
                                                                         output_device=int(os.environ['LOCAL_RANK']))

            if hparams.ellipse_bounds:
                assert hparams.ray_altitude_range is not None

                if self.ray_altitude_range is not None:
                    ground_poses = camera_positions.clone()
                    ground_poses[:, 0] = self.ray_altitude_range[1]
                    air_poses = camera_positions.clone()
                    air_poses[:, 0] = self.ray_altitude_range[0] # 把第一维全部赋值成altitude最小值
                    used_positions = torch.cat([camera_positions, air_poses, ground_poses]) # 3*n 3
                else:
                    used_positions = camera_positions

                max_position[0] = self.ray_altitude_range[1] # max_position是所有相机位姿里第一维最大的向量，把第一维重新赋值
                main_print('Camera range in [-1, 1] space with ray altitude range: {} {}'.format(min_position,
                                                                                                 max_position))

                self.sphere_center = ((max_position + min_position) * 0.5).to(self.device)
                self.sphere_radius = ((max_position - min_position) * 0.5).to(self.device)
                scale_factor = ((used_positions.to(self.device) - self.sphere_center) / self.sphere_radius).norm(
                    dim=-1).max()

                self.sphere_radius *= (scale_factor * hparams.ellipse_scale_factor)
            else:
                self.sphere_center = None
                self.sphere_radius = None

            main_print('Sphere center: {}, radius: {}'.format(self.sphere_center, self.sphere_radius))
        else:
            self.bg_nerf = None
            self.sphere_center = None
            self.sphere_radius = None

    def train(self):
        self._setup_experiment_dir()

        scaler = torch.cuda.amp.GradScaler(enabled=self.hparams.amp)

        optimizers = {}
        optimizers['nerf'] = Adam(self.nerf.parameters(), lr=self.hparams.lr)
        if self.bg_nerf is not None:
            optimizers['bg_nerf'] = Adam(self.bg_nerf.parameters(), lr=self.hparams.lr)

        if self.hparams.ckpt_path is not None:
            checkpoint = torch.load(self.hparams.ckpt_path, map_location='cpu')
            train_iterations = checkpoint['iteration']

            scaler_dict = scaler.state_dict()
            scaler_dict.update(checkpoint['scaler'])
            scaler.load_state_dict(scaler_dict)

            for key, optimizer in optimizers.items():
                optimizer_dict = optimizer.state_dict()
                optimizer_dict.update(checkpoint['optimizers'][key])
                optimizer.load_state_dict(optimizer_dict)
            discard_index = checkpoint['dataset_index'] if self.hparams.resume_ckpt_state else -1
        else:
            train_iterations = 0
            discard_index = -1

        schedulers = {}
        for key, optimizer in optimizers.items(): # 指数衰减调整学习率
            schedulers[key] = ExponentialLR(optimizer,
                                            gamma=self.hparams.lr_decay_factor ** (1 / self.hparams.train_iterations),
                                            last_epoch=train_iterations - 1)
        # 从文件中 或 内存 读取训练数据
        if self.hparams.dataset_type == 'filesystem':
            # Let the local master write data to disk first
            # We could further parallelize the disk writing process by having all of the ranks write data,
            # but it would make determinism trickier
            if 'RANK' in os.environ and (not self.is_local_master):
                dist.barrier()

            dataset = FilesystemDataset(self.train_items, self.near, self.far, self.ray_altitude_range,
                                        self.hparams.center_pixels, self.device,
                                        [Path(x) for x in sorted(self.hparams.chunk_paths)], self.hparams.num_chunks,
                                        self.hparams.train_scale_factor, self.hparams.disk_flush_size)
            if self.hparams.ckpt_path is not None and self.hparams.resume_ckpt_state:
                dataset.set_state(checkpoint['dataset_state'])
            if 'RANK' in os.environ and self.is_local_master:
                dist.barrier()
        elif self.hparams.dataset_type == 'memory':
            dataset = MemoryDataset(self.train_items, self.near, self.far, self.ray_altitude_range,
                                    self.hparams.center_pixels, self.device)
        else:
            raise Exception('Unrecognized dataset type: {}'.format(self.hparams.dataset_type))

        if self.is_master:
            pbar = tqdm(total=self.hparams.train_iterations)
            pbar.update(train_iterations)
        else:
            pbar = None

        while train_iterations < self.hparams.train_iterations:
            # If discard_index >= 0, we already set to the right chunk through set_state
            if self.hparams.dataset_type == 'filesystem' and discard_index == -1:
                dataset.load_chunk()

            if 'RANK' in os.environ:
                world_size = int(os.environ['WORLD_SIZE'])
                sampler = DistributedSampler(dataset, world_size, int(os.environ['RANK']))
                assert self.hparams.batch_size % world_size == 0
                data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size // world_size, sampler=sampler,
                                         num_workers=0, pin_memory=True)
            else:
                data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=0,
                                         pin_memory=True)
            # BARF策略
            self.nerf.embedding_xyz.progress.data.fill_(1.0) # train_iterations / self.hparams.train_iterations
            self.nerf.embedding_dir.progress.data.fill_(1.0) # train_iterations / self.hparams.train_iterations

            for dataset_index, item in enumerate(data_loader):
                if dataset_index <= discard_index:
                    continue

                discard_index = -1
                # 正向传播
                with torch.cuda.amp.autocast(enabled=self.hparams.amp):
                    if self.hparams.appearance_dim > 0:
                        image_indices = item['img_indices'].to(self.device, non_blocking=True)
                    else:
                        image_indices = None
                    # 计算loss
                    metrics, bg_nerf_rays_present = self._training_step(
                        item['rgbs'].to(self.device, non_blocking=True),
                        item['rays'].to(self.device, non_blocking=True),
                        image_indices)

                    with torch.no_grad():
                        for key, val in metrics.items():
                            if key == 'psnr' and math.isinf(val):  # a perfect reproduction will give PSNR = infinity
                                continue

                            if not math.isfinite(val):
                                raise Exception('Train metrics not finite: {}'.format(metrics))
                # 归零梯度
                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
                # 反向传播
                scaler.scale(metrics['loss']).backward()

                for key, optimizer in optimizers.items():
                    if key == 'bg_nerf' and (not bg_nerf_rays_present):
                        continue
                    else:
                        scaler.step(optimizer)

                scaler.update()
                # 更新学习率
                for scheduler in schedulers.values():
                    scheduler.step()

                train_iterations += 1
                if self.is_master:
                    pbar.update(1)
                    for key, value in metrics.items():
                        self.writer.add_scalar('train/{}'.format(key), value, train_iterations)

                    if train_iterations > 0 and train_iterations % self.hparams.ckpt_interval == 0:
                        self._save_checkpoint(optimizers, scaler, train_iterations, dataset_index,
                                              dataset.get_state() if self.hparams.dataset_type == 'filesystem' else None)

                if train_iterations > 0 and train_iterations % self.hparams.val_interval == 0:
                    self._run_validation(train_iterations)

                if train_iterations >= self.hparams.train_iterations:
                    break

        if 'RANK' in os.environ:
            dist.barrier()

        if self.is_master:
            pbar.close()
            self._save_checkpoint(optimizers, scaler, train_iterations, dataset_index,
                                  dataset.get_state() if self.hparams.dataset_type == 'filesystem' else None)

        if self.hparams.cluster_mask_path is None:
            val_metrics = self._run_validation(train_iterations)
            self._write_final_metrics(val_metrics)

    def eval(self):
        self._setup_experiment_dir() # 记录参数、命令、image_index等到.txt
        val_metrics = self._run_validation(0)
        self._write_final_metrics(val_metrics)

    def _write_final_metrics(self, val_metrics: Dict[str, float]) -> None:
        if self.is_master:
            with (self.experiment_path / 'metrics.txt').open('w') as f:
                for key in val_metrics:
                    avg_val = val_metrics[key] / len(self.val_items)
                    message = 'Average {}: {}'.format(key, avg_val)
                    main_print(message)
                    f.write('{}\n'.format(message))

            self.writer.flush()
            self.writer.close()

    def _setup_experiment_dir(self) -> None:
        if self.is_master:
            self.experiment_path.mkdir() # self.experiment_path 是pathlib.Path类的实例
            with (self.experiment_path / 'hparams.txt').open('w') as f:
                for key in vars(self.hparams):
                    f.write('{}: {}\n'.format(key, vars(self.hparams)[key])) # 写入参数表到hparams.txt
                if 'WORLD_SIZE' in os.environ:
                    f.write('WORLD_SIZE: {}\n'.format(os.environ['WORLD_SIZE']))

            with (self.experiment_path / 'command.txt').open('w') as f:
                f.write(' '.join(sys.argv))
                f.write('\n')

            self.model_path.mkdir(parents=True)

            with (self.experiment_path / 'image_indices.txt').open('w') as f:
                for i, metadata_item in enumerate(self.train_items):
                    f.write('{},{}\n'.format(metadata_item.image_index, metadata_item.image_path.name))
        self.writer = SummaryWriter(str(self.experiment_path / 'tb')) if self.is_master else None

        if 'RANK' in os.environ:
            dist.barrier()

    def _training_step(self, rgbs: torch.Tensor, rays: torch.Tensor, image_indices: Optional[torch.Tensor]) \
            -> Tuple[Dict[str, Union[torch.Tensor, float]], bool]:
        results, bg_nerf_rays_present = render_rays(nerf=self.nerf,
                                                    bg_nerf=self.bg_nerf,
                                                    rays=rays,
                                                    image_indices=image_indices,
                                                    hparams=self.hparams,
                                                    sphere_center=self.sphere_center,
                                                    sphere_radius=self.sphere_radius,
                                                    get_depth=False,
                                                    get_depth_variance=True,
                                                    get_bg_fg_rgb=False)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            depth_variance = results[f'depth_variance_{typ}'].mean()

        metrics = {
            'psnr': psnr_,
            'depth_variance': depth_variance,
        }

        photo_loss = F.mse_loss(results[f'rgb_{typ}'], rgbs, reduction='mean')
        metrics['photo_loss'] = photo_loss
        metrics['loss'] = photo_loss

        if self.hparams.use_cascade and typ != 'coarse':
            coarse_loss = F.mse_loss(results['rgb_coarse'], rgbs, reduction='mean')

            metrics['coarse_loss'] = coarse_loss
            metrics['loss'] += coarse_loss
            metrics['loss'] /= 2

        return metrics, bg_nerf_rays_present

    def _run_validation(self, train_index: int) -> Dict[str, float]:
        with torch.inference_mode():
            self.nerf.eval() # 调用nn.Modules.eval()

            val_metrics = defaultdict(float)
            base_tmp_path = None
            try:
                if 'RANK' in os.environ: # skip
                    base_tmp_path = Path(self.hparams.exp_name) / os.environ['TORCHELASTIC_RUN_ID']
                    metric_path = base_tmp_path / 'tmp_val_metrics'
                    image_path = base_tmp_path / 'tmp_val_images'

                    world_size = int(os.environ['WORLD_SIZE'])
                    indices_to_eval = np.arange(int(os.environ['RANK']), len(self.val_items), world_size)
                    if self.is_master:
                        base_tmp_path.mkdir()
                        metric_path.mkdir()
                        image_path.mkdir()
                    dist.barrier()
                else:
                    indices_to_eval = np.arange(len(self.val_items)) # length of val

                for i in main_tqdm(indices_to_eval): # 对于每一帧图像
                    metadata_item = self.val_items[i]  # class ImageMetadata
                    viz_rgbs = metadata_item.load_image().float() / 255. # Tensor.float32 0~1 加载图片
                    # results是一个dict，每个key对应的value是一个H*W的大向量
                    results, _ = self.render_image(metadata_item) # 获取渲染图像 消耗很长时间
                    typ = 'fine' if 'rgb_fine' in results else 'coarse' # fine
                    viz_result_rgbs = results[f'rgb_{typ}'].view(*viz_rgbs.shape).cpu() # 将rgb_fine转为[H, W, 3] 0~1

                    eval_rgbs = viz_rgbs[:, viz_rgbs.shape[1] // 2:].contiguous() # 取出后一半 H * W/2
                    eval_result_rgbs = viz_result_rgbs[:, viz_rgbs.shape[1] // 2:].contiguous()

                    val_psnr = psnr(eval_result_rgbs.view(-1, 3), eval_rgbs.view(-1, 3))

                    metric_key = 'val/psnr/{}'.format(i)
                    if self.writer is not None:
                        self.writer.add_scalar(metric_key, val_psnr, train_index)
                    else:
                        torch.save({'value': val_psnr, 'metric_key': metric_key, 'agg_key': 'val/psnr'},
                                   metric_path / 'psnr-{}.pt'.format(i))

                    val_metrics['val/psnr'] += val_psnr

                    val_ssim = ssim(eval_result_rgbs.view(*eval_rgbs.shape), eval_rgbs, 1)

                    metric_key = 'val/ssim/{}'.format(i)
                    if self.writer is not None:
                        self.writer.add_scalar(metric_key, val_ssim, train_index)
                    else:
                        torch.save({'value': val_ssim, 'metric_key': metric_key, 'agg_key': 'val/ssim'},
                                   metric_path / 'ssim-{}.pt'.format(i))

                    val_metrics['val/ssim'] += val_ssim

                    val_lpips_metrics = lpips(eval_result_rgbs.view(*eval_rgbs.shape), eval_rgbs)

                    for network in val_lpips_metrics:
                        agg_key = 'val/lpips/{}'.format(network)
                        metric_key = '{}/{}'.format(agg_key, i)
                        if self.writer is not None:
                            self.writer.add_scalar(metric_key, val_lpips_metrics[network], train_index)
                        else:
                            torch.save(
                                {'value': val_lpips_metrics[network], 'metric_key': metric_key, 'agg_key': agg_key},
                                metric_path / 'lpips-{}-{}.pt'.format(network, i))

                        val_metrics[agg_key] += val_lpips_metrics[network]

                    viz_result_rgbs = viz_result_rgbs.view(viz_rgbs.shape[0], viz_rgbs.shape[1], 3).cpu()
                    viz_depth = results[f'depth_{typ}']
                    if f'fg_depth_{typ}' in results:
                        to_use = results[f'fg_depth_{typ}'].view(-1)
                        while to_use.shape[0] > 2 ** 24:
                            to_use = to_use[::2]
                        ma = torch.quantile(to_use, 0.95)

                        viz_depth = viz_depth.clamp_max(ma)

                    img = Runner._create_result_image(viz_rgbs, viz_result_rgbs, viz_depth)
                    img.save(str("test/" + '{}.jpg'.format(i))) # GT render depth
                    if self.writer is not None:
                        self.writer.add_image('val/{}'.format(i), T.ToTensor()(img), train_index)
                    else:
                        img.save(str(image_path / '{}.jpg'.format(i)))

                    if self.hparams.bg_nerf:
                        if f'bg_rgb_{typ}' in results:
                            img = Runner._create_result_image(viz_rgbs,
                                                              results[f'bg_rgb_{typ}'].view(viz_rgbs.shape[0],
                                                                                            viz_rgbs.shape[1],
                                                                                            3).cpu(),
                                                              results[f'bg_depth_{typ}'])

                            if self.writer is not None:
                                self.writer.add_image('val/{}_bg'.format(i), T.ToTensor()(img), train_index)
                            else:
                                img.save(str(image_path / '{}_bg.jpg'.format(i)))

                            img = Runner._create_result_image(viz_rgbs,
                                                              results[f'fg_rgb_{typ}'].view(viz_rgbs.shape[0],
                                                                                            viz_rgbs.shape[1],
                                                                                            3).cpu(),
                                                              results[f'fg_depth_{typ}'])

                            if self.writer is not None:
                                self.writer.add_image('val/{}_fg'.format(i), T.ToTensor()(img), train_index)
                            else:
                                img.save(str(image_path / '{}_fg.jpg'.format(i)))

                    del results

                if 'RANK' in os.environ:
                    dist.barrier()
                    if self.writer is not None:
                        for metric_file in metric_path.iterdir():
                            metric = torch.load(metric_file, map_location='cpu')
                            self.writer.add_scalar(metric['metric_key'], metric['value'], train_index)
                            val_metrics[metric['agg_key']] += metric['value']
                        for image_file in image_path.iterdir():
                            img = Image.open(str(image_file))
                            self.writer.add_image('val/{}'.format(image_file.stem), T.ToTensor()(img), train_index)

                        for key in val_metrics:
                            avg_val = val_metrics[key] / len(self.val_items)
                            self.writer.add_scalar('{}/avg'.format(key), avg_val, 0)

                    dist.barrier()

                self.nerf.train()
            finally:
                if self.is_master and base_tmp_path is not None:
                    shutil.rmtree(base_tmp_path)

            return val_metrics

    def _save_checkpoint(self, optimizers: Dict[str, any], scaler: GradScaler, train_index: int, dataset_index: int,
                         dataset_state: Optional[str]) -> None:
        dict = {
            'model_state_dict': self.nerf.state_dict(),
            'scaler': scaler.state_dict(),
            'optimizers': {k: v.state_dict() for k, v in optimizers.items()},
            'iteration': train_index,
            'torch_random_state': torch.get_rng_state(),
            'np_random_state': np.random.get_state(),
            'random_state': random.getstate(),
            'dataset_index': dataset_index
        }

        if dataset_state is not None:
            dict['dataset_state'] = dataset_state

        if self.bg_nerf is not None:
            dict['bg_model_state_dict'] = self.bg_nerf.state_dict()

        torch.save(dict, self.model_path / '{}.pt'.format(train_index))
    # 渲染一帧 metadata包含一帧的信息
    def render_image(self, metadata: ImageMetadata) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        directions = get_ray_directions(metadata.W, # 返回图像上每一个像素点在相机坐标系下的单位方向矢量(H W 3)
                                        metadata.H,
                                        metadata.intrinsics[0],
                                        metadata.intrinsics[1],
                                        metadata.intrinsics[2],
                                        metadata.intrinsics[3],
                                        self.hparams.center_pixels,
                                        self.device)
        # directions为H*W*3 对于索引[i,j]其值含义为像素坐标为
        with torch.cuda.amp.autocast(enabled=self.hparams.amp):
            rays = get_rays(directions, metadata.c2w.to(self.device), self.near, self.far, self.ray_altitude_range) # H, W, 8
            # 上式中：c2w为3*4 directions为H*W*3
            rays = rays.view(-1, 8).to(self.device, non_blocking=True)  # (H*W, 8) 这里把rays归并成n*8的张量，也就是说假设我们不想渲染整个图，只要最后的rasy是batch_size*8大小就行
            image_indices = metadata.image_index * torch.ones(rays.shape[0], device=rays.device) \
                if self.hparams.appearance_dim > 0 else None # torch.size([H*W])
            results = {}

            if 'RANK' in os.environ: # 跳过
                nerf = self.nerf.module
            else:
                nerf = self.nerf

            if self.bg_nerf is not None and 'RANK' in os.environ: # skip
                bg_nerf = self.bg_nerf.module
            else:
                bg_nerf = self.bg_nerf
            # 对于所有的rays，分成多个batch（这里为65536），将求得结果放入resultes中 该过程耗时很长
            for i in range(0, rays.shape[0], self.hparams.image_pixel_batch_size): # 从0开始每隔image_pixel_batch_size选一次
                result_batch, _ = render_rays(nerf=nerf, bg_nerf=bg_nerf,
                                              rays=rays[i:i + self.hparams.image_pixel_batch_size],
                                              image_indices=image_indices[
                                                            i:i + self.hparams.image_pixel_batch_size] if self.hparams.appearance_dim > 0 else None,
                                              hparams=self.hparams,
                                              sphere_center=self.sphere_center,
                                              sphere_radius=self.sphere_radius,
                                              get_depth=True,
                                              get_depth_variance=False,
                                              get_bg_fg_rgb=True) # results是一个dict，包含7层

                for key, value in result_batch.items():
                    if key not in results: # 对于第一个batch，首先创建一个key 后续则依次append
                        results[key] = []

                    results[key].append(value.cpu()) # slow step
            # 将每个batch的cat成一个大向量，results每个key对应的value是一个H*W维的向量
            for key, value in results.items():
                results[key] = torch.cat(value)

            return results, rays

    @staticmethod
    def _create_result_image(rgbs: torch.Tensor, result_rgbs: torch.Tensor, result_depths: torch.Tensor) -> Image:
        depth_vis = Runner.visualize_scalars(torch.log(result_depths + 1e-8).view(rgbs.shape[0], rgbs.shape[1]).cpu())
        images = (rgbs * 255, result_rgbs * 255, depth_vis)
        return Image.fromarray(np.concatenate(images, 1).astype(np.uint8))

    @staticmethod
    def visualize_scalars(scalar_tensor: torch.Tensor) -> np.ndarray:
        to_use = scalar_tensor.view(-1)
        while to_use.shape[0] > 2 ** 24:
            to_use = to_use[::2]

        mi = torch.quantile(to_use, 0.05)
        ma = torch.quantile(to_use, 0.95)

        scalar_tensor = (scalar_tensor - mi) / max(ma - mi, 1e-8)  # normalize to 0~1
        scalar_tensor = scalar_tensor.clamp_(0, 1)

        scalar_tensor = ((1 - scalar_tensor) * 255).byte().numpy()  # inverse heatmap
        return cv2.cvtColor(cv2.applyColorMap(scalar_tensor, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)

    def _get_image_metadata(self) -> Tuple[List[ImageMetadata], List[ImageMetadata]]:
        dataset_path = Path(self.hparams.dataset_path)

        train_path_candidates = sorted(list((dataset_path / 'train' / 'metadata').iterdir()))
        train_paths = [train_path_candidates[i] for i in
                       range(0, len(train_path_candidates), self.hparams.train_every)]

        val_paths = sorted(list((dataset_path / 'val' / 'metadata').iterdir()))
        train_paths += val_paths
        train_paths.sort(key=lambda x: x.name) # val和train中顺序是随机的，进行一次排序
        val_paths_set = set(val_paths)

        image_indices = {}
        for i, train_path in enumerate(train_paths):
            image_indices[train_path.name] = i

        train_items = [
            self._get_metadata_item(x, image_indices[x.name], self.hparams.train_scale_factor, x in val_paths_set) for x
            in train_paths] # 这里train_items包含了val
        val_items = [
            self._get_metadata_item(x, image_indices[x.name], self.hparams.val_scale_factor, True) for x in val_paths]

        return train_items, val_items
        
    # metadata_path指向数据集中图片的.pt文件，包含图像位姿、内参、宽高
    def _get_metadata_item(self, metadata_path: Path, image_index: int, scale_factor: int,
                           is_val: bool) -> ImageMetadata:
        image_path = None
        for extension in ['.jpg', '.JPG', '.png', '.PNG']:
            candidate = metadata_path.parent.parent / 'rgbs' / '{}{}'.format(metadata_path.stem, extension)
            if candidate.exists():
                image_path = candidate
                break

        assert image_path.exists()

        metadata = torch.load(metadata_path, map_location='cpu')
        intrinsics = metadata['intrinsics'] / scale_factor
        assert metadata['W'] % scale_factor == 0
        assert metadata['H'] % scale_factor == 0

        dataset_mask = metadata_path.parent.parent.parent / 'masks' / metadata_path.name
        if self.hparams.cluster_mask_path is not None:
            if image_index == 0:
                main_print('Using cluster mask path: {}'.format(self.hparams.cluster_mask_path))
            mask_path = Path(self.hparams.cluster_mask_path) / metadata_path.name
        elif dataset_mask.exists():
            if image_index == 0:
                main_print('Using dataset mask path: {}'.format(dataset_mask.parent))
            mask_path = dataset_mask
        else:
            mask_path = None

        return ImageMetadata(image_path, metadata['c2w'], metadata['W'] // scale_factor, metadata['H'] // scale_factor,
                             intrinsics, image_index, None if (is_val and self.hparams.all_val) else mask_path, is_val)

    def _get_experiment_path(self) -> Path:
        exp_dir = Path(self.hparams.exp_name)
        exp_dir.mkdir(parents=True, exist_ok=True)
        existing_versions = [int(x.name) for x in exp_dir.iterdir()]
        version = 0 if len(existing_versions) == 0 else max(existing_versions) + 1
        experiment_path = exp_dir / str(version)
        return experiment_path

    def render_mega_nerf_imgs(self, args, dl, hwf, device):
        with torch.inference_mode():
            self.nerf.eval()
            H, W, focal = hwf
            target_list = []
            rgb_list = []
            pose_list = []
            directions = get_ray_directions(int(W // args.tinyscale), int(H // args.tinyscale), focal/args.tinyscale,
                                            focal/args.tinyscale, (W//args.tinyscale)/2, (H//args.tinyscale)/2,
                                            self.hparams.center_pixels, self.device)
            index = 0
            train_list = []
            for train_index in self.train_items:
                if train_index.is_val == False:
                    train_list.append(train_index.image_index)
            # random.shuffle(train_list)

            for batch_idx, (target, pose, img_idx) in enumerate(dl):
                if batch_idx % 10 == 0:
                    print("renders {}/total {}".format(batch_idx, len(dl.dataset)))

                target = target[0].permute(1, 2, 0).to(device)  # (240,360,3)
                # pose_mega_nerf = self.train_items[train_list[index]].c2w
                pose_mega_nerf = pose.view(3, 4)
                rays = get_rays(directions, pose_mega_nerf.to(self.device), self.near, self.far, self.ray_altitude_range)
                rays = rays.view(-1, 8).to(self.device, non_blocking=True)  # (H*W, 8)
                image_indices = train_list[index] * torch.ones(rays.shape[0], device=rays.device) \
                    if self.hparams.appearance_dim > 0 else None
                results = {}

                if 'RANK' in os.environ:
                    nerf = self.nerf.module
                else:
                    nerf = self.nerf

                if self.bg_nerf is not None and 'RANK' in os.environ:
                    bg_nerf = self.bg_nerf.module
                else:
                    bg_nerf = self.bg_nerf

                for i in range(0, rays.shape[0], self.hparams.image_pixel_batch_size):
                    result_batch, _ = render_rays(nerf=nerf, bg_nerf=bg_nerf,
                                                  rays=rays[i:i + self.hparams.image_pixel_batch_size],
                                                  image_indices=image_indices[
                                                            i:i + self.hparams.image_pixel_batch_size] if self.hparams.appearance_dim > 0 else None,
                                                  hparams=self.hparams,
                                                  sphere_center=self.sphere_center,
                                                  sphere_radius=self.sphere_radius,
                                                  get_depth=False,
                                                  get_depth_variance=False,
                                                  get_bg_fg_rgb=False)
                    for key, value in result_batch.items():
                        if key not in results:
                            results[key] = []
                        results[key].append(value.cpu())

                for key, value in results.items():
                    results[key] = torch.cat(value)

                typ = 'fine' if 'rgb_fine' in results else 'coarse'
                viz_result_rgbs = results[f'rgb_{typ}'].view(
                    torch.Size([int(H // args.tinyscale), int(W // args.tinyscale), 3])).cpu()
                # get rendered rgb(reshape to same size)
                rgb = viz_result_rgbs.contiguous()
                # generate nerf image
                # torch.set_default_tensor_type('torch.cuda.FloatTensor')
                # convert rgb to B,C,H,W
                rgb = rgb[None, ...].permute(0, 3, 1, 2)
                # upsample rgb to hwf size
                rgb = torch.nn.Upsample(size=(H, W), mode='bicubic')(rgb)
                # convert rgb back to H,W,C format
                rgb = rgb[0].permute(1, 2, 0)
                # # torch.set_default_tensor_type('torch.FloatTensor')

                target_list.append(target.cpu())
                rgb_list.append(rgb.cpu())
                pose_list.append(pose_mega_nerf.cpu())
                index+=1

            targets = torch.stack(target_list).detach()
            rgbs = torch.stack(rgb_list).detach()
            poses = torch.stack(pose_list).detach()
        return targets, rgbs, poses

    def render_virtual_meganerf_imgs(self, args, pose_perturb, hwf, device):
        ''' render mega-nerf images, save unscaled pose and results'''
        with torch.inference_mode():
            self.nerf.eval()
            H, W, focal = hwf
            rgb_list = []
            directions = get_ray_directions(int(W // args.tinyscale), int(H // args.tinyscale), focal/args.tinyscale,
                                            focal/args.tinyscale, (W//args.tinyscale)/2, (H//args.tinyscale)/2,
                                            self.hparams.center_pixels, self.device)
            index = 0
            train_list = []
            for train_index in self.train_items:
                if train_index.is_val == False:
                    train_list.append(train_index.image_index)
            # random.shuffle(train_list)

            for batch_idx in range(pose_perturb.shape[0]):
                if batch_idx % 10 == 0:
                    print("renders {} virtuals/total {}".format(batch_idx, pose_perturb.shape[0]))
                pose = pose_perturb[batch_idx]
                pose_nerf = pose.clone()
                rays = get_rays(directions, pose_nerf.to(device), self.near, self.far, self.ray_altitude_range)
                rays = rays.view(-1, 8).to(self.device, non_blocking=True)  # (H*W, 8)

                image_indices = train_list[index] * torch.ones(rays.shape[0], device=rays.device) \
                    if self.hparams.appearance_dim > 0 else None
                results = {}

                if 'RANK' in os.environ:
                    nerf = self.nerf.module
                else:
                    nerf = self.nerf

                if self.bg_nerf is not None and 'RANK' in os.environ:
                    bg_nerf = self.bg_nerf.module
                else:
                    bg_nerf = self.bg_nerf
                for i in range(0, rays.shape[0], self.hparams.image_pixel_batch_size):
                    result_batch, _ = render_rays(nerf=nerf, bg_nerf=bg_nerf,
                                                  rays=rays[i:i + self.hparams.image_pixel_batch_size],
                                                  image_indices=image_indices[
                                                            i:i + self.hparams.image_pixel_batch_size] if self.hparams.appearance_dim > 0 else None,
                                                  hparams=self.hparams,
                                                  sphere_center=self.sphere_center,
                                                  sphere_radius=self.sphere_radius,
                                                  get_depth=True,
                                                  get_depth_variance=False,
                                                  get_bg_fg_rgb=True)
                    for key, value in result_batch.items():
                        if key not in results:
                            results[key] = []
                        results[key].append(value.cpu())
                for key, value in results.items():
                    results[key] = torch.cat(value)

                typ = 'fine' if 'rgb_fine' in results else 'coarse'
                viz_result_rgbs = results[f'rgb_{typ}'].view(
                    torch.Size([int(H // args.tinyscale), int(W // args.tinyscale), 3])).cpu()
                # get rendered rgb(reshape to same size)
                rgb = viz_result_rgbs.contiguous()
                # generate nerf image
                # torch.set_default_tensor_type('torch.cuda.FloatTensor')
                # convert rgb to B,C,H,W
                rgb = rgb[None, ...].permute(0, 3, 1, 2)
                # upsample rgb to hwf size
                rgb = torch.nn.Upsample(size=(H, W), mode='bicubic')(rgb)
                # convert rgb back to H,W,C format
                rgb = rgb[0].permute(1, 2, 0)
                # torch.set_default_tensor_type('torch.FloatTensor')

                rgb_list.append(rgb.cpu())
                index+=1
            rgbs = torch.stack(rgb_list).detach()
        return rgbs


    def render_meganerf(self, H, W, focal, pose_img):
        with torch.inference_mode():
            self.nerf.eval()
            directions = get_ray_directions(W, H, focal,
                                            focal, W/2, H/2,
                                            self.hparams.center_pixels, self.device)
            pose_nerf = pose_img.clone()
            rays = get_rays(directions, pose_nerf.to(self.device), self.near, self.far, self.ray_altitude_range)
            rays = rays.view(-1, 8).to(self.device, non_blocking=True)  # (H*W, 8)
            image_indices = 5 * torch.ones(rays.shape[0], device=rays.device)
            results = {}
            if 'RANK' in os.environ:
                nerf = self.nerf.module
            else:
                nerf = self.nerf

            if self.bg_nerf is not None and 'RANK' in os.environ:
                bg_nerf = self.bg_nerf.module
            else:
                bg_nerf = self.bg_nerf
            for i in range(0, rays.shape[0], self.hparams.image_pixel_batch_size):
                result_batch, _ = render_rays(nerf=nerf, bg_nerf=bg_nerf,
                                              rays=rays[i:i + self.hparams.image_pixel_batch_size],
                                              image_indices=image_indices[
                                                    i:i + self.hparams.image_pixel_batch_size] if self.hparams.appearance_dim > 0 else None,
                                              hparams=self.hparams,
                                              sphere_center=self.sphere_center,
                                              sphere_radius=self.sphere_radius,
                                              get_depth=True,
                                              get_depth_variance=False,
                                              get_bg_fg_rgb=True)
                for key, value in result_batch.items():
                    if key not in results:
                        results[key] = []
                    results[key].append(value.cpu())
            for key, value in results.items():
                results[key] = torch.cat(value)

            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            viz_result_rgbs = results[f'rgb_{typ}'].view(torch.Size([H, W, 3])).cpu()
                # get rendered rgb(reshape to same size)
            rgb = viz_result_rgbs.contiguous()
        return rgb


    def render_megarays(self, H, W, focal, rays_in):
        with torch.inference_mode():
            self.nerf.eval()
            rays_o, rays_d = rays_in
            rays = get_rays_od(rays_o, rays_d, self.near, self.far, self.ray_altitude_range)
            rays = rays.view(-1, 8).to(self.device, non_blocking=True)  # (H*W, 8)
            image_indices = 5 * torch.ones(rays.shape[0], device=rays.device)
            results = {}
            if 'RANK' in os.environ:
                nerf = self.nerf.module
            else:
                nerf = self.nerf

            if self.bg_nerf is not None and 'RANK' in os.environ:
                bg_nerf = self.bg_nerf.module
            else:
                bg_nerf = self.bg_nerf
            for i in range(0, rays.shape[0], self.hparams.image_pixel_batch_size):
                result_batch, _ = render_rays(nerf=nerf, bg_nerf=bg_nerf,
                                              rays=rays[i:i + self.hparams.image_pixel_batch_size],
                                              image_indices=image_indices[
                                                    i:i + self.hparams.image_pixel_batch_size] if self.hparams.appearance_dim > 0 else None,
                                              hparams=self.hparams,
                                              sphere_center=self.sphere_center,
                                              sphere_radius=self.sphere_radius,
                                              get_depth=True,
                                              get_depth_variance=False,
                                              get_bg_fg_rgb=True)
                for key, value in result_batch.items():
                    if key not in results:
                        results[key] = []
                    results[key].append(value.cpu())
            for key, value in results.items():
                results[key] = torch.cat(value)

            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            viz_result_rgbs = results[f'rgb_{typ}']
                # get rendered rgb(reshape to same size)
            rgb = viz_result_rgbs.contiguous()
        return rgb