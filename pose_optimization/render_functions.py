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
from ipdb import set_trace

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

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from mega_nerf.datasets.filesystem_dataset import FilesystemDataset
from mega_nerf.datasets.memory_dataset import MemoryDataset
from mega_nerf.image_metadata import ImageMetadata
from mega_nerf.metrics import psnr, ssim, lpips
from mega_nerf.misc_utils import main_print, main_tqdm
from mega_nerf.models.model_utils import get_nerf, get_bg_nerf
from mega_nerf.ray_utils import get_rays, get_ray_directions
from mega_nerf.rendering import render_rays
from mega_nerf.opts import get_opts_base

device = torch.device("cuda") # if torch.cuda.is_available() else "cpu")

class Render:
    def __init__(self, config: str, data_dir, hwf, K, container_path: str = None, set_experiment_path: bool = False):
        assert container_path is not None
        parser = get_opts_base()
        self.hwf = hwf
        self.K = K
        self.writer = None
        parser.add_argument('--exp_name', type=str, default='0', help='experiment name')
        parser.add_argument('--dataset_path', type=str, default=data_dir)
        
        hparams =  parser.parse_args(["--config_file", config])
        hparams.container_path = container_path

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

        self.is_local_master = ('RANK' not in os.environ) or int(os.environ['LOCAL_RANK']) == 0 
        main_print(hparams)

        if set_experiment_path:
            self.experiment_path = self._get_experiment_path() if self.is_master else None #
            self.model_path = self.experiment_path / 'models' if self.is_master else None

        self.writer = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        coordinate_info = torch.load(Path(hparams.dataset_path) / 'coordinates.pt', map_location='cpu') 
        self.origin_drb = coordinate_info['origin_drb']
        self.pose_scale_factor = coordinate_info['pose_scale_factor'] 
        main_print('Origin: {}, scale factor: {}'.format(self.origin_drb, self.pose_scale_factor))

        self.near = hparams.near / self.pose_scale_factor 

        if self.hparams.far is not None: 
            self.far = hparams.far / self.pose_scale_factor
        elif hparams.bg_nerf: 
            self.far = 1e5
        else:
            self.far = 2

        main_print('Ray bounds: {}, {}'.format(self.near, self.far))

        self.ray_altitude_range = [(x - self.origin_drb[0]) / self.pose_scale_factor for x in 
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

        self.train_items, self.val_items = self._get_image_metadata() 
        main_print('Using {} train images and {} val images'.format(len(self.train_items), len(self.val_items)))

        camera_positions = torch.cat([x.c2w[:3, 3].unsqueeze(0) for x in self.train_items + self.val_items]) # n * 3 float32
        min_position = camera_positions.min(dim=0)[0] 
        max_position = camera_positions.max(dim=0)[0]

        # main_print('Camera range in metric space: {} {}'.format(min_position * self.pose_scale_factor + self.origin_drb,
                                                                # max_position * self.pose_scale_factor + self.origin_drb))

        # main_print('Camera range in [-1, 1] space: {} {}'.format(min_position, max_position))

        self.nerf = get_nerf(hparams, len(self.train_items)).to(self.device) # len(self.train_items)=50(train+val)
        if 'RANK' in os.environ:
            self.nerf = torch.nn.parallel.DistributedDataParallel(self.nerf, device_ids=[int(os.environ['LOCAL_RANK'])],
                                                                  output_device=int(os.environ['LOCAL_RANK']))

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
                    air_poses[:, 0] = self.ray_altitude_range[0] 
                    used_positions = torch.cat([camera_positions, air_poses, ground_poses]) # 3*n 3
                else:
                    used_positions = camera_positions

                max_position[0] = self.ray_altitude_range[1] 
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


    def eval(self):
        val_metrics = self._run_validation(0)
        self._write_final_metrics(val_metrics)


    def _setup_experiment_dir(self) -> None:
        if self.is_master:
            self.experiment_path.mkdir() 
            with (self.experiment_path / 'hparams.txt').open('w') as f:
                for key in vars(self.hparams):
                    f.write('{}: {}\n'.format(key, vars(self.hparams)[key])) 
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


    def _run_validation(self, train_index: int) -> Dict[str, float]:
        with torch.inference_mode():
            self.nerf.eval()

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

                for i in main_tqdm(indices_to_eval): 
                    metadata_item = self.val_items[i]  # class ImageMetadata
                    viz_rgbs = metadata_item.load_image().float() / 255. # Tensor.float32 0~1
                    
                    results, _ = self.render_image(metadata_item) 
                    typ = 'fine' if 'rgb_fine' in results else 'coarse' # fine
                    viz_result_rgbs = results[f'rgb_{typ}'].view(*viz_rgbs.shape).cpu() # [H, W, 3] 0~1

                    eval_rgbs = viz_rgbs[:, viz_rgbs.shape[1] // 2:].contiguous() # H * W/2
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


    # position, dire、near、far
    def _get_rays_inner(rays_o: torch.Tensor, rays_d: torch.Tensor, near: float, far: float,
                        ray_altitude_range: List[float]) -> torch.Tensor:
        # c2w is drb, ray_altitude_range is max_altitude (neg), min_altitude (neg)
        near_bounds = near * torch.ones_like(rays_o[..., :1])
        far_bounds = far * torch.ones_like(rays_o[..., :1])

        if ray_altitude_range is not None:
            _truncate_with_plane_intersection(rays_o, rays_d, ray_altitude_range[0], near_bounds)
            near_bounds = torch.clamp(near_bounds, min=near)
            _truncate_with_plane_intersection(rays_o, rays_d, ray_altitude_range[1], far_bounds)

            far_bounds = torch.clamp(far_bounds, max=far)
            far_bounds = torch.maximum(near_bounds, far_bounds)

        return torch.cat([rays_o,
                        rays_d,
                        near_bounds,
                        far_bounds],
                        -1)  # (h, w, 8)


    def get_rays(directions: torch.Tensor, c2w: torch.Tensor, near: float, far: float,
                ray_altitude_range: List[float]) -> torch.Tensor:
        # Rotate ray directions from camera coordinate to the world coordinate
        rays_d = directions @ c2w[:, :3].T  # (H, W, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        # The origin of all rays is the camera origin in world coordinate
        rays_o = c2w[:, 3].expand(rays_d.shape)  # (H, W, 3) 把当前图像的世界系位置拓展到H*W

        return _get_rays_inner(rays_o, rays_d, near, far, ray_altitude_range)


    def render_image(self, batch, sim_pose, HW=False):
        
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        x = torch.Tensor(batch[:,0])
        y = torch.Tensor(batch[:,1])
        i, j = torch.meshgrid(x, y, indexing="xy") # pixel coord
        # if center_pixels: 
        #     i = i.clone() + 0.5
        #     j = j.clone() + 0.5
        # set_trace()
        # directions维度为(batch_size, batch_size, 3) 可以看成三页，第一页放的是x坐标 第二页放的是y坐标 第三页是z 对角线上才是batch_size个射线
        directions = \
            torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1).to(device) # 相机坐标系，最后一维为-1
        directions /= torch.linalg.norm(directions, dim=-1, keepdim=True) # 张量的第三维求二范数，保证每个向量模长为1
        
        # with torch.cuda.amp.autocast(enabled=self.hparams.amp):
        rays = get_rays(directions, sim_pose, self.near, self.far, self.ray_altitude_range) # H, W, 8
        rays = torch.diagonal(rays).T # 只取出对角元素
        rays = rays.view(-1, 8).to(self.device, non_blocking=True)  # (H*W, 8)
        image_indices = 0 * torch.ones(rays.shape[0], device=rays.device) \
            if self.hparams.appearance_dim > 0 else None # torch.size([H*W]) 
        results = {}

        if 'RANK' in os.environ:
            nerf = self.nerf.module
        else:
            nerf = self.nerf

        if self.bg_nerf is not None and 'RANK' in os.environ: # skip
            bg_nerf = self.bg_nerf.module
        else:
            bg_nerf = self.bg_nerf
        # set_trace()
        for i in range(0, rays.shape[0], self.hparams.image_pixel_batch_size):
            # set_trace()
            result_batch, _ = render_rays(nerf=nerf, bg_nerf=bg_nerf,
                                            rays=rays[i:i + self.hparams.image_pixel_batch_size],
                                            image_indices=image_indices[
                                                            i:i + self.hparams.image_pixel_batch_size] if self.hparams.appearance_dim > 0 else None,
                                            hparams=self.hparams,
                                            sphere_center=self.sphere_center,
                                            sphere_radius=self.sphere_radius,
                                            get_depth=False,
                                            get_depth_variance=False,
                                            get_bg_fg_rgb=True) # results是一个dict，包含7层

            for key, value in result_batch.items():
                if key not in results: # 对于第一个batch，首先创建一个key 后续则依次append
                    results[key] = []

                results[key].append(value) # slow step
        # 将每个batch的cat成一个大向量，results每个key对应的value是一个H*W维的向量
        for key, value in results.items():
            results[key] = torch.cat(value)

        return results, rays


    def get_img_from_pix(self, batch, sim_pose, HW=False):
        '''
        Input: 
            batch:
            sim_pose:
        Output:

        '''
        results, _ = self.render_image(batch, sim_pose, HW=False) # 
        typ = 'fine' if 'rgb_fine' in results else 'coarse' # fine
        viz_result_rgbs = results[f'rgb_{typ}'] #.view((self.batch_size, 3)) # 将rgb_fine转为[H, W, 3] 0~1
        return viz_result_rgbs


    def get_img_from_pose(self, sim_pose):
        H = self.hwf[0]
        W = self.hwf[1]
        intrinsics_0 = self.K[0, 0]
        intrinsics_1 = self.K[1, 1]
        intrinsics_2 = self.K[0, 2]
        intrinsics_3 = self.K[1, 2]
        directions = get_ray_directions(W, # 返回图像上每一个像素点在相机坐标系下的单位方向矢量(H W 3)
                                        H,
                                        intrinsics_0,
                                        intrinsics_1,
                                        intrinsics_2,
                                        intrinsics_3,
                                        self.hparams.center_pixels,
                                        device)
        rays = get_rays(directions, sim_pose, self.near, self.far, self.ray_altitude_range) # H, W, 8
        # 上式中：c2w为3*4 directions为H*W*3
        rays = rays.view(-1, 8).to(self.device, non_blocking=True)  # (H*W, 8) 这里把rays归并成n*8的张量，也就是说假设我们不想渲染整个图，只要最后的rasy是batch_size*8大小就行
        
        image_indices = 0 * torch.ones(rays.shape[0], device=rays.device) \
            if self.hparams.appearance_dim > 0 else None # torch.size([H*W]) 
        results = {}

        if 'RANK' in os.environ:
            nerf = self.nerf.module
        else:
            nerf = self.nerf

        if self.bg_nerf is not None and 'RANK' in os.environ: # skip
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
                                            get_bg_fg_rgb=True) # results是一个dict，包含7层

            for key, value in result_batch.items():
                if key not in results: # 对于第一个batch，首先创建一个key 后续则依次append
                    results[key] = []

                results[key].append(value) # slow step
        # 将每个batch的cat成一个大向量，results每个key对应的value是一个H*W维的向量
        for key, value in results.items():
            results[key] = torch.cat(value)

        typ = 'fine' if 'rgb_fine' in results else 'coarse' # fine
        viz_result_rgbs = results[f'rgb_{typ}']
        
        return viz_result_rgbs


    def _get_experiment_path(self) -> Path:
        exp_dir = Path(self.hparams.exp_name)
        exp_dir.mkdir(parents=True, exist_ok=True)
        existing_versions = [int(x.name) for x in exp_dir.iterdir()]
        version = 0 if len(existing_versions) == 0 else max(existing_versions) + 1
        experiment_path = exp_dir / str(version)
        return experiment_path    


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
