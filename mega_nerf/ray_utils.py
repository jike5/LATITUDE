from typing import List

import torch


def get_ray_directions(W: int, H: int, fx: float, fy: float, cx: float, cy: float, center_pixels: bool,
                       device: torch.device) -> torch.Tensor:
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32, device=device),  # i 是每一行相同 H*W 表示一个x
                          torch.arange(H, dtype=torch.float32, device=device), indexing='xy')  # j 是每一列相同 H*W 表示一个y
    if center_pixels:  # 挪到像素中心位置
        i = i.clone() + 0.5
        j = j.clone() + 0.5

    directions = \
        torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)],
                    -1)  # (H, W, 3) float32 此时的directions[i,j]表示图像上第i行(H方向) 第j列(W方向) 像素点对应的相机系下坐标 directions[0,0]=[-0.7721, 0.5789, -1] directions[1,0]=[-0.7721, 0.5776, -1] 返回的向量中对应的为(x, y, z)
    directions /= torch.linalg.norm(directions, dim=-1, keepdim=True)  # 张量的第三维求二范数，保证每个向量模长为1

    return directions  # float32 x y z 总之，要获取图像上i行(H) j列(W)处的像素点在相机坐标系的坐标，索引directions[i,j] = [x, y, z]


def get_rays(directions: torch.Tensor, c2w: torch.Tensor, near: float, far: float,
             ray_altitude_range: List[float]) -> torch.Tensor:
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T  # directions维度为(H, W, 3) c2w使用其3*4维度 得到rays_d维度为(H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)  # 获得世界系下光线法线方向

    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # 把当前图像的世界系位置拓展到batch_size*batch_size*3

    return _get_rays_inner(rays_o, rays_d, near, far, ray_altitude_range)  # rays_o rays_d均表示在世界系下(下 右 后)


def get_rays_batch(directions: torch.Tensor, c2w: torch.Tensor, near: float, far: float,
                   ray_altitude_range: List[float]) -> torch.Tensor:
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :, :3].transpose(1, 2)  # (n, H*W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, :, 3].unsqueeze(1).expand(rays_d.shape)  # (n, H*W, 3)

    return _get_rays_inner(rays_o, rays_d, near, far, ray_altitude_range)


# 返回位置、方向、near、far
def _get_rays_inner(rays_o: torch.Tensor, rays_d: torch.Tensor, near: float, far: float,
                    ray_altitude_range: List[float]) -> torch.Tensor:
    # c2w is drb, ray_altitude_range is max_altitude (neg), min_altitude (neg)
    near_bounds = near * torch.ones_like(rays_o[..., :1])  # 取出batch_size*batch_size*1大小 即第一页
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


def _truncate_with_plane_intersection(rays_o: torch.Tensor, rays_d: torch.Tensor, altitude: float,
                                      default_bounds: torch.Tensor) -> None:
    starts_before = rays_o[:, :, 0] < altitude  #
    goes_down = rays_d[:, :, 0] > 0
    boundable_rays = torch.minimum(starts_before, goes_down)

    ray_points = rays_o[boundable_rays]
    if ray_points.shape[0] == 0:
        return

    ray_directions = rays_d[boundable_rays]

    plane_normal = torch.FloatTensor([-1, 0, 0]).to(rays_o.device).unsqueeze(1)
    ndotu = ray_directions.mm(plane_normal)

    plane_point = torch.FloatTensor([altitude, 0, 0]).to(rays_o.device)
    w = ray_points - plane_point
    si = -w.mm(plane_normal) / ndotu
    plane_intersection = w + si * ray_directions + plane_point
    default_bounds[boundable_rays] = (ray_points - plane_intersection).norm(dim=-1).unsqueeze(1)


def get_rays_od(rays_o, rays_d, near: float, far: float,
                ray_altitude_range: List[float]) -> torch.Tensor:
    return _get_rays_inner(rays_o, rays_d, near, far, ray_altitude_range)
