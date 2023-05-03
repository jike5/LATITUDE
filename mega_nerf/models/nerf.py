from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class Embedding(nn.Module):
    def __init__(self, num_freqs: int, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        super(Embedding, self).__init__()

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, num_freqs - 1, num_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (num_freqs - 1), num_freqs)

        self.progress = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = [x]
        for freq in self.freq_bands:
            out += [torch.sin(freq * x), torch.cos(freq * x)]
        output = torch.cat(out, -1) # x_dim + 2*x_dim*L
        alpha = (self.progress.data + 0.7) * self.freq_bands.shape[0] # 初始参数
        k = torch.arange(self.freq_bands.shape[0], dtype=torch.float32).cuda()
        k = torch.stack([k for i in range(2 * x.shape[1])], 1).view(-1)
        weight = (1 - (alpha - k).clamp_(min=0, max=1).mul_(np.pi).cos_()) / 2 # [L]
        one = torch.ones(x.shape[1]).cuda()
        weight = torch.cat((one, weight), 0)

        return output * weight


class ShiftedSoftplus(nn.Module):
    __constants__ = ['beta', 'threshold']
    beta: int
    threshold: int

    def __init__(self, beta: int = 1, threshold: int = 20) -> None:
        super(ShiftedSoftplus, self).__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x - 1, self.beta, self.threshold)

    def extra_repr(self) -> str:
        return 'beta={}, threshold={}'.format(self.beta, self.threshold)


class GaussianActivation(nn.Module):
    def __init__(self, sigma = 0.05):
        super(GaussianActivation, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        return torch.exp(-0.5 / self.sigma**2 * x**2)

class NeRF(nn.Module):
    def __init__(self, pos_xyz_dim: int, pos_dir_dim: int, layers: int, skip_layers: List[int], layer_dim: int,
                 appearance_dim: int, affine_appearance: bool, appearance_count: int, rgb_dim: int, xyz_dim: int,
                 sigma_activation: nn.Module): # pos_xyz_dim,pos_dir_dim分别为xyz和dir的位置编码频率k; layers为MIL的总层数; skip_layers为中间引入位置编码的层; layer_dim为最终MLP的输出维度; appearance_dim为nerf in the wild中的描述光度的向量l(a); affine_appearance应该为图像的某种变换这里为空; appearance_count为appearance embedding的大小; rgb_dim为输出的rgb维度; xyz_dim为输入的位置的维度; sigma_activation为输出的sigma的激活函数
        super(NeRF, self).__init__() # 
        self.xyz_dim = xyz_dim # xyz_dim = 3

        if rgb_dim > 3: # rgb_dim = 3
            assert pos_dir_dim == 0

        self.embedding_xyz = Embedding(pos_xyz_dim) # 位置编码 pos_xyz_dim = 12 Embedding输入的参数为PE-coder中的k
        in_channels_xyz = xyz_dim + xyz_dim * pos_xyz_dim * 2 #  = 75

        self.skip_layers = skip_layers #  = [4]

        xyz_encodings = []

        # xyz encoding layers
        for i in range(layers): # layers = 8
            if i == 0:
                layer = nn.Linear(in_channels_xyz, layer_dim) # layer_dim = 256  nn.Linear(in, out)
            elif i in skip_layers:
                layer = nn.Linear(layer_dim + in_channels_xyz, layer_dim)
            else:
                layer = nn.Linear(layer_dim, layer_dim)
            # layer = nn.Sequential(layer, GaussianActivation(0.05)) # 这里修改nn.ReLU为高斯激活层 TODO: 参数配置文件
            layer = nn.Sequential(layer, nn.ReLU(True)) 
            xyz_encodings.append(layer)

        self.xyz_encodings = nn.ModuleList(xyz_encodings)

        if pos_dir_dim > 0: # pos_dir_dim = 4
            self.embedding_dir = Embedding(pos_dir_dim) # 方向dir的位置编码频率 k = pos_dir_dim
            in_channels_dir = 3 + 3 * pos_dir_dim * 2 # in_channels_dir = 27
        else:
            self.embedding_dir = None
            in_channels_dir = 0

        if appearance_dim > 0: # appearance_dim = 48 appearance_count=50
            self.embedding_a = nn.Embedding(appearance_count, appearance_dim) # 词嵌入字典大小, 每个词嵌入向量的大小
        else:
            self.embedding_a = None

        if affine_appearance: # affine_appearance = False
            assert appearance_dim > 0
            self.affine = nn.Linear(appearance_dim, 12)
        else:
            self.affine = None

        if pos_dir_dim > 0 or (appearance_dim > 0 and not affine_appearance):
            self.xyz_encoding_final = nn.Linear(layer_dim, layer_dim)
            # direction and appearance encoding layers
            self.dir_a_encoding = nn.Sequential(
                nn.Linear(layer_dim + in_channels_dir + (appearance_dim if not affine_appearance else 0), # = 256 + 27 + 48 = 331
                          layer_dim // 2), # = 128
                nn.ReLU(True))
        else:
            self.xyz_encoding_final = None

        # output layers
        self.sigma = nn.Linear(layer_dim, 1)
        self.sigma_activation = sigma_activation

        self.rgb = nn.Linear(
            layer_dim // 2 if (pos_dir_dim > 0 or (appearance_dim > 0 and not affine_appearance)) else layer_dim, # = 128
            rgb_dim) # rgb_dim =3
        if rgb_dim == 3:
            self.rgb_activation = nn.Sigmoid()  # = nn.Sequential(rgb, nn.Sigmoid())
        else:
            self.rgb_activation = None  # We're using spherical harmonics and will convert to sigmoid in rendering.py

    def forward(self, x: torch.Tensor, sigma_only: bool = False,
                sigma_noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        expected = self.xyz_dim \
                   + (0 if (sigma_only or self.embedding_dir is None) else 3) \
                   + (0 if (sigma_only or self.embedding_a is None) else 1)

        if x.shape[1] != expected:
            raise Exception(
                'Unexpected input shape: {} (expected: {}, xyz_dim: {})'.format(x.shape, expected, self.xyz_dim))

        input_xyz = self.embedding_xyz(x[:, :self.xyz_dim]) # 位置编码
        xyz_ = input_xyz
        for i, xyz_encoding in enumerate(self.xyz_encodings):
            if i in self.skip_layers:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = xyz_encoding(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_noise is not None:
            sigma += sigma_noise

        sigma = self.sigma_activation(sigma)

        if sigma_only:
            return sigma

        if self.xyz_encoding_final is not None:
            xyz_encoding_final = self.xyz_encoding_final(xyz_)
            dir_a_encoding_input = [xyz_encoding_final]

            if self.embedding_dir is not None:
                dir_a_encoding_input.append(self.embedding_dir(x[:, -4:-1]))

            if self.embedding_a is not None and self.affine is None:
                dir_a_encoding_input.append(self.embedding_a(x[:, -1].long()))

            dir_a_encoding = self.dir_a_encoding(torch.cat(dir_a_encoding_input, -1))
            rgb = self.rgb(dir_a_encoding)
        else:
            rgb = self.rgb(xyz_)

        if self.affine is not None and self.embedding_a is not None:
            affine_transform = self.affine(self.embedding_a(x[:, -1].long())).view(-1, 3, 4)
            rgb = (affine_transform[:, :, :3] @ rgb.unsqueeze(-1) + affine_transform[:, :, 3:]).squeeze(-1)

        return torch.cat([self.rgb_activation(rgb) if self.rgb_activation is not None else rgb, sigma], -1)
