from typing import List, Dict

import torch
from torch.utils.data import Dataset
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from datasets.dataset_utils import get_rgb_index_mask
from image_metadata import ImageMetadata
from misc_utils import main_tqdm, main_print
from ray_utils import get_rays, get_ray_directions


class MemoryDataset(Dataset):

    def __init__(self, metadata_items: List[ImageMetadata], near: float, far: float, ray_altitude_range: List[float],
                 center_pixels: bool, device: torch.device):
        super(MemoryDataset, self).__init__()

        rgbs = []
        rays = []
        indices = []

        main_print('Loading data')

        for metadata_item in main_tqdm(metadata_items):
            image_data = get_rgb_index_mask(metadata_item)

            if image_data is None:
                continue

            image_rgbs, image_indices, image_keep_mask = image_data

            directions = get_ray_directions(metadata_item.W,
                                            metadata_item.H,
                                            metadata_item.intrinsics[0],
                                            metadata_item.intrinsics[1],
                                            metadata_item.intrinsics[2],
                                            metadata_item.intrinsics[3],
                                            center_pixels,
                                            device)
            image_rays = get_rays(directions, metadata_item.c2w.to(device), near, far, ray_altitude_range).view(-1,
                                                                                                                8).cpu()
            if image_keep_mask is not None:
                image_rays = image_rays[image_keep_mask == True]

            rgbs.append(image_rgbs.float() / 255.)
            rays.append(image_rays)
            indices.append(image_indices)

        main_print('Finished loading data')

        self._rgbs = torch.cat(rgbs)
        self._rays = torch.cat(rays)
        self._img_indices = torch.cat(indices)

    def __len__(self) -> int:
        return self._rgbs.shape[0]

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {
            'rgbs': self._rgbs[idx],
            'rays': self._rays[idx],
            'img_indices': self._img_indices[idx]
        }
