import torch
from torch.utils import data
from pathlib import Path
from PIL import Image
import numpy as np

class MegaNeRFDataset(data.Dataset):
    def __init__(self, dataroot, data_conf, is_train):
        self.dataroot = dataroot
        dataset_path = Path(self.dataroot)
        if is_train:
            sub_dir = 'train'
        else:
            sub_dir = 'val'
        img_path_candidates = sorted(list((dataset_path / sub_dir / 'rgbs').iterdir()))
        self.img_paths = [img_path_candidates[i] for i in
                        range(0, len(img_path_candidates))] # 000xx.jpg
        
        metadata_path_candidates = sorted(list((dataset_path / sub_dir / 'metadata').iterdir()))
        self.metadata_paths = [metadata_path_candidates[i] for i in
                        range(0, len(metadata_path_candidates))] # 000xx.pt
        
        self.coordiantes = torch.load(self.dataroot + "/coordinates.pt", map_location='cpu')
        self.MAP_SCALE = float(self.coordiantes["pose_scale_factor"])     

    def __getitem__(self, index):
        metadata_path = self.metadata_paths[index]
        img_path = self.img_paths[index]
        rgbs = Image.open(img_path).convert('RGB')
        obs_img = np.asarray(rgbs)
        img = torch.tensor(obs_img) # uint8
        metadata = torch.load(metadata_path, map_location='cpu')
        
        intrinsics = metadata['intrinsics']
        hwf = torch.tensor([int(metadata['H']), int(metadata['W']), int(intrinsics[0])])
        H, W, focal = hwf
        K = torch.tensor([
            [focal, 0, float(intrinsics[2])],
            [0, focal, float(intrinsics[3])],
            [0, 0, 1]
        ])
        pose = metadata["c2w"] # (3,4)
        return img, pose, hwf, K
        
    def get_hwf_K(self):
        _, _, hwf, K = self.__getitem__(0)
        return hwf, K
    
    def get_map_scale(self):
        return self.MAP_SCALE
    
    def __len__(self):
        return len(self.metadata_paths)
    
def get_dataloader(dataroot, data_conf, bsz, nworkers):
    train_dataset = MegaNeRFDataset(dataroot, data_conf, is_train=True)
    val_dataset = MegaNeRFDataset(dataroot, data_conf, is_train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bsz, shuffle=True,  generator=torch.Generator(device='cuda'), num_workers=nworkers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=nworkers)
    return train_dataset, val_dataset, train_loader, val_loader