"""
pytorch data loader for the mega_nerf dataset
"""
import os
import os.path as osp
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import sys
import cv2
from PoseRegressor.dataset_loaders.utils.color import rgb_to_yuv
from torchvision.datasets.folder import default_loader
sys.path.insert(0, '../../')


def load_image(filename, loader=default_loader):
    try:
        img = loader(filename)
    except IOError as e:
        print('Could not load image {:s}, IOError: {:s}'.format(filename, e))
        return None
    except:
        print('Could not load image {:s}, unexpected error'.format(filename))
        return None
    return img


def load_depth_image(filename):
    try:
        img_depth = Image.fromarray(np.array(Image.open(filename)).astype("uint16"))
    except IOError as e:
        print('Could not load image {:s}, IOError: {:s}'.format(filename, e))
        return None
    return img_depth


class mega_nerf_data(data.Dataset):
    def __init__(self, scene, data_path, train, transform=None, target_transform=None, seed=7, df=2., trainskip=1,
                 testskip=1, hwf=[480, 960, 480.], ret_idx=False, fix_idx=False, ret_hist=False, hist_bin=10):
        """
        :param scene: scene name ['chess', 'pumpkin', ...]
        :param data_path: root 7scenes data directory.
        Usually '../data/deepslam_data/7Scenes'
        :param train: if True, return the training images. If False, returns the
        testing images
        :param transform: transform to apply to the images
        :param target_transform: transform to apply to the poses
        :param skip_images: If True, skip loading images and return None instead
        :param df: downscale factor
        :param trainskip: due to 7scenes are so big, now can use less training sets # of trainset = 1/trainskip
        :param testskip: skip part of testset, # of testset = 1/testskip
        :param hwf: H,W,Focal from COLMAP
        """

        self.transform = transform
        self.target_transform = target_transform
        self.df = df

        self.H, self.W, self.focal = hwf
        self.H = int(self.H)
        self.W = int(self.W)
        np.random.seed(seed)

        self.train = train
        self.ret_idx = ret_idx
        self.fix_idx = fix_idx
        self.ret_hist = ret_hist
        self.hist_bin = hist_bin  # histogram bin size

        if self.train:
            root_dir = osp.join(data_path, scene) + '/train'
        else:
            root_dir = osp.join(data_path, scene) + '/val'

        rgb_dir = root_dir + '/rgbs/'

        pose_dir = root_dir + '/metadata/'

        # collect poses and image names
        self.rgb_files = os.listdir(rgb_dir)
        self.rgb_files = [rgb_dir + f for f in self.rgb_files]
        self.rgb_files.sort()

        self.pose_files = os.listdir(pose_dir)
        self.pose_files = [pose_dir + f for f in self.pose_files]
        self.pose_files.sort()

        if len(self.rgb_files) != len(self.pose_files):
            raise Exception('RGB file count does not match pose file count!')

        # trainskip and testskip
        frame_idx = np.arange(len(self.rgb_files))
        if train and trainskip > 1:
            frame_idx_tmp = frame_idx[::trainskip]
            frame_idx = frame_idx_tmp
        elif not train and testskip > 1:
            frame_idx_tmp = frame_idx[::testskip]
            frame_idx = frame_idx_tmp
        self.gt_idx = frame_idx

        self.rgb_files = [self.rgb_files[i] for i in frame_idx]
        self.pose_files = [self.pose_files[i] for i in frame_idx]

        if len(self.rgb_files) != len(self.pose_files):
            raise Exception('RGB file count does not match pose file count!')

        # read poses
        poses = []
        for i in range(len(self.pose_files)):
            pose = torch.load(self.pose_files[i])
            c2w = np.array(pose['c2w'])
            zero = [[0, 0, 0, 1]]
            pose = np.r_[c2w, zero]
            poses.append(pose)
        poses = np.array(poses)  # [N, 4, 4]
        self.poses = poses[:, :3, :4].reshape(poses.shape[0], 12)
        # debug read one img and get the shape of the img
        img = load_image(self.rgb_files[0])
        img_np = (np.array(img) / 255.).astype(np.float32)

        self.H, self.W = img_np.shape[:2]
        if self.df != 1.:
            self.H = int(self.H // self.df)
            self.W = int(self.W // self.df)
            self.focal = self.focal / self.df

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, index):
        img = load_image(self.rgb_files[index])
        pose = self.poses[index]
        if self.df != 1.:
            img_np = (np.array(img) / 255.).astype(np.float32)
            dims = (self.W, self.H)
            img_half_res = cv2.resize(img_np, dims, interpolation=cv2.INTER_AREA)  # (H, W, 3)
            img = img_half_res

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            pose = self.target_transform(pose)

        if self.ret_idx:
            if self.train and self.fix_idx == False:
                return img, pose, index
            else:
                return img, pose, 0

        if self.ret_hist:
            yuv = rgb_to_yuv(img)
            y_img = yuv[0]  # extract y channel only
            hist = torch.histc(y_img, bins=self.hist_bin, min=0., max=1.)  # compute intensity histogram
            hist = hist / (hist.sum()) * 100  # convert to histogram density, in terms of percentage per bin
            hist = torch.round(hist)
            return img, pose, hist
        return img, pose
