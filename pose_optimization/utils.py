import torch
import numpy as np
import imageio
import cv2
import json
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default='../dataset/high-simple',
                        help='path to datatset folder')
    parser.add_argument('--nerf_config', type=str, default='mega-nerf-output/configs/hparams/high/hparams_simple.txt',
                        help='mega-nerf config file path')
    parser.add_argument("--output_dir", type=str, default='./output/',
                        help='where to store output logs, images, videos')
    parser.add_argument("--container_path", type=str, default=None,
                        help='path to merged nerf model')
    parser.add_argument("--video", action='store_false', help='output video of pose optimization')
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--nworkers", type=int, default=0)
    # iNeRF options
    parser.add_argument("--dil_iter", type=int, default=1,
                        help='Number of iterations of dilation process')
    parser.add_argument("--kernel_size", type=int, default=3,
                        help='Kernel size for dilation')
    parser.add_argument("--batch_size", type=int, default=2048,
                        help='Number of sampled rays per gradient step')
    parser.add_argument("--lrate", type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument("--sampling_strategy", type=str, default='random',
                        help='options: random / interest_point / interest_region')
    parser.add_argument("--steps", type=int, default=500,
                        help='optimization steps')
    # parameters to define initial pose
    parser.add_argument("--delta_psi", type=float, default=0.0,
                        help='Rotate camera around x axis degree')
    parser.add_argument("--delta_phi", type=float, default=0.0,
                        help='Rotate camera around z axis degree')
    parser.add_argument("--delta_theta", type=float, default=0.0,
                        help='Rotate camera around y axis degree')
    parser.add_argument("--delta_x", type=float, default=0.0,
                        help='translation of camera m')
    parser.add_argument("--delta_y", type=float, default=0.0,
                        help='translation of camera m')
    parser.add_argument("--delta_z", type=float, default=0.0,
                        help='translation of camera m')    
    # TDLF
    parser.add_argument("--tdlf", action='store_false', help='without tdlf or not')
    parser.add_argument("--alpha0", type=float, default=0.0, help='tdlf start value')
    # optimizaion
    parser.add_argument("--inerf", action='store_true', help='optimization on SE3 space, default is on tangent space')
    return parser

rot_psi = lambda phi: np.array([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1]])

rot_theta = lambda th: np.array([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1]])

rot_phi = lambda psi: np.array([
        [np.cos(psi), -np.sin(psi), 0, 0],
        [np.sin(psi), np.cos(psi), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

R_psi = lambda phi: np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]])

R_theta = lambda th: np.array([
        [np.cos(th), 0, -np.sin(th)],
        [0, 1, 0],
        [np.sin(th), 0, np.cos(th)]])

R_phi = lambda psi: np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]])

trans_t = lambda t: np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1]])

trans_x = lambda t: np.array([
        [1, 0, 0, t],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

trans_y = lambda t: np.array([
        [1, 0, 0, 0],
        [0, 1, 0, t],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

trans_z = lambda t: np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1]])


def rgb2bgr(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr


def show_img(title, img_rgb):  # img - rgb image
    img_bgr = rgb2bgr(img_rgb)
    cv2.imshow(title, img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_POI(img_rgb, DEBUG=False): # img - RGB image in range 0...255
    img = np.copy(img_rgb)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    nfeatures = 1024
    sift = cv2.SIFT_create(nfeatures)
    keypoints = sift.detect(img_gray, None)
    if DEBUG:
        for keypoint in keypoints:
            cv2.circle(img, (int(keypoint.pt[0]),int(keypoint.pt[1])), 15, (0, 0, 255), -1)
        # img = cv2.drawKeypoints(img_rgb, keypoints, img, (0, 255, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite("find_POI_num_{}.png".format(len(keypoints)), img)
        # show_img("Detected points", img)
    xy = [keypoint.pt for keypoint in keypoints]
    xy = np.array(xy).astype(int)
    # Remove duplicate points
    xy_set = set(tuple(point) for point in xy)
    xy = np.array([list(point) for point in xy_set]).astype(int)
    return xy # pixel coordinates

def find_Uniform_POI(img_rgb, patch_nums, DEBUG=False):
    POI = []
    img = np.copy(img_rgb)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    nfeatures = 8
    sift = cv2.SIFT_create(nfeatures)
    W = img_rgb.shape[1]
    H = img_rgb.shape[0]
    x_step = H // (patch_nums - 1)
    y_step = W // (patch_nums - 1)
    if DEBUG:
        img_keypoints = np.empty((img_rgb.shape[0], img_rgb.shape[1], 3), dtype=np.uint8)
        img_all = np.empty((img_rgb.shape[0], img_rgb.shape[1], 3), dtype=np.uint8)
        
    for x_idx in range(patch_nums - 1):
        if x_idx == patch_nums - 2:
            x_end = H - 1
        else:
            x_end = (x_idx + 1) * x_step
        for y_idx in range(patch_nums - 1):
            if y_idx == patch_nums - 2:
                y_end = W - 1
            else:
                y_end = (y_idx + 1) * y_step
            sub_img = img_gray[x_idx*x_step:x_end, y_idx*y_step:y_end]
            keypoints = sift.detect(sub_img, None)
            for i in range(len(keypoints)):
                keypoints[i].pt = (keypoints[i].pt[0] + y_idx*y_step, keypoints[i].pt[1] + x_idx*x_step)
            if DEBUG:
                for keypoint in keypoints:
                    cv2.circle(img_rgb, (int(keypoint.pt[0]),int(keypoint.pt[1])), 15, (0, 0, 255), -1)
                # cv2.drawKeypoints(img_rgb, keypoints, img_keypoints, (0, 255, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # 会覆盖掉img_keypoints上一次的结果，使用img_all保存
                img_all[x_idx*x_step:x_end, y_idx*y_step:y_end, :] = img_rgb[x_idx*x_step:x_end, y_idx*y_step:y_end, :]
            xy = [keypoint.pt for keypoint in keypoints]
            xy = np.array(xy).astype(int)
            # Remove duplicate points
            xy_set = set(tuple(point) for point in xy)
            sub_POI = np.array([list(point) for point in xy_set]).astype(int)
            if(sub_POI.shape[0] != 0):
                POI.append(sub_POI) # + np.array([y_idx*y_step, x_idx*x_step]))
    POI = np.concatenate(POI)
    if DEBUG:
        cv2.imwrite("find_Uniform_POI_num_{}.png".format(POI.shape[0]), img_all)
    return POI

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from subprocess import check_output

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100. / r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')


def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape

    sfx = ''

    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1

    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print(imgdir, 'does not exist, returning')
        return

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print('Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]))
        return

    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor

    if not load_imgs:
        return poses, bds

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    imgs = imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
    imgs = np.stack(imgs, -1)

    print('Loaded image data', imgs.shape, poses[:, -1, 0])
    return poses, bds, imgs


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def recenter_poses(poses):
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds):
    p34_to_44 = lambda p: np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1)

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1, .2, .3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1. / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]

    poses_reset = np.concatenate(
        [poses_reset[:, :3, :4], np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)], -1)

    return poses_reset, bds


def get_tensorboard_writer(path):
    exp_dir = Path(path)
    exp_dir.mkdir(parents=True, exist_ok=True)
    existing_versions = [int(x.name) for x in exp_dir.iterdir()]
    version = 0 if len(existing_versions) == 0 else max(existing_versions) + 1
    experiment_path = exp_dir / str(version)
    writer = SummaryWriter(log_dir = experiment_path)
    return writer
