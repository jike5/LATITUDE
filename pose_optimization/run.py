import os
import torch
import imageio
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from utils import config_parser, find_Uniform_POI, find_POI, img2mse, trans_x, trans_y, trans_z, R_phi, R_theta, R_psi
from utils import read_pose_file
from render_helpers import to8b
from inerf_helpers import camera_transf

from render_functions import Render
from dataloader import get_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

def run(args, log_dir):
    Writer = SummaryWriter(log_dir=log_dir)
    
    data_conf = {}
    train_dataset, val_dataset, train_loader, val_loader = get_dataloader(args.data_dir, data_conf, args.bsz, args.nworkers)
    hwf, K = val_dataset.get_hwf_K()
    MAP_SCALE = val_dataset.get_map_scale()
    render = Render(args.nerf_config, args.data_dir, hwf, K, args.container_path)

    if args.tdlf and not args.inerf:
        for i in range(len(render.nerf.sub_modules)):
            render.nerf.sub_modules[i].embedding_xyz.progress.data.fill_(0. + args.alpha0) # default is 0.5
            render.nerf.sub_modules[i].embedding_dir.progress.data.fill_(0. + args.alpha0)

    delta_x = args.delta_x / MAP_SCALE
    delta_y = args.delta_y / MAP_SCALE
    delta_z = args.delta_z / MAP_SCALE

    use_pose_regressor_as_init = False
    init_poses_list = []
    if args.pose_regressor_input is not None:
        use_pose_regressor_as_init = True
        init_poses_list = read_pose_file(args.pose_regressor_input, MAP_SCALE)

    last_idx = len(train_loader) - 1
    for batchi, (imgs, poses, hwfs, Ks) in enumerate(train_loader):
        batch_len = imgs.shape[0] # B H W C
        for index in range(batch_len):
            img, pose, hwf, K = imgs[index].cpu().numpy(), poses[index].cpu(), hwfs[index].cpu(), Ks[index].cpu()
            video_save_dir = os.path.join(log_dir, "video", "{}_{}".format(str(batchi), str(index)))
            os.makedirs(video_save_dir, exist_ok=True)
            render_save_dir = os.path.join(log_dir, "redering", "{}_{}".format(str(batchi), str(index)))
            os.makedirs(render_save_dir, exist_ok=True)
            result_save_dir = os.path.join(log_dir, "results", "{}_{}".format(str(batchi), str(index)))
            os.makedirs(result_save_dir, exist_ok=True)

            H, W, Focal = hwf[0],  hwf[1],  hwf[2]
            # ground turth pose
            obs_img_pose = np.eye(4)
            obs_img_pose[:3, :4] = pose.numpy()
            # add noise
            obs_img_pose_3 = np.eye(3)
            obs_img_pose_3 = obs_img_pose[:3, :3]
            # rotation
            obs_img_pose_R = R_phi(args.delta_phi/180.*np.pi) @ \
                            R_theta(args.delta_theta/180.*np.pi) @ \
                            R_psi(args.delta_psi/180.*np.pi) @ obs_img_pose_3
            obs_img_pose_4 = np.eye(4)
            obs_img_pose_4[:3, :3] = obs_img_pose_R
            obs_img_pose_4[:3, 3] = obs_img_pose[:3, 3]
            if use_pose_regressor_as_init:
                start_pose = init_poses_list[index]
            else:
                start_pose = trans_x(delta_x) @ trans_y(delta_y) @ trans_z(delta_z) @ obs_img_pose_4

            # find points of interest of the observed image
            if args.sampling_strategy == 'uniform_interest_regions':
                POI = find_Uniform_POI(img, args.patch_nums, DEBUG)
            else:
                POI = find_POI(img, DEBUG)  # xy pixel coordinates of points of interest (N x 2)
            obs_img = (img / 255.).astype(np.float32)
            # create meshgrid from the observed image
            coords = np.asarray(np.stack(np.meshgrid(np.linspace(0, W - 1, W), np.linspace(0, H - 1, H)), -1),
                                dtype=int)
            
            # create sampling mask for interest region sampling strategy
            interest_regions = np.zeros((H, W, ), dtype=np.uint8)
            interest_regions[POI[:,1], POI[:,0]] = 1
            I = args.dil_iter
            interest_regions = cv2.dilate(interest_regions, np.ones((args.kernel_size, args.kernel_size), np.uint8), iterations=I)
            interest_regions = np.array(interest_regions, dtype=bool)
            interest_regions = coords[interest_regions]
            # not_POI -> contains all points except of POI
            coords = coords.reshape(H * W, 2)
            not_POI = set(tuple(point) for point in coords) - set(tuple(point) for point in POI)
            not_POI = np.array([list(point) for point in not_POI]).astype(int)

            # Create pose transformation model
            start_pose = torch.Tensor(start_pose).to(device)
            cam_transf = camera_transf().to(device)
            if args.inerf:
                pose = torch.clone(start_pose.detach())
                pose.requires_grad = True
                optimizer = torch.optim.Adam(params=[pose], lr=args.lrate, betas=(0.9, 0.999))
            else:
                optimizer = torch.optim.Adam(params=cam_transf.parameters(), lr=args.lrate, betas=(0.9, 0.999))
            # calculate angles and translation of the observed image's pose
            phi_ref = np.arctan2(obs_img_pose[1,0], obs_img_pose[0,0])*180/np.pi
            theta_ref = np.arctan2(-obs_img_pose[2, 0], np.sqrt(obs_img_pose[2, 1]**2 + obs_img_pose[2, 2]**2))*180/np.pi
            psi_ref = np.arctan2(obs_img_pose[2, 1], obs_img_pose[2, 2])*180/np.pi
            translation_ref = np.sqrt(obs_img_pose[0,3]**2 + obs_img_pose[1,3]**2 + obs_img_pose[2,3]**2)
            position_ref = obs_img_pose[:3, 3]
            # create a video of optimization process
            if args.video is True:
                imgs = []

            optimize_pose_csv = os.path.join(result_save_dir, "optimize_pose_{}_{}.csv".format(batchi, index))
            with open(optimize_pose_csv, 'w') as f:
                pass

            for k in range(args.steps):
                if args.sampling_strategy == 'random':
                    rand_inds = np.random.choice(coords.shape[0], size=args.batch_size, replace=False)
                    batch = coords[rand_inds]
                elif args.sampling_strategy == 'interest_points':
                    if POI.shape[0] >= args.batch_size:
                        rand_inds = np.random.choice(POI.shape[0], size=args.batch_size, replace=False)
                        batch = POI[rand_inds]
                    else:
                        batch = np.zeros((args.batch_size, 2), dtype=np.int)
                        batch[:POI.shape[0]] = POI
                        rand_inds = np.random.choice(not_POI.shape[0], size=args.batch_size-POI.shape[0], replace=False)
                        batch[POI.shape[0]:] = not_POI[rand_inds]
                elif args.sampling_strategy == 'interest_regions':
                    rand_inds = np.random.choice(interest_regions.shape[0], size=args.batch_size, replace=False)
                    batch = interest_regions[rand_inds]
                elif args.sampling_strategy == 'uniform_interest_regions':
                    rand_inds = np.random.choice(interest_regions.shape[0], size=args.batch_size, replace=False)
                    batch = interest_regions[rand_inds]
                else:
                    print('Unknown sampling strategy')
                    return
                target_s = obs_img[batch[:, 1], batch[:, 0]]
                target_s = torch.Tensor(target_s).to(device)
                if not args.inerf:
                    pose = cam_transf(start_pose) # 4 * 4

                rgb = render.get_img_from_pix(batch, pose)

                optimizer.zero_grad()
                loss = img2mse(rgb, target_s)
                loss.backward()
                optimizer.step()

                new_lrate = args.lrate * (0.8 ** ((k + 1) / 100))
                Writer.add_scalar("val{}/lrate".format(index), new_lrate, k)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lrate

                if (k + 1) % 50 == 0 or k == 0:
                    print('Step: ', k)
                    print('Loss: ', loss)

                    with torch.no_grad():
                        pose_dummy = pose.cpu().detach().numpy()
                        # calculate angles and translation of the optimized pose
                        phi = np.arctan2(pose_dummy[1, 0], pose_dummy[0, 0]) * 180 / np.pi
                        theta = np.arctan2(-pose_dummy[2, 0], np.sqrt(pose_dummy[2, 1] ** 2 + pose_dummy[2, 2] ** 2)) * 180 / np.pi
                        psi = np.arctan2(pose_dummy[2, 1], pose_dummy[2, 2]) * 180 / np.pi
                        translation = np.sqrt(pose_dummy[0,3]**2 + pose_dummy[1,3]**2 + pose_dummy[2,3]**2)
                        position = pose_dummy[:3, 3]
                        # calculate error between optimized and observed pose
                        phi_error = abs(phi_ref - phi) if abs(phi_ref - phi)<300 else abs(abs(phi_ref - phi)-360)
                        theta_error = abs(theta_ref - theta) if abs(theta_ref - theta)<300 else abs(abs(theta_ref - theta)-360)
                        psi_error = abs(psi_ref - psi) if abs(psi_ref - psi)<300 else abs(abs(psi_ref - psi)-360)
                        rot_error = phi_error + theta_error + psi_error
                        translation_error = MAP_SCALE * abs(translation_ref - translation)
                        position_error = MAP_SCALE * np.linalg.norm(position_ref - position)
                        Writer.add_scalar("val{}/error/R_deg".format(index), rot_error, k)
                        Writer.add_scalar("val{}/error/T_m".format(index), translation_error, k)
                        Writer.add_scalar("val{}/error/P_m".format(index), position_error, k)

                        print('Rotation error: ', rot_error)
                        print('Translation error: ', translation_error)
                        print('Position error: ', position_error)
                        print('-----------------------------------')
                        # d. 保存过程结果
                        with open(optimize_pose_csv, 'a') as f:
                            # 写入kitti结果中
                            rot = np.zeros((3, 4))
                            rot[0:3, 0:3] = np.array(pose[:3, :3].cpu().detach().numpy())  # 四元组转旋转矩阵后复制到新矩阵的左上角
                            rot[0:3, 3] = np.array(pose[:3, 3].cpu().detach().numpy())
                            output = np.resize(rot, (12)).tolist()
                            f.write(" ".join(str(i) for i in output))
                            f.write('\n')

                    if args.video:
                        with torch.no_grad():
                            rgb = render.get_img_from_pose(pose)
                            rgb = rgb.cpu().detach().numpy()
                            rgb.resize(obs_img.shape)
                            rgb8 = to8b(rgb)
                            ref = to8b(obs_img)
                            filename = os.path.join(video_save_dir, str(k)+'.png')
                            dst = cv2.addWeighted(rgb8, 0.7, ref, 0.3, 0)
                            imageio.imwrite(filename, dst)

                            filename = os.path.join(render_save_dir, str(k)+'.png')
                            imageio.imwrite(filename, rgb8)

                            imgs.append(dst)

                    if args.tdlf and not args.inerf:
                        # print(len(render.nerf.sub_modules))
                        for i in range(len(render.nerf.sub_modules)):
                            render.nerf.sub_modules[i].embedding_xyz.progress.data.fill_(k / args.steps + args.alpha0)
                            render.nerf.sub_modules[i].embedding_dir.progress.data.fill_(k / args.steps + args.alpha0)

            if args.video:
                imageio.mimwrite(os.path.join(video_save_dir, 'video.gif'), imgs, fps=8)

    Writer.close()

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    # Parameters
    parser = config_parser()
    args = parser.parse_args()
    
    # save parser args
    exp_dir = Path(args.output_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    existing_versions = [int(x.name) for x in exp_dir.iterdir()]
    exp_index = 0 if len(existing_versions) == 0 else max(existing_versions) + 1
    log_dir = os.path.join(args.output_dir, str(exp_index))
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'config.txt'), 'w') as f:
        args_dict = args.__dict__
        for each_arg, value in args_dict.items():
            f.writelines(each_arg + " : " + str(value) + "\n")

    run(args, log_dir)
