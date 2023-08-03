import os
import sys
sys.path.append('../../')
from mega_nerf.get_meganerf import load_mega_nerf
import random
from torch import optim
from pose_regressor.dataset_loaders.load_mega_nerf import load_mega_nerf_dataloader
from pose_regressor.script.utils import freeze_bn_layer, freeze_bn_layer_train
from tqdm import tqdm
from callbacks import EarlyStopping
from pose_regressor.script.dfnet import DFNet, DFNet_s
from pose_regressor.script.misc import *
from pose_regressor.script.options import config_parser
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)
parser = config_parser()
args = parser.parse_args()


def train_on_batch(args, rgbs, poses, feat_model, dset_size, FeatureLoss, optimizer, hwf):
    ''' core training loop for featurenet'''
    feat_model.train()
    H, W, focal = hwf
    H, W = int(H), int(W)
    if args.freezeBN:
        feat_model = freeze_bn_layer_train(feat_model)

    train_loss_epoch = []
    select_inds = np.random.choice(dset_size, size=[dset_size], replace=False)  # (N_rand,)

    batch_size=args.featurenet_batch_size # manual setting, use smaller batch size like featurenet_batch_size = 4 if OOM
    if dset_size % batch_size == 0:
        N_iters = dset_size//batch_size
    else:
        N_iters = dset_size//batch_size + 1
    i_batch = 0

    for i in range(0, N_iters):
        if i_batch + batch_size > dset_size:
            i_batch = 0
            break
        i_inds = select_inds[i_batch:i_batch+batch_size]
        i_batch = i_batch + batch_size

        # convert input shape to [B, 3, H, W]
        # target_in = targets[i_inds].clone().permute(0,3,1,2).to(device)
        rgb_in = rgbs[i_inds].clone().permute(0,3,1,2).to(device)
        pose = poses[i_inds].clone().reshape(batch_size, 12).to(device)
        # pose = torch.cat([pose, pose]) # double gt pose tensor

        features, predict_pose = feat_model(rgb_in, False, upsampleH=H, upsampleW=W) # features: (1, [2, B, C, H, W])
        pose[:, [3, 7, 11]] *= args.map_scale
        loss = PoseLoss(args, predict_pose, pose, device)  # target

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss_epoch.append(loss.item())
    train_loss = np.mean(train_loss_epoch)
    return train_loss


def train_on_batch_with_random_view_synthesis(args, rgbs, poses, virtue_view, poses_perturb, feat_model, dset_size, FeatureLoss, optimizer, hwf):
    ''' we implement random view synthesis for generating more views to help training posenet '''
    feat_model.train()

    H, W, focal = hwf
    H, W = int(H), int(W)

    if args.freezeBN:
        feat_model = freeze_bn_layer_train(feat_model)

    train_loss_epoch = []

    # random generate batch_size of idx
    select_inds = np.random.choice(dset_size, size=[dset_size], replace=False)  # (N_rand,)

    batch_size = args.featurenet_batch_size
    # manual setting, use smaller batch size like featurenet_batch_size = 4 if OOM
    if dset_size % batch_size == 0:
        N_iters = dset_size//batch_size
    else:
        N_iters = dset_size//batch_size + 1
    
    i_batch = 0
    for i in range(0, N_iters):
        if i_batch + batch_size > dset_size:
            break
        i_inds = select_inds[i_batch:i_batch+batch_size]
        i_batch = i_batch + batch_size

        # convert input shape to [B, 3, H, W]
        rgb_in = rgbs[i_inds].clone().permute(0, 3, 1, 2).to(device)
        pose = poses[i_inds].clone().reshape(batch_size, 12).to(device)
        rgb_perturb = virtue_view[i_inds].clone().permute(0, 3, 1, 2).to(device)
        pose_perturb = poses_perturb[i_inds].clone().reshape(batch_size, 12).to(device)

        # inference feature model for GT and nerf image
        _, predict_pose = feat_model(rgb_in, return_feature=False, upsampleH=H, upsampleW=W)
        pose[:, [3, 7, 11]] *= args.map_scale
        pose_perturb[:, [3, 7, 11]] *= args.map_scale
        loss_pose = PoseLoss(args, predict_pose, pose, device)  # target
        loss_f = 0  # we don't use it

        # inference model for RVS image
        _, virtue_pose = feat_model(rgb_perturb.to(device), False)
        loss_pose_perturb = PoseLoss(args, virtue_pose, pose_perturb, device)
        loss = args.combine_loss_w[0]*loss_pose + args.combine_loss_w[1]*loss_f + args.combine_loss_w[2]*loss_pose_perturb

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss_epoch.append(loss.item())
    train_loss = np.mean(train_loss_epoch)
    return train_loss


def train_feature(args, train_dl, val_dl, test_dl, hwf, i_split, mega_nerf_model):
    writer = SummaryWriter()
    # # load pretrained PoseNet model
    if args.DFNet_s:
        feat_model = DFNet_s()
    else:
        feat_model = DFNet()
    
    if args.pretrain_model_path != '':
        print("load posenet from ", args.pretrain_model_path)
        feat_model.load_state_dict(torch.load(args.pretrain_model_path))
    
    # # Freeze BN to not updating gamma and beta
    if args.freezeBN:
        feat_model = freeze_bn_layer(feat_model)

    feat_model.to(device)
    # summary(feat_model, (3, 240, 427))

    # set optimizer
    optimizer = optim.Adam(feat_model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.95, patience=args.patience[1], verbose=True)

    # set callbacks parameters
    early_stopping = EarlyStopping(args, patience=args.patience[0], verbose=False)

    # loss function
    loss_func = nn.MSELoss(reduction='mean')
    i_train, i_val, i_test = i_split
    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    N_epoch = args.epochs + 1  # epoch
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)


    if args.eval:
        feat_model.eval()
        get_error_in_q(args, test_dl, feat_model, len(val_dl.dataset), device, batch_size=1)
        sys.exit()

    rgb_list = []
    pose_list = []
    for data, pose, img_idx in train_dl:
        # convert to H W C
        rgb_list.append(data[0].permute(1, 2, 0))
        pose_list.append(pose[0].reshape(3, 4))
    rgbs = torch.stack(rgb_list).detach()
    poses = torch.stack(pose_list).detach()
    # rgbs = targets.clone()
    # targets, rgbs, poses = mega_nerf_model.render_mega_nerf_imgs(args, train_dl, hwf, device)

    # visualize rgb
    # unloader = torchvision.transforms.ToPILImage()
    # for i in range(rgbs.shape[0]):
    #     vis = rgbs[i].permute(2, 0, 1)
    #     writer.add_image("rgb_images", vis, i)
    #     image = vis.clone()  # clone the tensor
    #     image = image.squeeze(0)  # remove the fake batch dimension
    #     image = unloader(image)
    #     image.save('./output_img_rgb/' + str(i) + '.jpg')


    dset_size = len(train_dl.dataset)
    # clean GPU memory before testing, try to avoid OOM
    torch.cuda.empty_cache()

    model_log = tqdm(total=0, position=1, bar_format='{desc}')
    for epoch in tqdm(range(N_epoch), desc='epochs'):

        if args.random_view_synthesis:
            isRVS = epoch % args.rvs_refresh_rate == 0  # decide if to resynthesis new views

            if isRVS:
                # random sample virtual camera locations, todo:
                rand_trans = args.rvs_trans
                rand_rot = args.rvs_rotation

                # determine bounding box
                b_min = [poses[:, 0, 3].min()-args.d_max, poses[:, 1, 3].min()-args.d_max, poses[:, 2, 3].min()-args.d_max]
                b_max = [poses[:, 0, 3].max()+args.d_max, poses[:, 1, 3].max()+args.d_max, poses[:, 2, 3].max()+args.d_max]

                # 扰动
                poses_perturb = poses.clone().numpy()
                for i in range(dset_size):
                    poses_perturb[i] = perturb_single_render_pose(poses_perturb[i], rand_trans, rand_rot)
                    for j in range(3):
                        if poses_perturb[i,j,3] < b_min[j]:
                            poses_perturb[i,j,3] = b_min[j]
                        elif poses_perturb[i,j,3]> b_max[j]:
                            poses_perturb[i,j,3] = b_max[j]

                poses_perturb = torch.Tensor(poses_perturb).to(device)  # [B, 3, 4]
                tqdm.write("renders RVS...")
                virtue_view = mega_nerf_model.render_virtual_meganerf_imgs(args, poses_perturb, hwf, device)
                '''
                visualization
                '''
                # unloader = torchvision.transforms.ToPILImage()
                # for i in range(virtue_view.shape[0]):
                #     vis = virtue_view[i].permute(2, 0, 1)
                #     writer.add_image("virtual_images", vis, i)
                #     image = vis.clone()  # clone the tensor
                #     image = image.squeeze(0)  # remove the fake batch dimension
                #     image = unloader(image)
                #     image.save('./output_img_virtual/' + str(i) + '.jpg')

            train_loss = train_on_batch_with_random_view_synthesis(args, rgbs, poses, virtue_view, poses_perturb, feat_model, dset_size, loss_func, optimizer, hwf)
            
        else:
            train_loss = train_on_batch(args, rgbs, poses, feat_model, dset_size, loss_func, optimizer, hwf)

        feat_model.eval()
        val_loss_epoch = []
        for data, pose, _ in val_dl:
            inputs = data.to(device)
            labels = pose.to(device)
            # labels = labels.view(1, 12)
            # pose loss
            labels[:, [3, 7, 11]] *= args.map_scale
            _, predict = feat_model(inputs)
            loss = loss_func(predict, labels)
            val_loss_epoch.append(loss.item())
        val_loss = np.mean(val_loss_epoch)

        # reduce LR on plateau
        scheduler.step(val_loss)

        # logging
        tqdm.write('At epoch {0:6d} : train loss: {1:.4f}, val loss: {2:.4f}'.format(epoch, train_loss, val_loss))
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', val_loss, epoch)

        # check wether to early stop
        early_stopping(val_loss, feat_model, epoch=epoch, save_multiple=(not args.no_save_multiple), save_all=args.save_all_ckpt)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        model_log.set_description_str(f'Best val loss: {early_stopping.val_loss_min:.4f}')
        if epoch % args.i_eval == 0:
            mediaT, mediaR, meanT, meanR = get_error_in_q(args, test_dl, feat_model, len(test_dl.dataset), device, batch_size=1)
            writer.add_scalar("Test/mediaTranslation", mediaT, epoch)
            writer.add_scalar("Test/mediaRotation", mediaR, epoch)
            writer.add_scalar("Test/meanTranslation", meanT, epoch)
            writer.add_scalar("Test/meanRotation", meanR, epoch)

    writer.close()    # global_step += 1
    return


def train():
    print(parser.format_values())
    mega_nerf_model = load_mega_nerf(args.exp_name, args.datadir, args.config_file, args.container_path)
    assert args.dataset_type == 'mega_nerf'
    train_dl, val_dl, test_dl, hwf, i_split = load_mega_nerf_dataloader(args, mega_nerf_model)
    train_feature(args, train_dl, val_dl, test_dl, hwf, i_split, mega_nerf_model)
    return


if __name__ == "__main__":
    train()