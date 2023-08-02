import os.path as osp
import torch.cuda
from torch.utils.data import DataLoader
from torchvision import transforms
from pose_regressor.dataset_loaders.mega_nerf_data import mega_nerf_data


def load_mega_nerf_dataloader(args, mega_nerf_model):
    ''' Data loader for Pose Regression Network '''
    if args.pose_only:  # if train posenet is true
        pass
    else:
        raise Exception('wrong setting')
    data_dir, scene = osp.split(args.datadir)

    # transformer
    data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    target_transform = transforms.Lambda(lambda x: torch.Tensor(x))

    ret_idx = False  # return frame index
    fix_idx = False  # return frame index=0 in training
    ret_hist = False

    if 'NeRFH' in args:
        if args.NeRFH == True:
            ret_idx = True
            if args.fix_index:
                fix_idx = True

    # encode hist experiment
    if args.encode_hist:
        ret_idx = False
        fix_idx = False
        ret_hist = True

    kwargs = dict(scene=scene, data_path=data_dir,
                  transform=data_transform, target_transform=target_transform,
                  df=args.df, ret_idx=ret_idx, fix_idx=fix_idx,
                  ret_hist=ret_hist, hist_bin=args.hist_bin,
                  hwf=[mega_nerf_model.train_items[0].H, mega_nerf_model.train_items[0].W,
                       float(mega_nerf_model.train_items[0].intrinsics[0])])

    train_set = mega_nerf_data(train=True, trainskip=args.trainskip, **kwargs)
    val_set = mega_nerf_data(train=False, testskip=args.testskip, **kwargs)

    i_train = train_set.gt_idx
    i_val = val_set.gt_idx
    i_test = val_set.gt_idx

    train_dl = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=8)
    val_dl = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, num_workers=2)
    test_dl = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)

    hwf = [train_set.H, train_set.W, train_set.focal]
    i_split = [i_train, i_val, i_test]

    return train_dl, val_dl, test_dl, hwf, i_split
