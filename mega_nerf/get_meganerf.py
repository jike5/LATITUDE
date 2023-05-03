from mega_nerf.opts import get_opts_base
from mega_nerf.runner import Runner


def load_mega_nerf(exp_name, dataset_path, config_file, container_path):
    parser2 = get_opts_base()
    parser2.add_argument('--exp_name', default="simple", type=str, help='experiment name')
    parser2.add_argument('--dataset_path', default="", type=str)
    a = parser2.parse_args(["--config_file", config_file, "--exp_name", exp_name, "--dataset_path", dataset_path,
                            "--container_path", container_path])
    assert a.ckpt_path is not None or a.container_path is not None
    return Runner(a)