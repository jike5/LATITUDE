from argparse import Namespace
import sys

import torch
from torch.distributed.elastic.multiprocessing.errors import record

from opts import get_opts_base
from runner import Runner


def _get_eval_opts() -> Namespace:
    parser = get_opts_base()

    parser.add_argument('--exp_name', type=str, required=True, help='experiment name')
    parser.add_argument('--dataset_path', type=str, required=True)

    return parser.parse_args()

@record
def main(hparams: Namespace) -> None:
    assert hparams.ckpt_path is not None or hparams.container_path is not None

    if hparams.detect_anomalies: # 检测异常 None
        with torch.autograd.detect_anomaly():
            Runner(hparams).eval()
    else:
        Runner(hparams).eval() # goin


if __name__ == '__main__':
    main(_get_eval_opts())
