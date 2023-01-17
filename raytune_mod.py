import argparse
import os
import shutil
import sys

from ray import tune
from ray.air.config import RunConfig
from torchvision.datasets import CIFAR100


class HiddenPrints:
    def __init__(self):
        self._original_stdout = None

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w', encoding="utf8")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def train(config):
    args, _ = get_args_parser().parse_known_args()
    args.ray = True
    args.wandb = True
    args.wandb_project = "icml"
    args.wandb_tag = ["sgd_torch"]
    args.ignore_warning = True
    args.momentum = 0
    args.data_path = os.path.join(os.getenv("HINADORI_LOCAL_SCRATCH"), "cifar10")
    opt, dataset, model, lr, ema_decay = config["setting"]
    args.opt = opt
    args.dataset = dataset
    args.model = model
    args.lr = lr
    args.ema_decay = ema_decay
    args.precision = config["precision"]
    args.accutype = config["accutype"]
    args.inverse = config["inverse"]
    with HiddenPrints():
        main(args)


def parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--sgd", action="store_true")
    parser.add_argument("--sgd-torch", action="store_true")
    parser.add_argument("--shampoo", action="store_true")
    parser.add_argument("--kfac-emp", action="store_true")
    parser.add_argument("--kfac-emp-local", action="store_true")
    parser.add_argument("--precision", type=str, choices=["std", "bf", "bf_as", "fp", "fp_as"])
    parser.add_argument("--accutype", type=str, choices=["std", "single", "double", "bf", "fp_s"])
    parser.add_argument("--inverse", type=str, choices=["lu", "cholesky"])
    return parser


if __name__ == "__main__":
    args = parser().parse_args()
    shutil.copytree(
        "/mnt/nfs/datasets/cifar-10-batches-py",
        os.path.join(os.getenv("HINADORI_LOCAL_SCRATCH"), "cifar10", "cifar-10-batches-py")
    )
    CIFAR100(root=os.path.join(os.getenv("HINADORI_LOCAL_SCRATCH"), "cifar10"), download=True)
    settings = []
    if args.sgd:
        settings += [
            ("sgd", "cifar10", "timm_vit_tiny_patch16_224", 3e-2, -1),
            ("sgd", "cifar100", "timm_vit_tiny_patch16_224", 3e-2, -1),
            ("sgd", "cifar10", "timm_resnet18", 1e-1, -1),
            ("sgd", "cifar100", "timm_resnet18", 1e-1, -1),
            ("sgd", "cifar10", "timm_vit_base_patch16_224", 1e-2, -1),
            ("sgd", "cifar100", "timm_vit_base_patch16_224", 1e-2, -1),
        ]
    if args.sgd_torch:
        settings += [
            ("sgd_torch", "cifar10", "timm_vit_tiny_patch16_224", 3e-2, -1),
            ("sgd_torch", "cifar100", "timm_vit_tiny_patch16_224", 3e-2, -1),
            ("sgd_torch", "cifar10", "timm_resnet18", 1e-1, -1),
            ("sgd_torch", "cifar100", "timm_resnet18", 1e-1, -1),
            ("sgd_torch", "cifar10", "timm_vit_base_patch16_224", 1e-2, -1),
            ("sgd_torch", "cifar100", "timm_vit_base_patch16_224", 1e-2, -1),
        ]
    if args.shampoo:
        settings += [
            ("shampoo", "cifar10", "timm_vit_tiny_patch16_224", 3e-2, -1),
            ("shampoo", "cifar100", "timm_vit_tiny_patch16_224", 3e-2, -1),
            ("shampoo", "cifar10", "timm_resnet18", 1e-1, -1),
            ("shampoo", "cifar100", "timm_resnet18", 1e-1, -1),
            ("shampoo", "cifar10", "timm_vit_base_patch16_224", 1e-2, -1),
            ("shampoo", "cifar100", "timm_vit_base_patch16_224", 1e-2, -1),
        ]
    if args.kfac_emp:
        settings += [
            ("kfac_emp", "cifar10", "timm_vit_tiny_patch16_224", 3e-2, 1e-3),
            ("kfac_emp", "cifar100", "timm_vit_tiny_patch16_224", 3e-2, 1e-3),
            ("kfac_emp", "cifar10", "timm_resnet18", 1e-1, 1e-1),
            ("kfac_emp", "cifar100", "timm_resnet18", 1e-1, 1e-1),
            ("kfac_emp", "cifar10", "timm_vit_base_patch16_224", 1e-2, 1e-4),
            ("kfac_emp", "cifar100", "timm_vit_base_patch16_224", 1e-2, 1e-4),
        ]
    if args.kfac_emp_local:
        settings += [
            ("kfac_emp", "cifar10", "timm_vit_tiny_patch16_224", 3e-1, -1),
            ("kfac_emp", "cifar100", "timm_vit_tiny_patch16_224", 3e-1, -1),
            ("kfac_emp", "cifar10", "timm_resnet18", 3e-2, -1),
            ("kfac_emp", "cifar100", "timm_resnet18", 3e-2, -1),
            ("kfac_emp", "cifar10", "timm_vit_base_patch16_224", 1e-1, -1),
            ("kfac_emp", "cifar100", "timm_vit_base_patch16_224", 1e-1, -1),
        ]

    search_space = {
        "setting": tune.grid_search(settings),
        "precision": tune.grid_search([args.precision]),
        "accutype": tune.grid_search([args.accutype]),
        "inverse": tune.grid_search([args.inverse]),
    }

    from train import get_args_parser, main
    tuner = tune.Tuner(
        tune.with_resources(train, {"gpu": 1}),
        run_config=RunConfig(verbose=0, name="icml"),
        param_space=search_space,
    )

    results = tuner.fit()
