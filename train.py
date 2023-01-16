import argparse
import datetime
import os
import random
import time
import warnings

import numpy as np
import timm
import torch
import torchvision
import wandb
from ray import tune
from timm.data.transforms_factory import create_transform
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder


def train_one_epoch(
    model, optimizer, grad_maker, loss_func, data_loader,
    device="cuda", clip_grad_norm=0, use_wandb=False, print_freq=10
):
    model.train()
    end_time = time.time()

    for i, (image, target) in enumerate(data_loader):
        start_time = time.time()
        image, target = image.to(device), target.to(device)

        core_time1 = time.time()
        optimizer.zero_grad()
        dummy_y = grad_maker.setup_model_call(model, image)
        grad_maker.setup_loss_call(loss_func, dummy_y, target)
        output, loss = grad_maker.forward_and_backward()
        if clip_grad_norm:
            norm = nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        core_time2 = time.time()

        with torch.no_grad():
            acc = torch.sum(torch.argmax(output, dim=1) == target) / len(target) * 100
        if i % print_freq == 0:
            print(
                f"[{i}/{len(data_loader)}]\t loss: {loss:.4f}\t acc: {acc:.3f}%\t"
                f"time: {time.time() - end_time:.3f}\t data_time: {start_time - end_time:.3f}"
            )
        if use_wandb:
            log = {
                "loss": loss, "lr": optimizer.param_groups[0]["lr"], "acc": acc, "norm": norm,
                "total_time": time.time() - end_time, "data_time": start_time - end_time, "iter_time": core_time2 - core_time1
            }
            wandb.log(log)

        end_time = time.time()


def evaluate(model, criterion, data_loader, device="cuda"):
    model.eval()
    target_list = []
    output_list = []

    with torch.inference_mode():
        for image, target in data_loader:
            image = image.to(device, non_blocking=True)
            target_list.append(target.to(device, non_blocking=True))
            output_list.append(model(image))

        target = torch.cat(target_list)
        output = torch.cat(output_list)
        loss = criterion(output, target)
        acc = torch.sum(torch.argmax(output, dim=1) == target) / len(target) * 100
    return loss, acc.cpu()


def main(args):
    print(args)

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    os.environ["precision"] = args.precision
    os.environ["accutype"] = args.accutype
    import asdl
    from asdl import FISHER_MC, FISHER_EMP

    gpu = torch.tensor(0).cuda().device
    torch.cuda.reset_peak_memory_stats(gpu)

    # Initialization
    if args.ignore_warning:
        warnings.filterwarnings("ignore")
    if args.wandb:
        wandb.init(
            mode=args.wandb_mode, project=args.wandb_project, tags=args.wandb_tag, config=args
        )
    device = torch.device(args.device)
    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    # Data
    train_transform = create_transform(
        args.train_input_size, is_training=True,
        interpolation=args.interpolation, auto_augment=args.auto_augment
    )
    val_transform = create_transform(
        args.val_input_size, interpolation=args.interpolation, crop_pct=args.val_crop_pct
    )

    if args.dataset == "cifar10":
        dataset = CIFAR10(root=args.data_path, transform=train_transform, download=True)
        dataset_test = CIFAR10(root=args.data_path, train=False, transform=val_transform)
    if args.dataset == "cifar100":
        dataset = CIFAR100(root=args.data_path, transform=train_transform, download=True)
        dataset_test = CIFAR100(root=args.data_path, train=False, transform=val_transform)
    elif args.dataset == "imagenet":
        dataset = ImageFolder(args.traindir, transform=train_transform)
        dataset_test = ImageFolder(args.valdir, transform=val_transform)
    num_classes = len(dataset.classes)

    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=True
    )
    data_loader_test = DataLoader(
        dataset_test, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True
    )


    # Model
    print("Creating model")
    if args.model.startswith("timm_"):
        model = timm.create_model(
            args.model.replace("timm_", ""), pretrained=args.pretrained, num_classes=num_classes
        ) # Add the img_size kwarg if finetune ViT on different image size
    else:
        model = torchvision.models.__dict__[args.model](
            pretrained=args.pretrained, num_classes=num_classes
        )
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Optimizer
    parameters = model.parameters()
    opt_name = args.opt.lower()
    if args.momentum == 0:
        args.nesterov = False
    if opt_name == "rmsprop":
        optimizer = optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    elif opt_name in ("sgd", "kfac_mc", "kfac_emp", "shampoo", "smw_ngd"):
        optimizer = optim.SGD(
            parameters, lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay, nesterov=args.nesterov
        )

    ignore_modules = None
    if args.ignore_norm_layer:
        ignore_modules = [nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm]
    config = asdl.PreconditioningConfig(
        data_size=args.batch_size,
        damping=args.damping,
        curvature_upd_interval=args.cov_update_freq,
        preconditioner_upd_interval=args.inv_update_freq,
        ema_decay=args.ema_decay,
        ignore_modules=ignore_modules,
    )
    # if opt_name == "kfac_mc":
    #     grad_maker = asdl.KfacGradientMaker(model, config, fisher_type=FISHER_MC)
    if opt_name == "kfac_emp":
        grad_maker = asdl.KfacGradientMaker(model, config, fisher_type=FISHER_EMP)
    elif opt_name == "shampoo":
        grad_maker = asdl.ShampooGradientMaker(model, config)
    elif opt_name == "smw_ngd":
        grad_maker = asdl.SmwEmpNaturalGradientMaker(model, config)
    elif opt_name == "sgd":
        grad_maker = asdl.GradientMaker(model)

    # Learning rate scheduler
    main_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr * args.lr_eta_min
    )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. "
                "Only linear and constant are supported."
            )
        lr_scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, main_lr_scheduler],
            milestones=[args.lr_warmup_epochs],
        )
    else:
        lr_scheduler = main_lr_scheduler

    print("Start training")
    start_time = time.time()
    acc_list = []
    for epoch in range(args.epochs):
        train_one_epoch(
            model, optimizer, grad_maker, criterion, data_loader,
            device, args.clip_grad_norm, args.wandb, args.print_freq,
        )
        lr_scheduler.step()
        loss, acc = evaluate(model, criterion, data_loader_test, device)
        acc_list.append(acc)
        print(f"Epoch {epoch} acc: {acc:.3f}\t loss: {loss:.4f}")
        if args.ray:
            tune.report(mean_accuracy=max(acc_list))
        if args.wandb:
            log = {"epoch": epoch, "val_acc": acc, "val_loss": loss}
            wandb.log(log)

    if args.wandb:
        wandb.run.summary["peak_gpu_memory"] = torch.cuda.max_memory_allocated(gpu)
        wandb.run.summary["best_acc"] = max(acc_list)
        wandb.finish()
    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(
        f"Training time: {total_time_str}, Peak GPU memory: {torch.cuda.max_memory_allocated(gpu)}"
    )


def get_args_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="cifar10", type=str)
    # When using Ray, it should be an absolute path
    parser.add_argument("--data-path", default="data", type=str)
    # To use a timm model, add "timm_" before the timm model name, e.g. timm_deit_tiny_patch16_224
    parser.add_argument("--model", default="timm_vit_tiny_patch16_224", type=str)
    parser.add_argument("--pretrained", default=True, dest="pretrained", action="store_true")
    parser.add_argument("--ray", action="store_true")
    parser.add_argument("--use-deterministic-algorithms", action="store_true")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-mode", default="online", type=str)
    parser.add_argument("--wandb-project", type=str)
    parser.add_argument("--wandb-tag", default=None, type=str)

    parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu", "mps"])
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--print-freq", default=50, type=int)
    parser.add_argument("--ignore-warning", action="store_true")

    opt_choices = ["sgd", "kfac_emp", "shampoo", "smw_ngd"]
    parser.add_argument("--opt", default="sgd", type=str, choices=opt_choices)
    parser.add_argument("--lr", default=3e-3, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--nesterov", default=True, action="store_true")
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--lr-scheduler", default="cosineannealinglr", type=str)
    parser.add_argument("--lr-eta-min", default=0, type=float)
    parser.add_argument("--lr-warmup-epochs", default=0, type=int)
    parser.add_argument("--lr-warmup-method", default="linear", type=str)
    parser.add_argument("--lr-warmup-decay", default=0.1, type=float)  # First epoch lr decay
    parser.add_argument("--clip-grad-norm", default=10, type=float)
    # K-FAC
    parser.add_argument("--cov-update-freq", type=int, default=100)
    parser.add_argument("--inv-update-freq", type=int, default=100)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--ema-decay", type=float, default=-1)
    parser.add_argument("--ignore-norm-layer", default=True, action="store_true")

    parser.add_argument("--interpolation", default="bilinear", type=str)
    parser.add_argument("--val-input-size", default=224, type=int)
    parser.add_argument("--val-crop-pct", default=1, type=int)
    parser.add_argument("--train-input-size", default=224, type=int)
    parser.add_argument("--auto-augment", default="rand-m9-mstd0.5-inc1", type=str)

    parser.add_argument("--precision", type=str, choices=["std", "bf", "bf_as", "fp", "fp_as"])
    parser.add_argument("--accutype", type=str, choices=["std", "single", "bf", "fp_s"])
    return parser


if __name__ == "__main__":
    main(get_args_parser().parse_args())
