import time

import torch
import wandb
from torch import nn

def train_one_epoch_asdl(
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
                "total_time": time.time() - end_time, "data_time": start_time - end_time,
                "iter_time": core_time2 - core_time1
            }
            wandb.log(log)

        end_time = time.time()


def train_one_epoch_sgd_amp(
    model, optimizer, autocast_dtype, scaler, loss_func, data_loader,
    device="cuda", clip_grad_norm=0, use_wandb=False, print_freq=10
):
    model.train()
    end_time = time.time()

    for i, (image, target) in enumerate(data_loader):
        start_time = time.time()
        image, target = image.to(device), target.to(device)

        core_time1 = time.time()
        optimizer.zero_grad()
        with torch.autocast(
            device_type="cuda", dtype=autocast_dtype, enabled=(autocast_dtype is not None)
        ):
            output = model(image)
            loss = loss_func(output, target)
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        if clip_grad_norm:
            norm = nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
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
                "total_time": time.time() - end_time, "data_time": start_time - end_time,
                "iter_time": core_time2 - core_time1
            }
            wandb.log(log)

        end_time = time.time()
