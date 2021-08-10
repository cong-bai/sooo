import torch
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from .core import extend
from .operations import OP_ACCUMULATE_GRADS, OP_BATCH_GRADS

__all__ = ['data_loader_gradient', 'batch_gradient', 'jacobian']


def data_loader_gradient(
    model,
    data_loader,
    loss_fn=None,
    has_accumulated=False,
    is_distributed=False,
    all_reduce=False,
    is_master=True,
    data_average=False
):
    if not has_accumulated:
        # accumulate gradient for an epoch
        assert loss_fn is not None, 'loss_fn must be specified when has_accumulated is False.'
        device = next(model.parameters()).device
        for data, target in data_loader:
            with extend(model, OP_ACCUMULATE_GRADS):
                data, target = data.to(device), target.to(device)
                model.zero_grad()
                loss = loss_fn(model(data), target)
                loss.backward()

    # take average of accumulated gradient
    scale = 1 / len(data_loader.dataset) if data_average else 1
    for param in model.parameters():
        if param.grad is None:
            continue
        param.grad = param.acc_grad.mul(scale)

    # reduce gradient
    if is_distributed:
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        # pack
        packed_tensor = parameters_to_vector(grads)
        # reduce
        if all_reduce:
            dist.all_reduce(packed_tensor)
        else:
            dist.reduce(packed_tensor, dst=0)
        # unpack
        if is_master or all_reduce:
            vector_to_parameters(
                packed_tensor.div_(dist.get_world_size()), grads
            )

        dist.barrier()


def batch_gradient(model, loss_fn, inputs, targets):
    with extend(model, OP_BATCH_GRADS):
        model.zero_grad()
        f = model(inputs)
        loss = loss_fn(f, targets)
        loss.backward()
    return f


def jacobian(model, x):
    f = model(x)
    assert f.ndim == 2  # (n, c)
    n, c = f.shape
    rst = []
    for i in range(c):
        with extend(model, OP_BATCH_GRADS):
            model.zero_grad()
            loss = f[:, i].sum()
            loss.backward()
        grads = [p.batch_grads for p in model.parameters() if p.requires_grad]
        grads = torch.hstack([g.view(n, -1) for g in grads])  # (n, p)
        rst.append(grads)
    return torch.vstack(rst)  # (cn, p)
