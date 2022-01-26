import torch
from torch import nn
import torch.nn.functional as F

from .operation import Operation


class Linear(Operation):
    """
    module.weight: f_out x f_in
    module.bias: f_out x 1

    Argument shapes
    in_data: n x f_in
    out_grads: n x f_out
    """
    @staticmethod
    def batch_grads_weight(
        module: nn.Module, in_data: torch.Tensor, out_grads: torch.Tensor
    ):
        return torch.bmm(
            out_grads.unsqueeze(2), in_data.unsqueeze(1)
        )  # n x f_out x f_in

    @staticmethod
    def batch_grads_bias(module, out_grads):
        return out_grads

    @staticmethod
    def cov_diag_weight(module, in_data, out_grads):
        in_in = in_data.mul(in_data)  # n x f_in
        grad_grad = out_grads.mul(out_grads)  # n x f_out
        return torch.matmul(grad_grad.T, in_in)  # f_out x f_in

    @staticmethod
    def cov_diag_bias(module, out_grads):
        grad_grad = out_grads.mul(out_grads)  # n x f_out
        return grad_grad.sum(dim=0)  # f_out x 1

    @staticmethod
    def cov_kron_A(module, in_data):
        return torch.matmul(in_data.T, in_data)  # f_in x f_in

    @staticmethod
    def cov_kron_B(module, out_grads):
        return torch.matmul(out_grads.T, out_grads)  # f_out x f_out

    @staticmethod
    def cov_unit_wise(module, in_data, out_grads):
        n, f_in = in_data.shape[0], in_data.shape[1]
        in_in = torch.bmm(in_data.unsqueeze(2), in_data.unsqueeze(1)).view(n, -1)  # n x (f_in x f_in)
        grad_grad = out_grads.mul(out_grads)  # n x f_out
        return torch.matmul(grad_grad.T, in_in).view(-1, f_in, f_in)  # f_out x f_in x_fin

    @staticmethod
    def gram_A(module, in_data1, in_data2):
        return torch.matmul(in_data1, in_data2.T)  # n x n

    @staticmethod
    def gram_B(module, out_grads1, out_grads2):
        return torch.matmul(out_grads1, out_grads2.T)  # n x n

    @staticmethod
    def rfim_relu(module, in_data, out_data):
        nu = torch.sigmoid(out_data) ** 2  # n x f_out
        xxt = torch.einsum('bi,bj->bij', in_data, in_data)  # n x f_in x f_in
        return torch.einsum('bi,bjk->ijk', nu, xxt)  # f_out x f_in x f_in

    @staticmethod
    def rfim_softmax(module, in_data, out_data):
        # equivalent to fisher_exact_for_cross_entropy
        probs = F.softmax(out_data, dim=1)  # n x f_out
        ppt = torch.bmm(probs.unsqueeze(2), probs.unsqueeze(1))  # n x f_out x f_out
        diag_p = torch.stack([torch.diag(p) for p in probs], dim=0)  # n x f_out x f_out
        f = diag_p - ppt  # n x f_out x f_out
        xxt = torch.einsum('bi,bj->bij', in_data, in_data)  # n x f_in x f_in
        return torch.einsum('bij,bkl->ikjl', f, xxt)  # (f_out)(f_in)(f_out)(f_in)
