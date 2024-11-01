import torch
import torch.nn as nn
from typing import Optional


"""
The PyTorch implementation:
https://github.com/yubeic/Sparse-Coding/
"""


def _fista(
        inp,
        dictionary,
        eta: float = None,
        num_iterations: int = 100,
        lambda_value: float = 0.1,
        pre_norm: bool = True,
        return_residual: bool = False
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    num_iter = num_iterations
    inp = inp.view(-1, inp.shape[-1])
    if pre_norm:
        inp = nn.functional.normalize(inp, p=2, dim=1)

    batch_size = inp.size(0)
    num_basis = dictionary.size(1)
    if eta is None:
        lips = torch.linalg.eigvalsh(dictionary.t() @ dictionary)[-1]
        eta = 1. / lips
    if isinstance(eta, float):
        eta = torch.tensor(eta, device=inp.device)

    tk_n = torch.tensor(1.0, device=inp.device)
    # tk = 1.
    # residual = torch.FloatTensor(inp.size()).fill_(0).to(inp.device)
    alpha_hat = torch.FloatTensor(batch_size, num_basis).fill_(0).to(inp.device)
    alpha_hat_y = torch.FloatTensor(batch_size, num_basis).fill_(0).to(inp.device)

    for t in range(num_iter):
        tk = tk_n
        tk_n = (1 + torch.sqrt(1 + 4 * tk ** 2)) / 2
        ahat_pre = alpha_hat
        residual = inp - (alpha_hat_y @ dictionary.T)
        alpha_hat_y = alpha_hat_y.add(eta * residual @ dictionary)
        alpha_hat = alpha_hat_y.sub(eta * lambda_value).clamp(min=0.)
        alpha_hat_y = alpha_hat.add(alpha_hat.sub(ahat_pre).mul((tk - 1) / tk_n))

    output = (alpha_hat, )

    residual: Optional[torch.Tensor] = None
    if return_residual:
        residual = inp - (alpha_hat @ dictionary.T)
    output = output + (residual, )

    return output


def fista_torch(
        inp,
        dictionary,
        eta: float = None,
        num_iterations: int = 100,
        lambda_value: float = 0.1,
        pre_norm: bool = True,
        return_residual: bool = False
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    return _fista(inp, dictionary, eta, num_iterations, lambda_value, pre_norm, return_residual)

@torch.jit.script
def _fista_jit_step(tk_n, alpha_hat, alpha_hat_y, inp, dictionary, eta, lambda_value):
    tk = tk_n
    tk_n_ = (1 + torch.sqrt(1 + 4 * tk ** 2)) / 2
    ahat_pre = alpha_hat
    residual = inp - (alpha_hat_y @ dictionary.T)
    alpha_hat_y = alpha_hat_y.add(eta * residual @ dictionary)
    alpha_hat = alpha_hat_y.sub(eta * lambda_value).clamp(min=0.)
    alpha_hat_y = alpha_hat.add(alpha_hat.sub(ahat_pre).mul((tk - 1) / tk_n_))
    return tk_n_, alpha_hat, alpha_hat_y


def _fista_jit_iter(inp, dictionary, eta, num_iterations, lambda_value):
    batch_size = inp.size(0)
    num_basis = dictionary.size(1)
    tk_n = torch.tensor(1.0, device=inp.device)
    alpha_hat = torch.zeros((batch_size, num_basis), device=inp.device)
    alpha_hat_y = torch.zeros((batch_size, num_basis), device=inp.device)
    lambda_value_ = torch.tensor(lambda_value, device=inp.device)
    for _ in range(num_iterations):
        tk_n, alpha_hat, alpha_hat_y = _fista_jit_step(tk_n, alpha_hat, alpha_hat_y, inp, dictionary, eta, lambda_value_)
    return alpha_hat


def fista_torch_script(
        inp,
        dictionary,
        eta: float = None,
        num_iterations: int = 100,
        lambda_value: float = 0.1,
        pre_norm: bool = True,
        return_residual: bool = False
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    inp = inp.view(-1, inp.shape[-1])
    if pre_norm:
        inp = nn.functional.normalize(inp, p=2.0, dim=1)

    if eta is None:
        lips = torch.linalg.eigvalsh(dictionary.t() @ dictionary)[-1]
        eta = 1. / lips
    if isinstance(eta, float):
        eta = torch.tensor(eta, device=inp.device)

    alpha_hat = _fista_jit_iter(inp, dictionary, eta, num_iterations, lambda_value)
    output = (alpha_hat,)

    residual: Optional[torch.Tensor] = None
    if return_residual:
        residual = inp - (alpha_hat @ dictionary.T)
    output = output + (residual,)

    return output
