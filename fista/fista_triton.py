from typing import Optional, Literal

import torch
import tqdm
from numpy import sqrt

from fista.kernels.mm_sub import matmul_sub
from fista.kernels.fista_step import fista_step


def fista_triton(
        inp: torch.Tensor,
        dictionary: torch.Tensor,
        num_iterations: int,
        lambda_value: float,
        normalize_vectors: bool = True,
        eta: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
        verbose: bool = False,
        input_precision: Literal["tf32", "ieee"] = "tf32",
) -> Optional[torch.Tensor]:
    """
    FISTA algorithm
    :param inp: Input tensor (B, N, D)
    :param dictionary: Dictionary matrix (D, K)
    :param num_iterations: Number of iterations
    :param lambda_value: Lambda
    :param normalize_vectors: Normalize vectors before applying FISTA (helps convergence)
    :param eta: LR
    :param out: Output tensor (B, N, K)
    :param verbose: Verbose mode
    :param input_precision:
    :return:
    """
    input_shape = inp.shape
    inp = inp.view(-1, inp.shape[-1])
    if normalize_vectors:
        inp = torch.nn.functional.normalize(inp, p=2.0, dim=1)

    if eta is None:
        lips = torch.linalg.eigvalsh(dictionary.t() @ dictionary)[-1]
        eta = (1. / lips).detach().cpu().item()

    t_step = lambda t: (1 + sqrt(1 + 4 * t ** 2)) / 2

    tk = 1.0
    tk_n = t_step(tk)

    if out is None:
        alpha_hat = torch.zeros((inp.shape[0], dictionary.shape[1]), dtype=inp.dtype, device=inp.device)
    else:
        alpha_hat = out.view(-1, out.shape[-1])

    alpha_hat_y = torch.zeros_like(alpha_hat, dtype=inp.dtype, device=inp.device)

    progress = tqdm.trange(num_iterations, desc=f"FISTA", disable=not verbose)
    for _ in progress:
        residual = matmul_sub(alpha_hat_y, dictionary, inp, input_precision=input_precision)
        if verbose:
            f_error = torch.norm(residual, p="fro")
            progress.set_description(f"FISTA, fro: {f_error.detach().cpu().item():.2f}")
        fista_step(
            residual, dictionary, eta, lambda_value, tk, tk_n, alpha_hat, alpha_hat_y,
            input_precision=input_precision
        )
        tk = tk_n
        tk_n = t_step(tk)

    return alpha_hat.view(*input_shape[:-1], -1)



# if __name__ == '__main__':
#     a_ = torch.randn((512, 768), dtype=torch.float32, device='cuda')
#     b_ = torch.randn((768, 1024), dtype=torch.float32, device='cuda')
#
#     fista_triton(a_, b_)
