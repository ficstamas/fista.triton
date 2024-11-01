from typing import Optional, Literal

import torch

import triton
import triton.language as tl
from .mm_sub import get_autotune_config


@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def fista_step_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr, d_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        stride_dm, stride_dn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        tk: tl.constexpr,
        tk_n: tl.constexpr,
        eta: tl.constexpr,
        lambda_value: tl.constexpr,
        input_precision: tl.constexpr = "tf32"
):
    """
    Kernel for computing a FISTA step, assuming the residual is already calculated

    :param a_ptr: residual
    :param b_ptr: dictionary
    :param c_ptr: alpha_hat
    :param d_ptr: alpha_hat_y

    Process:
    tk = tk
    tk_n = tk_n // which is precalculated as: (1 + sqrt(1 + 4 * tk ** 2)) / 2
    cache_alpha = alpha_hat

    residual = input - (alpha_hat_y @ dictionary.T)
    T = eta * residual @ dictionary
    alpha_hat_y = alpha_hat_y + T

    alpha_hat = (alpha_hat_y - (eta * lambda_value)).clamp(min=0.)

    alpha_hat_y = (alpha_hat - cache_alpha) * ((tk - 1) / tk_n_) + alpha_hat

    store: alpha_hat_y
    store: alpha_hat
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    offs_pm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_pn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_pm[:, None] + stride_cn * offs_pn[None, :]
    c_mask = (offs_pm[:, None] < M) & (offs_pn[None, :] < N)

    offs_d_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_d_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    d_ptrs = d_ptr + stride_dm * offs_d_m[:, None] + stride_dn * offs_d_n[None, :]
    d_mask = (offs_d_m[:, None] < M) & (offs_d_n[None, :] < N)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator, input_precision=input_precision)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    cache_alpha = tl.load(c_ptrs, mask=c_mask)
    alpha_hat_y = tl.load(d_ptrs, mask=d_mask)
    t = accumulator * eta
    alpha_hat_y += t
    alpha_hat_y = alpha_hat_y - (eta * lambda_value)
    alpha_hat = tl.clamp(alpha_hat_y, min=0.0, max=tl.max(alpha_hat_y))
    alpha_hat_y = (alpha_hat - cache_alpha) * ((tk - 1) / tk_n) + alpha_hat


    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    tl.store(c_ptrs, alpha_hat, mask=c_mask)
    tl.store(d_ptrs, alpha_hat_y, mask=d_mask)


# TODO dictionary transpose
def fista_step(
        input_tensor: torch.Tensor,
        dictionary: torch.Tensor,
        eta: float,
        lambda_value: float,
        tk: float,
        tk_n: float,
        alpha_hat: Optional[torch.Tensor] = None,
        alpha_hat_y: Optional[torch.Tensor] = None,
        input_precision: str = Literal["tf32", "ieee"]
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """
    Computes a single FISTA step
    :param input_tensor:
    :param dictionary:
    :param eta:
    :param lambda_value:
    :param tk:
    :param tk_n:
    :param alpha_hat:
    :param alpha_hat_y:
    :param input_precision:
    :return:
    """
    if alpha_hat is None:
        alpha_hat = torch.zeros((input_tensor.shape[0], dictionary.shape[0]), dtype=input_tensor.dtype, device=input_tensor.device)

    if alpha_hat_y is None:
        alpha_hat_y = torch.zeros_like(alpha_hat, dtype=input_tensor.dtype, device=input_tensor.device)


    assert input_tensor.shape[1] == dictionary.shape[0], "Incompatible dimensions"
    assert input_tensor.is_contiguous(), "Matrix A must be contiguous"
    M, K = input_tensor.shape
    K, N = dictionary.shape
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    fista_step_kernel[grid](
        input_tensor, dictionary, alpha_hat, alpha_hat_y,  #
        M, N, K,  #
        input_tensor.stride(0), input_tensor.stride(1),  #
        dictionary.stride(0), dictionary.stride(1),  #
        alpha_hat.stride(0), alpha_hat.stride(1),  #
        alpha_hat_y.stride(0), alpha_hat_y.stride(1),  #
        eta=eta,
        lambda_value=lambda_value,
        tk=tk,
        tk_n=tk_n,
        input_precision=input_precision
    )
    return alpha_hat, alpha_hat_y
