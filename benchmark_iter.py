import triton
import torch
from fista.utils import set_seed
from fista import fista_torch, fista_torch_script, fista_triton


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=['I'],  # argument names to use as an x-axis for the plot
            x_vals=[i for i in range(10, 250, 10)],  # different possible values for `x_name`
            line_arg='provider',  # argument name whose value corresponds to a different line in the plot
            line_vals=['triton_tf32', 'triton_ieee', 'torch', 'torch_script'],  # possible values for `line_arg``
            line_names=[
                "Triton TF32",
                "Triton IEEE",
                "Torch",
                "Torch Script",
            ],  # label name for the lines
            styles=[('blue', '-'), ('green', '-'), ('red', '--'), ('yellow', '--')],  # line styles
            ylabel="Median (ms)",  # label name for the y-axis
            plot_name="fista_num_iter",  # name for the plot. Used also as a file name for saving the plot.
            args={},  # values for function arguments not in `x_names` and `y_name`
        )
    ]
)
def benchmark(I, provider, **kwargs):
    set_seed(0)

    inp = torch.randn((512, 768), dtype=torch.float32, device='cuda')
    dictionary = torch.randn((768, 3072), dtype=torch.float32, device='cuda')
    lips = torch.linalg.eigvalsh(dictionary.t() @ dictionary)[-1]
    eta = (1. / lips).detach().cpu().item()

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    warmups = 100
    runs = 100
    quantiles = [0.5, 0.2, 0.8]

    if provider == 'triton_tf32':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda : fista_triton(
                inp, dictionary, I, 0.1,
                eta=eta, normalize_vectors=True, verbose=False, input_precision="tf32"
            ),
            warmup=warmups, return_mode="median", rep=runs, quantiles=quantiles
        )
    if provider == 'triton_ieee':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda : fista_triton(
                inp, dictionary, I, 0.1,
                eta=eta, normalize_vectors=True, verbose=False, input_precision="ieee"
            ),
            warmup=warmups, return_mode="median", rep=runs, quantiles=quantiles
        )
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda : fista_torch(inp, dictionary, eta, I, 0.1, pre_norm=True),
            warmup=warmups, return_mode="median", rep=runs, quantiles=quantiles
        )
    if provider == 'torch_script':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda : fista_torch_script(inp, dictionary, eta, I, 0.1, pre_norm=True),
            warmup=warmups, return_mode="median", rep=runs, quantiles=quantiles
        )

    return ms, min_ms, max_ms


benchmark.run(show_plots=True, print_data=True, save_path="benchmark/")
