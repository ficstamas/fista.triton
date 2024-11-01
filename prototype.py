from fista.fista_triton import fista_triton
from fista.utils import set_seed
import torch


set_seed(0)
inp = torch.randn((512, 768), dtype=torch.float32, device='cuda')
dictionary = torch.randn((768, 3072), dtype=torch.float32, device='cuda')
lips = torch.linalg.eigvalsh(dictionary.t() @ dictionary)[-1]
eta = (1. / lips).detach().cpu().item()

fista_triton(
    inp, dictionary, 100, 0.1,
    eta=eta, normalize_vectors=True, verbose=False, input_precision="tf32"
)