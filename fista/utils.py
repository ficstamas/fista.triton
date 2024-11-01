import random
import torch


def create_dictionary(input_shape: tuple[int, int]) -> torch.Tensor:
    return torch.rand(input_shape, dtype=torch.float32, requires_grad=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
