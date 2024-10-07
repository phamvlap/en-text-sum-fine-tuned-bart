import random
import numpy as np
import torch

from .mix import is_torch_cuda_available


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if is_torch_cuda_available():
        torch.cuda.manual_seed_all(seed)
