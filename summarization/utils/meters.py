import torch
import torch.distributed as dist

from typing import Optional

from .mix import is_torch_cuda_available


class AverageMeter:
    """
    A class for computing and storing average and current meters values
    """

    def __init__(
        self,
        name: str,
        value: int | float = 0.0,
        sum: int | float = 0.0,
        count: int = 0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.name = name
        self.value = value
        self.sum = sum
        self.count = count
        if device is None:
            device = torch.device("cuda" if is_torch_cuda_available() else "cpu")
        self.device = device

    def update(self, value: int | float, n: int = 1) -> None:
        self.value = value
        self.sum += value * n
        self.count += n

    def reset(self) -> None:
        self.value = 0.0
        self.sum = 0.0
        self.count = 0

    def all_reduce(self) -> None:
        data_to_reduce = torch.tensor(
            [self.sum, self.count],
            dtype=torch.float32,
            device=self.device,
        )
        dist.all_reduce(data_to_reduce, op=dist.ReduceOp.SUM)
        self.sum, self.count = data_to_reduce.tolist()

    @property
    def average(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0
