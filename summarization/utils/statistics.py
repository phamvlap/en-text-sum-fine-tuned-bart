class Statistics:
    def __init__(
        self,
        num_batchs: int = 0,
        loss: float = 0.0,
    ) -> None:
        self.num_batchs = num_batchs
        self.loss = loss

    def update(self, loss: float) -> None:
        self.loss += loss
        self.num_batchs += 1

    def compute(self) -> dict[str, float]:
        loss = self.loss / self.num_batchs

        return {
            "loss": loss,
        }

    def reset(self) -> None:
        self.num_batchs = 0
        self.loss = 0.0
