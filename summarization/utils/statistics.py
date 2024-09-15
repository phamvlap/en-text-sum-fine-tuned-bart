import numpy as np

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy, Precision, Recall, FBetaScore


class Statistics:
    def __init__(
        self,
        vocab_size: int,
        num_batchs: int = 0,
        loss: float = 0.0,
        preds: list[np.ndarray] | None = None,
        targets: list[np.ndarray] | None = None,
        average: str = "weighted",
        beta: float = 0.5,
        ignore_index: int = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.num_batchs = num_batchs
        self.loss = loss
        self.preds = preds if preds is not None else []
        self.targets = targets if targets is not None else []
        self.average = average
        self.beta = beta

        if ignore_index is not None:
            raise ValueError("ignore_index must be provided")
        self.ignore_index = ignore_index

    def update(
        self,
        loss: float,
        pred: np.ndarray | Tensor,
        target: np.ndarray | Tensor,
    ) -> None:
        self.loss += loss
        self.num_batchs += 1

        if isinstance(pred, Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, Tensor):
            target = target.detach().cpu().numpy()

        pred = pred.ravel()
        target = target.ravel()

        self.preds.append(pred)
        self.targets.append(target)

    def _compute(self) -> dict:
        loss = self.loss / self.num_batchs

        preds = np.concatenate(self.preds)
        targets = np.concatenate(self.targets)

        accuracy, recall, precision, f1_beta = calc_accuracy_recall_precision_f1beta(
            vocab_size=self.vocab_size,
            preds=preds,
            targets=targets,
            average=self.average,
            beta=self.beta,
            ignore_index=self.ignore_index,
        )

        return {
            "loss": loss,
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "f1_beta": f1_beta,
        }

    def write_to_tensorboard(
        self,
        writer: SummaryWriter,
        mode: str,
        step: int,
    ) -> None:
        stats = self._compute()

        for metric_key, metric_value in stats.item():
            writer.add_scalar(
                f"{mode}/{metric_key}",
                metric_value,
                step,
            )

    def reset(self) -> None:
        self.num_batchs = 0
        self.loss = 0.0
        self.preds = []
        self.targets = []


"""
accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
f1 = 2 * (precision * recall) / (precision + recall)
f1_beta = (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)
"""


def calc_accuracy_recall_precision_f1beta(
    vocab_size: int,
    preds: np.ndarray,
    targets: np.ndarray,
    ignore_index: int,
    average: str = "weighted",
    beta: float = 0.5,
) -> tuple:
    accuracy = Accuracy(
        task="multiclass",
        average=average,
        num_classes=vocab_size,
        ignore_index=ignore_index,
    )
    recall = Recall(
        task="multiclass",
        average=average,
        num_classes=vocab_size,
        ignore_index=ignore_index,
    )
    precision = Precision(
        task="multiclass",
        average=average,
        num_classes=vocab_size,
        ignore_index=ignore_index,
    )
    f1_beta = FBetaScore(
        task="multiclass",
        average=average,
        num_classes=vocab_size,
        ignore_index=ignore_index,
        beta=beta,
    )

    accuracy_score = accuracy(preds, targets)
    recall_score = recall(preds, targets)
    precision_score = precision(preds, targets)
    f1_beta_score = f1_beta(preds, targets)

    return accuracy_score, recall_score, precision_score, f1_beta_score
