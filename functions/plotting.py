from typing import Iterable

import seaborn as sns

from matplotlib import pyplot as plt


def plot_losses(num_epochs: int, train_losses: Iterable[float], val_losses: Iterable[float]) -> None:
    plt.figure(figsize=(10, 5))

    epochs = range(1, num_epochs + 1)

    sns.lineplot(x=epochs, y=train_losses, label="Train Loss")
    sns.lineplot(x=epochs, y=val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train Loss vs Validation Loss")
    plt.legend()
    plt.show()
