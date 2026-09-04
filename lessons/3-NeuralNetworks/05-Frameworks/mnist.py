import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common.mnist import read_mnist, MnistData

torch.manual_seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_net() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(784, 100),
        nn.Tanh(),
        nn.Linear(100, 10),
    )


def to_tensors(mnist: MnistData) -> tuple[torch.Tensor, ...]:
    return (
        torch.tensor(mnist.train.features, dtype=torch.float32, device=DEVICE) / 255.0,
        torch.tensor(mnist.train.labels, dtype=torch.long, device=DEVICE),
        torch.tensor(mnist.test.features, dtype=torch.float32, device=DEVICE) / 255.0,
        torch.tensor(mnist.test.labels, dtype=torch.long, device=DEVICE),
    )


@torch.no_grad()
def get_loss_accuracy(
    net: nn.Module, x: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module
) -> tuple[float, float]:
    """Evaluate on the whole split at once, gradients switched off."""
    net.eval()
    z = net(x)
    accuracy = (z.argmax(dim=1) == y).float().mean()
    return float(loss_fn(z, y)), float(accuracy)


def train_epoch(
    net: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    net.train()
    for xb, yb in loader:
        loss = loss_fn(net(xb), yb)

        # The three lines that replace our whole Net.backward/Net.update pair:
        # clear last step's gradients, let autograd fill in new ones, then step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train(
    net: nn.Module,
    mnist: MnistData,
    n_epoch: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.1,
) -> None:
    train_x, train_labels, test_x, test_labels = to_tensors(mnist)
    loader = DataLoader(
        TensorDataset(train_x, train_labels), batch_size=batch_size, shuffle=False
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

    l, acc = get_loss_accuracy(net, train_x, train_labels, loss_fn)
    print(f"Initial       -> train loss={l:.4f} accuracy={acc:.4f}")

    for epoch in range(1, n_epoch + 1):
        train_epoch(net, loader, loss_fn, optimizer)
        l, acc = get_loss_accuracy(net, train_x, train_labels, loss_fn)
        vl, vacc = get_loss_accuracy(net, test_x, test_labels, loss_fn)
        print(
            f"Epoch {epoch:2d}      -> train loss={l:.4f} acc={acc:.4f} | "
            f"test loss={vl:.4f} acc={vacc:.4f}"
        )


def main() -> None:
    print(f"Training on {DEVICE}")
    mnist = read_mnist()
    net = build_net().to(DEVICE)
    train(net, mnist)


if __name__ == "__main__":
    main()
