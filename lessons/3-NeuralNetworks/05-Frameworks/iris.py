import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

features, labels = load_iris(return_X_y=True)

# load_iris(return_X_y=True) drops the metadata, so name the columns/classes here
feature_names = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
class_names = ["setosa", "versicolor", "virginica"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_EPOCH = 50
BATCH_SIZE = 20
LR = 0.05

# Four features, 3 classes
torch.manual_seed(42)
net = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 3)).to(DEVICE)


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


x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)
x_train = torch.tensor(x_train, dtype=torch.float32, device=DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32, device=DEVICE)
y_train = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
y_test = torch.tensor(y_test, dtype=torch.long, device=DEVICE)
loader = DataLoader(
    TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True
)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=LR)

history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

l, acc = get_loss_accuracy(net, x_train, y_train, loss_fn)
print(f"Initial       -> train loss={l:.4f} accuracy={acc:.4f}")
for epoch in range(1, N_EPOCH + 1):
    train_epoch(net, loader, loss_fn, optimizer)
    l, acc = get_loss_accuracy(net, x_train, y_train, loss_fn)
    vl, vacc = get_loss_accuracy(net, x_test, y_test, loss_fn)
    history["train_loss"].append(l)
    history["train_acc"].append(acc)
    history["test_loss"].append(vl)
    history["test_acc"].append(vacc)
    print(
        f"Epoch {epoch:2d}      -> train loss={l:.4f} acc={acc:.4f} | "
        f"test loss={vl:.4f} acc={vacc:.4f}"
    )

epochs = range(1, N_EPOCH + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(epochs, history["train_loss"], label="train")
ax1.plot(epochs, history["test_loss"], label="test")
ax1.set_xlabel("epoch")
ax1.set_ylabel("loss")
ax1.legend()

ax2.plot(epochs, history["train_acc"], label="train")
ax2.plot(epochs, history["test_acc"], label="test")
ax2.set_xlabel("epoch")
ax2.set_ylabel("accuracy")
ax2.legend()

fig.tight_layout()
plt.show()
