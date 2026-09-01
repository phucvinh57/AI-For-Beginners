"""The neural network framework built up in `OwnFramework.ipynb`, as a module.

Layers own their parameters and their gradients; `Net` chains them and pushes
the gradient back through in reverse. Everything runs on plain NumPy.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from common.mnist import read_mnist
np.random.seed(0)

Array = NDArray[np.floating]


class Layer(ABC):
    """A step in the network: maps inputs forward, gradients backward."""

    @abstractmethod
    def forward(self, x: Array, /) -> Array:
        """Compute this layer's output for a batch of inputs."""

    @abstractmethod
    def backward(self, dy: Array, /) -> Array:
        """Turn the gradient w.r.t. our output into the gradient w.r.t. our input."""


class UpdatableLayer(Layer):
    """A layer with parameters of its own, so a training step can change it."""

    @abstractmethod
    def update(self, learning_rate: float, /) -> None:
        """Take one gradient-descent step on the parameters."""


class Linear(UpdatableLayer):
    """Fully connected layer: z = x @ W.T + b."""

    def __init__(self, nin: int, nout: int) -> None:
        # std of 1/sqrt(nin) keeps the output variance near the input's,
        # so activations neither explode nor vanish as layers stack up.
        self.W: Array = np.random.normal(0, 1.0 / np.sqrt(nin), (nout, nin))
        self.b: Array = np.zeros((1, nout))
        self.dW: Array = np.zeros_like(self.W)
        self.db: Array = np.zeros_like(self.b)
        self.x: Array  # set by forward, needed by backward

    def forward(self, x: Array) -> Array:
        self.x = x
        return np.dot(x, self.W.T) + self.b

    def backward(self, dz: Array) -> Array:
        dx = np.dot(dz, self.W)
        self.dW = np.dot(dz.T, self.x)
        self.db = dz.sum(axis=0, keepdims=True)
        return dx

    def update(self, lr: float) -> None:
        self.W -= lr * self.dW
        self.b -= lr * self.db


class Softmax(Layer):
    """Normalises scores into a probability distribution over the classes."""

    def __init__(self) -> None:
        self.p: Array  # cached output, reused by backward

    def forward(self, z: Array) -> Array:
        # subtracting the row max is a no-op mathematically but stops exp overflowing
        zmax = z.max(axis=1, keepdims=True)
        expz = np.exp(z - zmax)
        self.p = expz / expz.sum(axis=1, keepdims=True)
        return self.p

    def backward(self, dp: Array) -> Array:
        pdp = self.p * dp
        return pdp - self.p * pdp.sum(axis=1, keepdims=True)


class Tanh(Layer):
    """Non-linearity, without which stacked Linear layers collapse into one."""

    def __init__(self) -> None:
        self.y: Array

    def forward(self, x: Array) -> Array:
        self.y = np.tanh(x)
        return self.y

    def backward(self, dy: Array) -> Array:
        return (1.0 - self.y**2) * dy


class CrossEntropyLoss:
    """Negative log-likelihood of the true class, averaged over the batch.

    Not a `Layer`: `forward` needs the labels as well as the predictions, so it
    sits outside the `Net` rather than being appended to it.
    """

    def __init__(self) -> None:
        self.p: Array
        self.y: NDArray[np.integer]

    def forward(self, p: Array, y: NDArray[np.integer]) -> float:
        self.p = p
        self.y = y
        p_of_y = p[np.arange(len(y)), y]
        return float(-np.log(p_of_y).mean())

    def backward(self, loss: float) -> Array:
        # d(loss)/dp: only the true-class entry of each row is non-zero.
        dlog_softmax = np.zeros_like(self.p)
        dlog_softmax[np.arange(len(self.y)), self.y] -= 1.0 / len(self.y)
        return dlog_softmax / self.p


class Net:
    """A stack of layers, run forward in order and backward in reverse."""

    def __init__(self) -> None:
        self.layers: list[Layer] = []

    def add(self, layer: Layer) -> None:
        self.layers.append(layer)

    def forward(self, x: Array) -> Array:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dz: Array) -> Array:
        for layer in self.layers[::-1]:
            dz = layer.backward(dz)
        return dz

    def update(self, learning_rate: float) -> None:
        for layer in self.layers:
            if isinstance(layer, UpdatableLayer):
                layer.update(learning_rate)


def get_loss_acc(
    net: Net, x: Array, y: NDArray[np.integer], loss: CrossEntropyLoss
) -> tuple[float, float]:
    p = net.forward(x)
    return loss.forward(p, y), float((np.argmax(p, axis=1) == y).mean())


def train_epoch(
    net: Net,
    train_x: Array,
    train_labels: NDArray[np.integer],
    loss: CrossEntropyLoss,
    batch_size: int = 4,
    learning_rate: float = 0.1,
) -> None:
    for i in range(0, len(train_x), batch_size):
        xb = train_x[i : i + batch_size]
        yb = train_labels[i : i + batch_size]

        p = net.forward(xb)
        l = loss.forward(p, yb)
        net.backward(loss.backward(l))
        net.update(learning_rate)


def train(
    net: Net,
    train_x: Array,
    train_labels: NDArray[np.integer],
    test_x: Array,
    test_labels: NDArray[np.integer],
    n_epoch: int = 5,
    batch_size: int = 64,
    learning_rate: float = 0.1,
) -> None:
    loss = CrossEntropyLoss()
    l, acc = get_loss_acc(net, train_x, train_labels, loss)
    print(f"Initial       -> train loss={l:.4f} acc={acc:.4f}")

    for epoch in range(1, n_epoch + 1):
        train_epoch(net, train_x, train_labels, loss, batch_size, learning_rate)
        l, acc = get_loss_acc(net, train_x, train_labels, loss)
        vl, vacc = get_loss_acc(net, test_x, test_labels, loss)
        print(
            f"Epoch {epoch:2d}      -> train loss={l:.4f} acc={acc:.4f} | "
            f"test loss={vl:.4f} acc={vacc:.4f}"
        )


def main() -> None:
    mnist = read_mnist()
    # pixels arrive as 0-255 uint8; scale to 0-1 floats so the first layer's
    # weighted sums stay in a range tanh and softmax can work with
    train_x = mnist.train.features.astype(np.float32) / 255.0
    test_x = mnist.test.features.astype(np.float32) / 255.0
    train_labels = mnist.train.labels.astype(np.int64)
    test_labels = mnist.test.labels.astype(np.int64)

    net = Net()
    net.add(Linear(784, 100))
    net.add(Tanh())
    net.add(Linear(100, 10))
    net.add(Softmax())

    train(net, train_x, train_labels, test_x, test_labels)


if __name__ == "__main__":
    main()
