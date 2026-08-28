from __future__ import annotations
from pathlib import Path
import gzip
import pickle
import numpy as np
from numpy.typing import NDArray
import pylab
from dataclasses import dataclass

@dataclass
class MnistData:
    @dataclass
    class Data:
        features: NDArray[np.uint8]
        labels: NDArray[np.uint8]

    train: Data
    test: Data

def read_mnist() -> MnistData:
    repo_root = Path(__file__).resolve().parents[3]
    mnist_path = repo_root / "data" / "mnist.pkl.gz"

    with gzip.open(mnist_path) as digit_files:
        MNIST = pickle.load(digit_files, encoding="latin1")

    (mnist_train_x, mnist_train_y), _, (mnist_test_x, mnist_test_y) = MNIST

    def to_pixels(x: NDArray[np.number]) -> NDArray[np.uint8]:
        return (
            (x * 255).round().astype(np.uint8) if x.max() <= 1.0 else x.astype(np.uint8)
        )

    return MnistData(
        train=MnistData.Data(features=to_pixels(mnist_train_x), labels=to_pixels(mnist_train_y)),
        test=MnistData.Data(features=to_pixels(mnist_test_x), labels=to_pixels(mnist_test_y))
    )


def main() -> None:
    mnist = read_mnist()

    features = mnist.train.features

    fig = pylab.figure(figsize=(10,10))
    for i in range(10):
        fig.add_subplot(1,10,i+1)
        pylab.imshow(features[i].reshape(28,28))
    pylab.show()

if __name__ == "__main__":
    main()
