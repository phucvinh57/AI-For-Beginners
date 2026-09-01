"""Loading the MNIST digits shipped in this repo's `data/` directory.

Shared by every lesson that trains on MNIST, so the pickle layout and the
uint8 pixel convention are defined in exactly one place.
"""

from __future__ import annotations

import gzip
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# common/ sits next to data/ at the repository root.
DEFAULT_MNIST_PATH = Path(__file__).resolve().parents[1] / "data" / "mnist.pkl.gz"


@dataclass
class MnistData:
    @dataclass
    class Data:
        features: NDArray[np.uint8]
        labels: NDArray[np.uint8]

    train: Data
    test: Data


def _to_pixels(x: NDArray[np.number]) -> NDArray[np.uint8]:
    """Normalise a split to 0-255 uint8, whether it arrives scaled or not."""
    return (x * 255).round().astype(np.uint8) if x.max() <= 1.0 else x.astype(np.uint8)


def read_mnist(path: Path | str = DEFAULT_MNIST_PATH) -> MnistData:
    with gzip.open(path) as digit_files:
        MNIST = pickle.load(digit_files, encoding="latin1")

    (mnist_train_x, mnist_train_y), _, (mnist_test_x, mnist_test_y) = MNIST

    return MnistData(
        train=MnistData.Data(
            features=_to_pixels(mnist_train_x), labels=_to_pixels(mnist_train_y)
        ),
        test=MnistData.Data(
            features=_to_pixels(mnist_test_x), labels=_to_pixels(mnist_test_y)
        ),
    )
