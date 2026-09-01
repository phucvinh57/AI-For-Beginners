from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from numpy.typing import NDArray
import random
from itertools import cycle
from common.mnist import MnistData, read_mnist
from .diagnosis import analyse_failures, show_samples

def get_train_images(
    data: MnistData.Data, label: int
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    pos_imgs: list[NDArray[np.uint8]] = []
    neg_imgs: list[NDArray[np.uint8]] = []
    for i, img in enumerate(data.features):
        if data.labels[i] == label:
            pos_imgs.append(img)
        else:
            neg_imgs.append(img)
    return np.asarray(pos_imgs), np.asarray(neg_imgs)


def train(
    positive_examples: NDArray[np.uint8],
    negative_examples: NDArray[np.uint8],
    learning_rate: float = 0.2,
) -> NDArray[np.float64]:
    # Should be 28^2 = 784
    num_dimensions = positive_examples.shape[1]
    weights: NDArray[np.float64] = np.zeros((num_dimensions, 1))

    # Positives are the minority class, so cycle them while walking every
    # negative once: each image is seen, and updates stay balanced.
    for pos, neg in zip(cycle(positive_examples), negative_examples):
        z = np.dot(pos, weights)
        if z < 0:
            weights += learning_rate * pos.reshape(weights.shape)
        z = np.dot(neg, weights)
        if z >= 0:
            weights -= learning_rate * neg.reshape(weights.shape)

    return weights

WEIGHTS_PATH = Path(__file__).resolve().with_name("weights.json")


def save_weights(weights: dict[int, NDArray[np.float64]], path: Path) -> None:
    payload = {str(label): w.ravel().tolist() for label, w in weights.items()}
    path.write_text(json.dumps(payload))


def load_weights(path: Path) -> dict[int, NDArray[np.float64]]:
    payload: dict[str, list[float]] = json.loads(path.read_text())
    return {
        int(label): np.asarray(w, dtype=np.float64).reshape(-1, 1)
        for label, w in payload.items()
    }


def predict(
    image: NDArray[np.uint8], weights: dict[int, NDArray[np.float64]]
) -> tuple[int, dict[int, float]]:
    scores = {label: float(np.dot(image, w).item()) for label, w in weights.items()}
    return max(scores, key=lambda label: scores[label]), scores


def predict_all(
    images: NDArray[np.uint8], weights: dict[int, NDArray[np.float64]]
) -> NDArray[np.int64]:
    labels = sorted(weights)
    stacked = np.hstack([weights[label] for label in labels])
    scores = images.astype(np.float64) @ stacked
    return np.asarray(labels)[scores.argmax(axis=1)]


def main() -> None:
    mnist = read_mnist()

    # Load weights
    trained: dict[int, NDArray[np.float64]]
    if WEIGHTS_PATH.exists():
        trained = load_weights(WEIGHTS_PATH)
    else:
        trained = {}
        for i in range(0, 10):
            pos_images, neg_images = get_train_images(mnist.train, i)
            trained[i] = train(pos_images, neg_images)
        save_weights(trained, WEIGHTS_PATH)

    guessed = predict_all(mnist.test.features, trained)

    rows, cols = 3, 5
    indices = random.sample(range(len(mnist.test.features)), rows * cols)
    show_samples(mnist.test.features, mnist.test.labels, guessed, indices, rows, cols)

    analyse_failures(mnist.test.features, mnist.test.labels, guessed)


if __name__ == "__main__":
    main()
