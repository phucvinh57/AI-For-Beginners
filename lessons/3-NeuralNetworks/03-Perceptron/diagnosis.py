"""Plots for looking at what the trained perceptrons get wrong."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def show_samples(
    images: NDArray[np.uint8],
    actual: NDArray[np.uint8],
    guessed: NDArray[np.int64],
    indices: list[int],
    rows: int,
    cols: int,
) -> None:
    """A grid of individual test digits, titled `actual -> guess`."""
    _, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.2 * rows))
    for ax, i in zip(axes.ravel(), indices):
        hit = guessed[i] == actual[i]
        print(f"#{i}: actual {actual[i]}, guess {guessed[i]}{'' if hit else '  <-- wrong'}")

        ax.imshow(images[i].reshape(28, 28), cmap="gray")
        ax.set_title(f"{actual[i]} -> {guessed[i]}", color="green" if hit else "red")
        ax.axis("off")

    correct = int((guessed[indices] == actual[indices]).sum())
    print(f"{correct}/{len(indices)} correct")
    plt.tight_layout()
    plt.show()


def pca_analysis(
    images: NDArray[np.uint8],
    actual: NDArray[np.uint8],
    guessed: NDArray[np.int64],
    positive_label: int,
    negative_label: int,
    ax: plt.Axes,
) -> None:
    """Fit PCA on just two digit classes and plot their 2-D projection, the way
    the notebook does, but with the misclassified images marked.

    If the two clouds are linearly separable in this projection (0 vs 1), the
    perceptron gets them right; where they overlap (2 vs 5), it fails.
    """
    pair = (actual == positive_label) | (actual == negative_label)
    mypca = PCA(n_components=2)
    mypca.fit(images[pair].astype(np.float64))

    for label, colour in ((positive_label, "b"), (negative_label, "r")):
        of_label = actual == label
        points = mypca.transform(images[of_label].astype(np.float64))
        wrong = guessed[of_label] != label
        ax.plot(*points[~wrong].T, colour + "o", alpha=0.25, markersize=4, label=f"{label}")
        # Failures on top, ringed in black so they stand out of their cloud.
        ax.plot(
            *points[wrong].T,
            colour + "o",
            markersize=6,
            markeredgecolor="k",
            label=f"{label} misclassified ({int(wrong.sum())})",
        )

    ax.set_title(f"{positive_label} vs {negative_label}")
    ax.legend(fontsize=8)


def worst_confusions(
    actual: NDArray[np.uint8], guessed: NDArray[np.int64], top: int
) -> list[tuple[int, int]]:
    """The digit pairs the perceptron mixes up most, counting both directions."""
    counts: dict[tuple[int, int], int] = {}
    for a, g in zip(actual, guessed):
        if a != g:
            counts[(min(a, g), max(a, g))] = counts.get((min(a, g), max(a, g)), 0) + 1
    return sorted(counts, key=lambda pair: counts[pair], reverse=True)[:top]


def analyse_failures(
    images: NDArray[np.uint8],
    actual: NDArray[np.uint8],
    guessed: NDArray[np.int64],
) -> None:
    wrong = guessed != actual
    print(f"\nmisclassified {int(wrong.sum())}/{len(actual)} test images")
    for label in range(10):
        of_label = actual == label
        print(f"  {label}: {int((wrong & of_label).sum()):4d}/{int(of_label.sum()):4d} wrong")

    pairs = worst_confusions(actual, guessed, top=4)
    print("worst confusions:", ", ".join(f"{a} vs {b}" for a, b in pairs))

    _, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (a, b) in zip(axes.ravel(), pairs):
        pca_analysis(images, actual, guessed, a, b, ax)

    plt.tight_layout()
    plt.show()
