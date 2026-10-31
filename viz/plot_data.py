"""Step 1 -- look at the fruit.

Scatters both datasets side by side with the ground truth rule drawn over
them, so you can see what the network will be asked to reverse-engineer.

    .venv/bin/python viz/plot_data.py

Writes viz/out/fruit-data.png and fruit-data-dark.png.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nn.data import (  # noqa: E402
    BLOB_CENTRE,
    BLOB_RADIUS,
    CLASS_NAMES,
    FEATURE_NAMES,
    LINEAR_INTERCEPT,
    POISONOUS,
    SAFE,
    class_balance,
    make_fruit_data,
)

import style  # noqa: E402

PANELS = (
    ("linear", "one straight line separates them"),
    ("blob", "no straight line can"),
)


def draw_truth(ax, kind, colour):
    """Draw the ground truth rule -- the answer key the network never sees."""
    common = dict(color=colour, linewidth=1.5, linestyle=(0, (5, 4)), zorder=2)
    if kind == "linear":
        x = np.array([0.0, 1.0])
        ax.plot(x, LINEAR_INTERCEPT - x, **common)
    else:
        ax.add_patch(Circle(BLOB_CENTRE, BLOB_RADIUS, fill=False, **common))


def scatter_panel(ax, X, y, p):
    for label, name in enumerate(CLASS_NAMES):
        pts = X[y == label]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            # triangles read smaller than circles at equal area; nudge for parity
            s=52 if name == "poisonous" else 42,
            marker=style.MARKERS[name],
            facecolor=p[name],
            edgecolor=p["surface"],
            linewidth=1.0,
            zorder=3,
        )
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")  # the blob is a circle; don't let it look elliptical
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_xlabel(FEATURE_NAMES[0])


def figure(n=200, seed=1, dark=False):
    p = style.use(dark)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.3))

    for ax, (kind, subtitle) in zip(axes, PANELS):
        X, y = make_fruit_data(n, kind, seed=seed)
        scatter_panel(ax, X, y, p)
        draw_truth(ax, kind, p["truth"])
        ax.set_title(f"{kind}\n{subtitle}", loc="left")

    axes[0].set_ylabel(FEATURE_NAMES[1])

    handles = [
        Line2D(
            [], [], linestyle="none", marker=style.MARKERS[name],
            markerfacecolor=p[name], markeredgecolor=p["surface"],
            markersize=7, label=name,
        )
        for name in CLASS_NAMES
    ] + [
        Line2D([], [], color=p["truth"], linewidth=1.5,
               linestyle=(0, (5, 4)), label="true boundary")
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.11, 1.02),
               ncol=3, columnspacing=1.6, handletextpad=0.5)

    fig.suptitle(
        f"{n} fictitious fruit per dataset",
        x=0.11, y=1.10, ha="left", fontsize=12,
    )
    return fig


if __name__ == "__main__":
    for dark in (False, True):
        print(style.save(figure(dark=dark), "fruit-data", dark=dark))

    for kind, _ in PANELS:
        _, y = make_fruit_data(200, kind, seed=1)
        b = class_balance(y)
        print(f"{kind:7} {b['fractions']}  majority-guess baseline {b['baseline']}")
