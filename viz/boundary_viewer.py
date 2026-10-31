"""Step 2 -- drag the weights, watch the boundary move.

    .venv/bin/python viz/boundary_viewer.py
    .venv/bin/python viz/boundary_viewer.py --save    # static png instead

Four sliders, one per weight in a Network([Layer(2, 2)]). The background is
shaded by whichever class wins at that point, the solid line is where the two
outputs are exactly equal, and the dashed line is the rule the data was
actually generated from.

The thing to notice: the solid line pivots about the origin and will not leave
it, however you drag. No bias, no shifting. That is step 3's job.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, RadioButtons, Slider

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nn.data import (  # noqa: E402
    BLOB_CENTRE,
    BLOB_RADIUS,
    CLASS_NAMES,
    FEATURE_NAMES,
    LINEAR_INTERCEPT,
    class_balance,
    make_fruit_data,
)
from nn.network import Layer, Network  # noqa: E402

import style  # noqa: E402

GRID = 240          # background resolution
SLIDER_RANGE = 3.0
N_POINTS = 200
SEED = 1

# A deliberately mediocre starting point -- better than chance, clearly wrong.
START_WEIGHTS = np.array([[0.9, -0.4], [-0.6, 1.1]])


def grid_inputs(n=GRID):
    """Points covering the unit square, shaped (2, n*n) -- the layout
    Layer.calculate_output already accepts, so the whole background is one
    matrix multiply rather than n*n separate calls."""
    axis = np.linspace(0, 1, n)
    gx, gy = np.meshgrid(axis, axis)
    return gx, gy, np.stack([gx.ravel(), gy.ravel()])


class Viewer:
    def __init__(self, dark=False):
        self.p = style.use(dark)
        self.net = Network([Layer(2, 2)])
        self.net.layers[0].weights = START_WEIGHTS.copy()
        self.kind = "linear"
        self.gx, self.gy, self.grid = grid_inputs()

        self.fig = plt.figure(figsize=(10.2, 5.6))
        self.ax = self.fig.add_axes([0.07, 0.12, 0.52, 0.76])
        self._build_plot()
        self._build_widgets()
        self.load_data("linear")

    # -- layout ----------------------------------------------------------
    def _build_plot(self):
        p = self.p
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel(FEATURE_NAMES[0])
        self.ax.set_ylabel(FEATURE_NAMES[1])

        self.shading = self.ax.imshow(
            np.zeros((GRID, GRID)), origin="lower", extent=(0, 1, 0, 1),
            cmap=plt.matplotlib.colors.ListedColormap([p["safe"], p["poisonous"]]),
            alpha=0.16, vmin=0, vmax=1, interpolation="nearest", zorder=0,
        )
        self.boundary = None
        self.truth = None
        self.scatters = {}
        for name in CLASS_NAMES:
            self.scatters[name] = self.ax.scatter(
                [], [], s=52 if name == "poisonous" else 42,
                marker=style.MARKERS[name], facecolor=p[name],
                edgecolor=p["surface"], linewidth=1.0, zorder=4,
            )
        # the pivot the boundary can never escape without a bias
        self.ax.plot([0], [0], marker="o", markersize=5, color=p["truth"], zorder=5)
        self.ax.annotate(
            "pinned here", xy=(0.004, 0.004), xytext=(0.10, 0.11),
            color=p["muted"], fontsize=8.5, zorder=6,
            arrowprops=dict(arrowstyle="-", color=p["muted"], linewidth=0.9),
            bbox=dict(facecolor=p["surface"], edgecolor="none",
                      alpha=0.85, pad=2.0),
        )
        self.readout = self.ax.set_title("", loc="left", fontsize=10)

        handles = [
            Line2D([], [], linestyle="none", marker=style.MARKERS[n],
                   markerfacecolor=p[n], markeredgecolor=p["surface"],
                   markersize=7, label=n)
            for n in CLASS_NAMES
        ] + [
            Line2D([], [], color=p["text"], linewidth=2, label="learned boundary"),
            Line2D([], [], color=p["truth"], linewidth=1.5,
                   linestyle=(0, (5, 4)), label="true boundary"),
        ]
        self.fig.legend(handles=handles, loc="upper left",
                        bbox_to_anchor=(0.07, 0.99), ncol=4, columnspacing=1.4,
                        handletextpad=0.5, fontsize=9)

    def _widget_ax(self, rect):
        ax = self.fig.add_axes(rect)
        ax.grid(False)
        ax.set_facecolor(self.p["surface"])
        return ax

    def _build_widgets(self):
        p = self.p
        self.sliders = []
        labels = [
            ("w[0,0]", "spot size", "safe"),
            ("w[0,1]", "spike length", "safe"),
            ("w[1,0]", "spot size", "poisonous"),
            ("w[1,1]", "spike length", "poisonous"),
        ]
        for i, (name, src, dst) in enumerate(labels):
            ax = self._widget_ax([0.70, 0.74 - i * 0.085, 0.26, 0.035])
            s = Slider(ax, "", -SLIDER_RANGE, SLIDER_RANGE,
                       valinit=START_WEIGHTS.ravel()[i], valstep=0.01,
                       color=p[dst], track_color=p["grid"])
            s.valtext.set_color(p["muted"])
            s.valtext.set_fontsize(9)
            ax.set_title(f"{name}   {src} → {dst}", loc="left",
                         fontsize=8.5, color=p["muted"], pad=4)
            s.on_changed(self.on_slider)
            self.sliders.append(s)

        radio_ax = self._widget_ax([0.70, 0.26, 0.26, 0.13])
        radio_ax.set_title("dataset", loc="left", fontsize=8.5,
                           color=p["muted"], pad=4)
        for side in radio_ax.spines.values():
            side.set_visible(False)
        self.radio = RadioButtons(radio_ax, ("linear", "blob"),
                                  activecolor=p["text"])
        for lbl in self.radio.labels:
            lbl.set_fontsize(9)
            lbl.set_color(p["text"])
        self.radio.on_clicked(self.load_data)

        reset_ax = self._widget_ax([0.70, 0.14, 0.12, 0.05])
        self.reset = Button(reset_ax, "reset", color=p["grid"],
                            hovercolor=p["muted"])
        self.reset.label.set_color(p["text"])
        self.reset.label.set_fontsize(9)
        self.reset.on_clicked(self.on_reset)

    # -- behaviour -------------------------------------------------------
    def load_data(self, kind):
        self.kind = kind
        self.X, self.y = make_fruit_data(N_POINTS, kind, seed=SEED)
        self.baseline = class_balance(self.y)["baseline"]
        for label, name in enumerate(CLASS_NAMES):
            self.scatters[name].set_offsets(self.X[self.y == label])
        self._draw_truth()
        self.refresh()

    def _draw_truth(self):
        if self.truth is not None:
            for artist in self.truth:
                artist.remove()
        common = dict(color=self.p["truth"], linewidth=1.5,
                      linestyle=(0, (5, 4)), zorder=2)
        if self.kind == "linear":
            x = np.array([0.0, 1.0])
            self.truth = self.ax.plot(x, LINEAR_INTERCEPT - x, **common)
        else:
            circle = plt.Circle(BLOB_CENTRE, BLOB_RADIUS, fill=False, **common)
            self.ax.add_patch(circle)
            self.truth = [circle]

    def on_slider(self, _):
        self.net.layers[0].weights = np.array(
            [s.val for s in self.sliders]
        ).reshape(2, 2)
        self.refresh()

    def on_reset(self, _):
        for s in self.sliders:
            s.reset()

    def refresh(self):
        # One forward pass for the whole background grid.
        out = self.net.calculate_output(self.grid)       # (2, GRID*GRID)
        margin = out[0] - out[1]                         # >0 means safe wins
        self.shading.set_data((margin < 0).reshape(GRID, GRID).astype(float))

        if self.boundary is not None:
            self.boundary.remove()
        self.boundary = self.ax.contour(
            self.gx, self.gy, margin.reshape(GRID, GRID), levels=[0],
            colors=[self.p["text"]], linewidths=2, zorder=3,
        )

        predicted = np.argmax(self.net.calculate_output(self.X.T), axis=0)
        acc = float(np.mean(predicted == self.y))
        self.readout.set_text(
            f"accuracy {acc:6.1%}      always-guess-majority {self.baseline:.0%}"
        )
        self.fig.canvas.draw_idle()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true",
                    help="write a png instead of opening a window")
    ap.add_argument("--dark", action="store_true")
    args = ap.parse_args()

    if args.save:
        plt.switch_backend("Agg")
    v = Viewer(dark=args.dark)
    if args.save:
        print(style.save(v.fig, "boundary-viewer", dark=args.dark))
    else:
        plt.show()


if __name__ == "__main__":
    main()
