"""Steps 2-3 -- drag the parameters, watch the boundary move.

    .venv/bin/python viz/boundary_viewer.py
    .venv/bin/python viz/boundary_viewer.py --save    # static png instead

Six sliders over a Network([Layer(2, 2)]): four weights and two biases. The
background is shaded by whichever class wins, the solid line is where the two
outputs are exactly equal, and the dashed line is the rule the data came from.

Step 2 (weights only, biases at 0): the solid line pivots about the origin and
will not leave it. Step 3: the biases let it translate, and the linear dataset
becomes solvable at 100%.
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

# Layer.calculate_output takes one point at a time, so the background costs
# GRID*GRID calls -- about 23 ms at 180, which still feels live when dragging.
GRID = 180
SLIDER_RANGE = 3.0
N_POINTS = 200
SEED = 1

# Weights give a visible diagonal through the origin; biases start at zero so
# the viewer opens on exactly the step 2 ceiling.
START_WEIGHTS = np.array([[0.9, -0.4], [-0.6, 1.1]])
START_BIASES = np.array([0.0, 0.0])

PARAMS = [
    ("w[0,0]", "spot size → safe", "safe"),
    ("w[0,1]", "spike length → safe", "safe"),
    ("w[1,0]", "spot size → poisonous", "poisonous"),
    ("w[1,1]", "spike length → poisonous", "poisonous"),
    ("b[0]", "bias → safe", "safe"),
    ("b[1]", "bias → poisonous", "poisonous"),
]


def grid_axes(n=GRID):
    axis = np.linspace(0, 1, n)
    gx, gy = np.meshgrid(axis, axis)
    return gx, gy, np.stack([gx.ravel(), gy.ravel()], axis=1)


class Viewer:
    def __init__(self, dark=False):
        self.p = style.use(dark)
        self.net = Network([Layer(2, 2)])
        self.layer = self.net.layers[0]
        self.kind = "linear"
        self.gx, self.gy, self.grid = grid_axes()

        self.fig = plt.figure(figsize=(10.2, 5.8))
        self.ax = self.fig.add_axes([0.07, 0.11, 0.52, 0.76])
        self._build_plot()
        self._build_widgets()
        self.push_params()
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
        for label, name in enumerate(CLASS_NAMES):
            self.scatters[name] = self.ax.scatter(
                [], [], s=52 if name == "poisonous" else 42,
                marker=style.MARKERS[name], facecolor=p[name],
                edgecolor=p["surface"], linewidth=1.0, zorder=4,
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
        start = np.concatenate([START_WEIGHTS.ravel(), START_BIASES])
        self.sliders = []
        for i, (name, desc, dst) in enumerate(PARAMS):
            y = 0.82 - i * 0.078 - (0.02 if i >= 4 else 0.0)  # gap before biases
            ax = self._widget_ax([0.70, y, 0.26, 0.032])
            s = Slider(ax, "", -SLIDER_RANGE, SLIDER_RANGE, valinit=start[i],
                       valstep=0.01, color=p[dst], track_color=p["grid"])
            s.valtext.set_color(p["muted"])
            s.valtext.set_fontsize(9)
            ax.set_title(f"{name}   {desc}", loc="left", fontsize=8.5,
                         color=p["muted"], pad=3)
            s.on_changed(self.on_slider)
            self.sliders.append(s)

        radio_ax = self._widget_ax([0.70, 0.19, 0.26, 0.12])
        radio_ax.set_title("dataset", loc="left", fontsize=8.5,
                           color=p["muted"], pad=3)
        for side in radio_ax.spines.values():
            side.set_visible(False)
        self.radio = RadioButtons(radio_ax, ("linear", "blob"),
                                  activecolor=p["text"])
        for lbl in self.radio.labels:
            lbl.set_fontsize(9)
            lbl.set_color(p["text"])
        self.radio.on_clicked(self.load_data)

        reset_ax = self._widget_ax([0.70, 0.08, 0.12, 0.048])
        self.reset = Button(reset_ax, "reset", color=p["grid"],
                            hovercolor=p["muted"])
        self.reset.label.set_color(p["text"])
        self.reset.label.set_fontsize(9)
        self.reset.on_clicked(lambda _: [s.reset() for s in self.sliders])

    # -- behaviour -------------------------------------------------------
    def push_params(self):
        vals = np.array([s.val for s in self.sliders])
        self.layer.weights = vals[:4].reshape(2, 2)
        self.layer.biases = vals[4:]

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
        self.push_params()
        self.refresh()

    def refresh(self):
        out = np.array([self.net.calculate_output(pt) for pt in self.grid])
        margin = (out[:, 0] - out[:, 1]).reshape(GRID, GRID)
        self.shading.set_data((margin < 0).astype(float))

        if self.boundary is not None:
            self.boundary.remove()
        self.boundary = self.ax.contour(
            self.gx, self.gy, margin, levels=[0],
            colors=[self.p["text"]], linewidths=2, zorder=3,
        )

        predicted = np.array([self.net.classify(x) for x in self.X])
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
