"""Steps 2-5 -- drag every parameter, watch the boundary move.

    .venv/bin/python viz/boundary_viewer.py
    .venv/bin/python viz/boundary_viewer.py --shape 2 3 2
    .venv/bin/python viz/boundary_viewer.py --save

Sliders are generated from the network's actual shape, one per weight and
bias, so any layer sizes work. --shape takes node counts including the input
layer: "2 2" is a single layer, "2 3 2" has one hidden layer of 3 nodes.

Background is shaded by whichever class wins, the solid line is where the two
outputs are exactly equal, the dashed line is the rule the data came from.

While no activation function exists, the readout prints the single linear
equation the whole network collapses to -- however many layers you stack.
That is step 4's lesson, in the one place it cannot be argued with.
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

GRID = 150          # background resolution; costs GRID*GRID forward passes
SLIDER_RANGE = 3.0
N_POINTS = 200
DATA_SEED = 1
INIT_SEED = 7


def build_network(shape):
    """Network from node counts, e.g. [2, 3, 2]. Weights are seeded here so
    the viewer opens identically every time -- Layer's own init uses a fresh
    unseeded generator."""
    rng = np.random.default_rng(INIT_SEED)
    layers = [Layer(a, b) for a, b in zip(shape, shape[1:])]
    for layer in layers:
        layer.weights = rng.normal(0, 0.9, layer.weights.shape).round(2)
        layer.biases = rng.normal(0, 0.4, layer.biases.shape).round(2)
    return Network(layers)


def param_slots(net):
    """(label, layer, kind, index) for every tunable number in the network."""
    slots = []
    for li, layer in enumerate(net.layers, start=1):
        for j in range(layer.output_size):
            for k in range(layer.input_size):
                slots.append((f"W{li}[{j},{k}]", layer, "w", (j, k)))
        for j in range(layer.output_size):
            slots.append((f"b{li}[{j}]", layer, "b", (j,)))
    return slots


def collapsed_line(net):
    """With no activation, any stack of layers is one affine map. Compose it
    and return the (a, b, c) of the boundary a*spot + b*spike + c = 0.
    Returns None once a layer gains an activation function."""
    if any(getattr(layer, "activation", None) is not None for layer in net.layers):
        return None
    W = np.eye(net.layers[0].input_size)
    b = np.zeros(net.layers[0].input_size)
    for layer in net.layers:
        W = layer.weights @ W
        b = layer.weights @ b + layer.biases
    return W[0, 0] - W[1, 0], W[0, 1] - W[1, 1], b[0] - b[1]


def grid_axes(n=GRID):
    axis = np.linspace(0, 1, n)
    gx, gy = np.meshgrid(axis, axis)
    return gx, gy, np.stack([gx.ravel(), gy.ravel()], axis=1)


class Viewer:
    def __init__(self, shape, dark=False):
        self.p = style.use(dark)
        self.shape = shape
        self.net = build_network(shape)
        self.slots = param_slots(self.net)
        self.kind = "linear"
        self.gx, self.gy, self.grid = grid_axes()

        self.fig = plt.figure(figsize=(11.4, 6.8))
        self.ax = self.fig.add_axes([0.05, 0.22, 0.44, 0.66])
        self._build_plot()
        self._build_sliders()
        self._build_controls()
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
        for name in CLASS_NAMES:
            self.scatters[name] = self.ax.scatter(
                [], [], s=52 if name == "poisonous" else 42,
                marker=style.MARKERS[name], facecolor=p[name],
                edgecolor=p["surface"], linewidth=1.0, zorder=4,
            )
        self.readout = self.ax.set_title("", loc="left", fontsize=10)
        self.collapse_text = self.fig.text(
            0.05, 0.10, "", fontsize=9, color=p["muted"], family="monospace",
        )

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
                        bbox_to_anchor=(0.05, 0.99), ncol=4, columnspacing=1.4,
                        handletextpad=0.5, fontsize=9)

    def _widget_ax(self, rect):
        ax = self.fig.add_axes(rect)
        ax.grid(False)
        ax.set_facecolor(self.p["surface"])
        return ax

    def _build_sliders(self):
        """One slider per parameter, in a single column, grouped by layer.
        Sliders are deliberately neutral-coloured -- blue and orange mean
        'safe' and 'poisonous' in the plot and must not mean anything here."""
        p = self.p
        n_rows = len(self.slots) + len(self.net.layers)   # params + headings
        top, bottom = 0.94, 0.06
        pitch = min(0.044, (top - bottom) / max(n_rows, 1))

        self.sliders = []
        y = top
        current_layer = None
        for label, layer, kind, idx in self.slots:
            if layer is not current_layer:
                current_layer = layer
                li = self.net.layers.index(layer) + 1
                self.fig.text(0.60, y - 0.012,
                              f"layer {li}   {layer.input_size} → {layer.output_size}",
                              fontsize=8.5, color=p["muted"])
                y -= pitch
            ax = self._widget_ax([0.70, y, 0.20, min(0.024, pitch * 0.55)])
            value = (layer.weights[idx] if kind == "w" else layer.biases[idx])
            s = Slider(ax, label, -SLIDER_RANGE, SLIDER_RANGE,
                       valinit=float(value), valstep=0.01,
                       color=p["muted"], track_color=p["grid"])
            s.label.set_fontsize(8)
            s.label.set_color(p["muted"])
            s.label.set_family("monospace")
            s.valtext.set_fontsize(8)
            s.valtext.set_color(p["muted"])
            s.on_changed(self.on_slider)
            self.sliders.append(s)
            y -= pitch

    def _build_controls(self):
        p = self.p
        radio_ax = self._widget_ax([0.05, 0.005, 0.12, 0.080])
        for side in radio_ax.spines.values():
            side.set_visible(False)
        self.radio = RadioButtons(radio_ax, ("linear", "blob"),
                                  activecolor=p["text"])
        for lbl in self.radio.labels:
            lbl.set_fontsize(9)
            lbl.set_color(p["text"])
        self.radio.on_clicked(self.load_data)

        def button(rect, text, fn):
            ax = self._widget_ax(rect)
            b = Button(ax, text, color=p["grid"], hovercolor=p["muted"])
            b.label.set_color(p["text"])
            b.label.set_fontsize(9)
            b.on_clicked(fn)
            return b

        self.reset_btn = button([0.20, 0.02, 0.10, 0.048], "reset",
                                lambda _: [s.reset() for s in self.sliders])
        self.rand_btn = button([0.32, 0.02, 0.12, 0.048], "randomize",
                               self.on_randomize)

    # -- behaviour -------------------------------------------------------
    def push_params(self):
        for s, (_, layer, kind, idx) in zip(self.sliders, self.slots):
            if kind == "w":
                layer.weights[idx] = s.val
            else:
                layer.biases[idx] = s.val

    def on_randomize(self, _):
        rng = np.random.default_rng()
        for s in self.sliders:
            s.set_val(round(float(rng.normal(0, 0.9)), 2))

    def load_data(self, kind):
        self.kind = kind
        self.X, self.y = make_fruit_data(N_POINTS, kind, seed=DATA_SEED)
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
        shape = " → ".join(str(n) for n in self.shape)
        self.readout.set_text(
            f"{shape}   {len(self.slots)} parameters        "
            f"accuracy {acc:5.1%}     baseline {self.baseline:.0%}"
        )

        line = collapsed_line(self.net)
        self.collapse_text.set_text(
            "no activation → the whole network collapses to one line:\n"
            f"   {line[0]:+.3f}·spot {line[1]:+.3f}·spike {line[2]:+.3f} = 0"
            if line is not None else ""
        )
        self.fig.canvas.draw_idle()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", type=int, nargs="+", default=[2, 2],
                    help="node counts including the input layer, e.g. 2 3 2")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--dark", action="store_true")
    args = ap.parse_args()

    if args.save:
        plt.switch_backend("Agg")
    v = Viewer(args.shape, dark=args.dark)
    if args.save:
        name = "boundary-viewer-" + "-".join(str(n) for n in args.shape)
        print(style.save(v.fig, name, dark=args.dark))
    else:
        plt.show()


if __name__ == "__main__":
    main()
