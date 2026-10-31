"""Steps 2-5 -- drag every parameter, watch the boundary move.

    .venv/bin/python viz/boundary_viewer.py
    .venv/bin/python viz/boundary_viewer.py --shape 2 3 2
    .venv/bin/python viz/boundary_viewer.py --save

Sliders are generated from the network's shape, one per weight and bias. Type
a new shape into the box at the bottom ("2 3 2", "2 5 2", "2 4 4 2") and the
whole control panel rebuilds itself around the new network.

Background is shaded by whichever class wins, the solid line is where the two
outputs are exactly equal, the dashed line is the rule the data came from.

While no layer has an activation, the readout prints the single linear
equation the whole stack collapses to, however many layers deep it is. Once
an activation appears, that readout switches off and the line can bend.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, RadioButtons, Slider, TextBox

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nn.data import (  # noqa: E402
    BLOB_CENTRE,
    BLOB_RADIUS,
    CLASS_NAMES,
    FEATURE_NAMES,
    LINEAR_INTERCEPT,
    class_balance,
    make_fruit_data,
    one_hot,
)
from nn.network import Layer, Network  # noqa: E402

import style  # noqa: E402

GRID = 150
SLIDER_RANGE = 6.0
N_POINTS = 200
DATA_SEED = 1
INIT_SEED = 7
# Weights are scaled by 1/sqrt(fan_in). A node's weighted input is a sum of
# fan_in terms, so its spread grows like sqrt(fan_in) -- a fixed scale that
# suits 2 inputs drives a 16-input layer deep into the sigmoid's flat tails,
# where sigma' is ~1e-6 and nothing can learn. Dividing it out keeps every
# layer in the responsive range whatever its width.
INIT_GAIN = 3.0
MAX_SLIDERS = 28     # past this, hand-tuning is the thing step 6 replaces


def grid_axes(n=GRID):
    axis = np.linspace(0, 1, n)
    gx, gy = np.meshgrid(axis, axis)
    return gx, gy, np.stack([gx.ravel(), gy.ravel()], axis=1)


_GX, _GY, _GRID_PTS = grid_axes()


def build_network(shape, seed=INIT_SEED):
    """Network from node counts, e.g. [2, 3, 2].

    Seeded here so the viewer opens identically every time, and the output
    bias is then shifted so the tie line actually crosses the unit square.
    Without that shift most random inits classify the entire square as one
    class and there is no boundary to look at -- with a sigmoid squashing
    hidden values into a narrow band, the output bias difference easily
    dominates everything the weights do.
    """
    rng = np.random.default_rng(seed)
    layers = [Layer(a, b) for a, b in zip(shape, shape[1:])]
    for layer in layers:
        scale = INIT_GAIN / np.sqrt(layer.input_size)
        layer.weights = rng.normal(0, scale, layer.weights.shape).round(3)
        layer.biases = np.zeros_like(layer.biases)
    net = Network(layers)
    _centre_boundary(net)
    return net


def _poisonous_share(net, pts):
    out = np.array([net.calculate_output(p) for p in pts])
    return float(np.mean(out[:, 0] - out[:, 1] < 0))


def _centre_boundary(net, lo=-25.0, hi=25.0, steps=40):
    """Slide the output bias until roughly half the square goes each way.

    Bisection rather than arithmetic: with a sigmoid on the output layer the
    class split is not a linear function of the bias, so subtracting a mean
    margin does not centre anything. Searching on the split itself works
    whatever activation sits in front of it, and at any depth.
    """
    coarse = np.stack(
        [g.ravel() for g in np.meshgrid(np.linspace(0, 1, 40), np.linspace(0, 1, 40))],
        axis=1,
    )
    bias = net.layers[-1].biases
    base = float(bias[0])
    for _ in range(steps):
        mid = (lo + hi) / 2
        bias[0] = base + mid
        # raising bias[0] raises the "safe" output, so the poisonous share
        # DECREASES as mid increases -- the search runs the other way round
        if _poisonous_share(net, coarse) > 0.5:
            lo = mid          # too much poisonous -- push the safe output up
        else:
            hi = mid
    bias[0] = base + (lo + hi) / 2


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
    and return (a, b, c) of the boundary a*spot + b*spike + c = 0, or None
    once a layer has an activation function."""
    if any(getattr(layer, "activation", None) is not None for layer in net.layers):
        return None
    W = np.eye(net.layers[0].input_size)
    b = np.zeros(net.layers[0].input_size)
    for layer in net.layers:
        W = layer.weights @ W
        b = layer.weights @ b + layer.biases
    return W[0, 0] - W[1, 0], W[0, 1] - W[1, 1], b[0] - b[1]


class Viewer:
    def __init__(self, shape, dark=False):
        self.p = style.use(dark)
        self.fig = plt.figure(figsize=(11.4, 6.8))
        self.ax = self.fig.add_axes([0.05, 0.26, 0.44, 0.62])
        self.sliders = []
        self.headings = []
        self.note = None
        self.kind = "linear"
        self.prev = None          # (cost, accuracy) from the last redraw
        self.frozen = 0           # consecutive nudges with accuracy unmoved

        self._build_plot()
        self._build_controls()
        self.set_shape(shape)
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
            0.05, 0.155, "", fontsize=9, color=p["muted"], family="monospace",
        )
        self.delta_text = self.fig.text(
            0.05, 0.125, "", fontsize=9, color=p["muted"], family="monospace",
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

    def _build_controls(self):
        p = self.p
        self.shape_box = TextBox(
            self._widget_ax([0.115, 0.06, 0.11, 0.05]), "shape  ",
            initial="", color=p["surface"], hovercolor=p["grid"],
        )
        self.shape_box.label.set_color(p["muted"])
        self.shape_box.label.set_fontsize(9)
        self.shape_box.text_disp.set_color(p["text"])
        self.shape_box.text_disp.set_family("monospace")
        self.shape_box.on_submit(self.on_shape_submit)

        radio_ax = self._widget_ax([0.05, 0.005, 0.12, 0.05])
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

        self.reset_btn = button([0.25, 0.06, 0.09, 0.05], "reset",
                                lambda _: [s.reset() for s in self.sliders])
        self.rand_btn = button([0.36, 0.06, 0.11, 0.05], "randomize",
                               self.on_randomize)

    def _clear_sliders(self):
        for s in self.sliders:
            self.fig.delaxes(s.ax)
        for t in self.headings:
            t.remove()
        self.sliders, self.headings = [], []
        if self.note is not None:
            self.note.remove()
            self.note = None

    def _build_sliders(self):
        """One slider per parameter, grouped by layer. Deliberately neutral
        in colour -- blue and orange already mean safe and poisonous."""
        p = self.p
        if len(self.slots) > MAX_SLIDERS:
            self.note = self.fig.text(
                0.60, 0.5,
                f"{len(self.slots)} parameters.\n\n"
                f"Past about {MAX_SLIDERS} there is no\n"
                "tuning these by hand -- which is\n"
                "exactly what step 6 onwards is for.\n\n"
                "Use randomize, or pick a smaller shape.",
                fontsize=10, color=p["muted"], va="center",
            )
            return

        n_rows = len(self.slots) + len(self.net.layers)
        top, bottom = 0.94, 0.08
        pitch = min(0.044, (top - bottom) / max(n_rows, 1))

        y = top
        current_layer = None
        for label, layer, kind, idx in self.slots:
            if layer is not current_layer:
                current_layer = layer
                li = self.net.layers.index(layer) + 1
                self.headings.append(self.fig.text(
                    0.60, y - 0.012,
                    f"layer {li}   {layer.input_size} → {layer.output_size}",
                    fontsize=8.5, color=p["muted"]))
                y -= pitch
            ax = self._widget_ax([0.70, y, 0.20, min(0.024, pitch * 0.55)])
            value = layer.weights[idx] if kind == "w" else layer.biases[idx]
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

    # -- behaviour -------------------------------------------------------
    def set_shape(self, shape):
        self.shape = list(shape)
        self.net = build_network(self.shape)
        self.slots = param_slots(self.net)
        self._clear_sliders()
        self._build_sliders()
        self.prev, self.frozen = None, 0
        self.shape_box.set_val(" ".join(str(n) for n in self.shape))
        self.push_params()
        self.refresh()

    def on_shape_submit(self, text):
        try:
            shape = [int(t) for t in text.replace(",", " ").split()]
        except ValueError:
            return self.warn("shape must be whole numbers, e.g. 2 3 2")
        if len(shape) < 2 or any(n < 1 for n in shape):
            return self.warn("need at least two sizes, all positive")
        if shape[0] != 2 or shape[-1] != 2:
            return self.warn("must start and end at 2 -- 2 features, 2 classes")
        if shape != self.shape:
            self.set_shape(shape)

    def warn(self, msg):
        self.readout.set_text(msg)
        self.fig.canvas.draw_idle()

    def push_params(self):
        for s, (_, layer, kind, idx) in zip(self.sliders, self.slots):
            if kind == "w":
                layer.weights[idx] = s.val
            else:
                layer.biases[idx] = s.val

    def on_randomize(self, _):
        if self.sliders:
            rng = np.random.default_rng()
            for s in self.sliders:
                s.set_val(round(float(rng.normal(0, INIT_GAIN / np.sqrt(2))), 2))
        else:
            self.net = build_network(self.shape, seed=None)
            self.refresh()

    def load_data(self, kind):
        self.kind = kind
        self.prev, self.frozen = None, 0
        self.X, self.y = make_fruit_data(N_POINTS, kind, seed=DATA_SEED)
        self.expected = one_hot(self.y, len(CLASS_NAMES))
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

    def _show_deltas(self, cost, acc):
        """Side-by-side change in cost and in accuracy since the last redraw.

        This is step 6's whole argument: accuracy only moves when a fruit
        crosses the boundary, so most nudges leave it frozen and a learning
        algorithm reading it would get no feedback. Cost moves every time.
        """
        if self.prev is None:
            self.delta_text.set_text("")
            self.prev = (cost, acc)
            return
        d_cost, d_acc = cost - self.prev[0], acc - self.prev[1]
        self.frozen = 0 if d_acc else self.frozen + 1
        if d_acc:
            acc_note = f"accuracy moved {d_acc:+.1%}"
        elif self.frozen > 1:
            acc_note = f"accuracy frozen for {self.frozen} nudges"
        else:
            acc_note = "accuracy unchanged"
        self.delta_text.set_text(
            f"last nudge:  cost {d_cost:+.6f}      {acc_note}"
        )
        self.prev = (cost, acc)

    def refresh(self):
        if not hasattr(self, "X"):
            return
        out = np.array([self.net.calculate_output(pt) for pt in _GRID_PTS])
        margin = (out[:, 0] - out[:, 1]).reshape(GRID, GRID)
        self.shading.set_data((margin < 0).astype(float))

        if self.boundary is not None:
            self.boundary.remove()
        self.boundary = self.ax.contour(
            _GX, _GY, margin, levels=[0],
            colors=[self.p["text"]], linewidths=2, zorder=3,
        )

        predicted = np.array([self.net.classify(x) for x in self.X])
        acc = float(np.mean(predicted == self.y))
        cost = self.net.average_cost(self.X, self.expected)
        shape = " → ".join(str(n) for n in self.shape)
        self.readout.set_text(
            f"{shape}   {len(self.slots)} parameters        "
            f"cost {cost:.5f}       accuracy {acc:5.1%}     "
            f"baseline {self.baseline:.0%}"
        )
        self._show_deltas(cost, acc)

        line = collapsed_line(self.net)
        self.collapse_text.set_text(
            "no activation → the whole network collapses to one line:\n"
            f"   {line[0]:+.3f}·spot {line[1]:+.3f}·spike {line[2]:+.3f} = 0"
            if line is not None else ""
        )
        self.fig.canvas.draw_idle()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", type=int, nargs="+", default=[2, 3, 2],
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
