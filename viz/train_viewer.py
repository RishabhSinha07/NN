"""Step 8 -- watch it learn.

    .venv/bin/python viz/train_viewer.py
    .venv/bin/python viz/train_viewer.py --shape 2 5 2 --dataset blob
    .venv/bin/python viz/train_viewer.py --gif          # animated gif
    .venv/bin/python viz/train_viewer.py --save         # final frame only

Left: the decision boundary, redrawn after every gradient step. Right: cost
and accuracy over time, on separate axes because they are different
quantities measured in different units -- sharing one would be a lie about
scale.

Cost falls smoothly. Accuracy lurches, plateaus, and sometimes drops while
cost is still improving. That disagreement is step 6's point, now visible
over a whole training run rather than a single nudge.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
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
    one_hot,
)

import style  # noqa: E402
from boundary_viewer import build_network  # noqa: E402

GRID = 110           # coarser than the slider viewer -- this redraws every frame
N_POINTS = 200
DATA_SEED = 1


def grid_axes(n=GRID):
    axis = np.linspace(0, 1, n)
    gx, gy = np.meshgrid(axis, axis)
    return gx, gy, np.stack([gx.ravel(), gy.ravel()], axis=1)


class Trainer:
    def __init__(self, shape, dataset="blob", learn_rate=10.0, dark=False):
        self.p = style.use(dark)
        self.shape = list(shape)
        self.gx, self.gy, self.grid = grid_axes()
        self.running = False

        self.fig = plt.figure(figsize=(11.6, 6.2))
        self.ax = self.fig.add_axes([0.05, 0.26, 0.42, 0.62])
        self.ax_cost = self.fig.add_axes([0.58, 0.60, 0.38, 0.28])
        self.ax_acc = self.fig.add_axes([0.58, 0.20, 0.38, 0.28])

        self._build_plot()
        self._build_curves()
        self._build_controls(learn_rate)
        self.reset(dataset)

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
        self.title = self.ax.set_title("", loc="left", fontsize=10)
        handles = [
            Line2D([], [], linestyle="none", marker=style.MARKERS[n],
                   markerfacecolor=p[n], markeredgecolor=p["surface"],
                   markersize=7, label=n) for n in CLASS_NAMES
        ] + [
            Line2D([], [], color=p["text"], linewidth=2, label="learned"),
            Line2D([], [], color=p["truth"], linewidth=1.5,
                   linestyle=(0, (5, 4)), label="true"),
        ]
        self.fig.legend(handles=handles, loc="upper left",
                        bbox_to_anchor=(0.05, 0.99), ncol=4, columnspacing=1.3,
                        handletextpad=0.5, fontsize=9)

    def _build_curves(self):
        p = self.p
        self.cost_line, = self.ax_cost.plot([], [], color=p["text"], linewidth=2)
        self.ax_cost.set_title("cost", loc="left", fontsize=10)
        self.ax_cost.set_xlabel("")

        self.acc_line, = self.ax_acc.plot([], [], color=p["text"], linewidth=2)
        self.baseline_line = self.ax_acc.axhline(
            0.5, color=p["truth"], linewidth=1.2, linestyle=(0, (5, 4)))
        self.ax_acc.set_title("accuracy", loc="left", fontsize=10)
        self.ax_acc.set_xlabel("gradient steps")
        self.ax_acc.set_ylim(0, 1.02)
        self.ax_acc.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        self.ax_acc.set_yticklabels(["0", "25%", "50%", "75%", "100%"])

    def _widget_ax(self, rect):
        ax = self.fig.add_axes(rect)
        ax.grid(False)
        ax.set_facecolor(self.p["surface"])
        return ax

    def _build_controls(self, learn_rate):
        p = self.p
        ax = self._widget_ax([0.10, 0.11, 0.30, 0.03])
        self.lr = Slider(ax, "learn rate  ", 0.1, 40.0, valinit=learn_rate,
                         valstep=0.1, color=p["muted"], track_color=p["grid"])
        for t in (self.lr.label, self.lr.valtext):
            t.set_color(p["muted"])
            t.set_fontsize(9)

        radio_ax = self._widget_ax([0.05, 0.005, 0.11, 0.075])
        for side in radio_ax.spines.values():
            side.set_visible(False)
        self.radio = RadioButtons(radio_ax, ("linear", "blob"), active=1,
                                  activecolor=p["text"])
        for lbl in self.radio.labels:
            lbl.set_fontsize(9)
            lbl.set_color(p["text"])
        self.radio.on_clicked(self.reset)

        def button(rect, text, fn):
            b = Button(self._widget_ax(rect), text,
                       color=p["grid"], hovercolor=p["muted"])
            b.label.set_color(p["text"])
            b.label.set_fontsize(9)
            b.on_clicked(fn)
            return b

        self.run_btn = button([0.18, 0.02, 0.10, 0.05], "train", self.toggle)
        self.step_btn = button([0.30, 0.02, 0.09, 0.05], "step",
                               lambda _: self.train_step())
        self.reset_btn = button([0.41, 0.02, 0.09, 0.05], "reset",
                                lambda _: self.reset(self.dataset))

    # -- behaviour -------------------------------------------------------
    def reset(self, dataset):
        self.dataset = dataset
        self.net = build_network(self.shape)
        self.X, self.y = make_fruit_data(N_POINTS, dataset, seed=DATA_SEED)
        self.Y = one_hot(self.y, len(CLASS_NAMES))
        self.baseline = class_balance(self.y)["baseline"]
        self.costs, self.accs = [], []
        for label, name in enumerate(CLASS_NAMES):
            self.scatters[name].set_offsets(self.X[self.y == label])
        self.baseline_line.set_ydata([self.baseline, self.baseline])
        self._draw_truth()
        self.record()
        self.draw()

    def _draw_truth(self):
        if self.truth is not None:
            for a in self.truth:
                a.remove()
        common = dict(color=self.p["truth"], linewidth=1.5,
                      linestyle=(0, (5, 4)), zorder=2)
        if self.dataset == "linear":
            x = np.array([0.0, 1.0])
            self.truth = self.ax.plot(x, LINEAR_INTERCEPT - x, **common)
        else:
            c = plt.Circle(BLOB_CENTRE, BLOB_RADIUS, fill=False, **common)
            self.ax.add_patch(c)
            self.truth = [c]

    def record(self):
        self.costs.append(self.net.average_cost(self.X, self.Y))
        self.accs.append(float(np.mean(
            [self.net.classify(x) for x in self.X] == self.y)))

    def train_step(self):
        self.net.learn(self.X, self.Y, learning_rate=self.lr.val)
        self.record()
        self.draw()

    def toggle(self, _):
        self.running = not self.running
        self.run_btn.label.set_text("pause" if self.running else "train")
        self.fig.canvas.draw_idle()

    def tick(self, _frame):
        if self.running:
            self.train_step()
        return ()

    def draw(self):
        out = np.array([self.net.calculate_output(pt) for pt in self.grid])
        margin = (out[:, 0] - out[:, 1]).reshape(GRID, GRID)
        self.shading.set_data((margin < 0).astype(float))
        if self.boundary is not None:
            self.boundary.remove()
        self.boundary = self.ax.contour(
            self.gx, self.gy, margin, levels=[0],
            colors=[self.p["text"]], linewidths=2, zorder=3)

        steps = np.arange(len(self.costs))
        self.cost_line.set_data(steps, self.costs)
        self.acc_line.set_data(steps, self.accs)
        for ax in (self.ax_cost, self.ax_acc):
            ax.set_xlim(0, max(len(self.costs), 10))
        self.ax_cost.set_ylim(0, max(self.costs) * 1.08 + 1e-9)

        shape = " → ".join(str(n) for n in self.shape)
        self.title.set_text(
            f"{shape}   {self.net.num_parameters()} parameters   "
            f"step {len(self.costs) - 1}      "
            f"cost {self.costs[-1]:.5f}   accuracy {self.accs[-1]:5.1%}"
        )
        self.fig.canvas.draw_idle()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", type=int, nargs="+", default=[2, 5, 2])
    ap.add_argument("--dataset", choices=("linear", "blob"), default="blob")
    ap.add_argument("--learn-rate", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=200,
                    help="steps to run for --save / --gif")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--dark", action="store_true")
    args = ap.parse_args()

    if args.save or args.gif:
        plt.switch_backend("Agg")
    t = Trainer(args.shape, args.dataset, args.learn_rate, dark=args.dark)
    name = "training-" + "-".join(str(n) for n in args.shape) + f"-{args.dataset}"

    if args.gif:
        every = max(1, args.steps // 120)      # ~120 frames whatever the length
        def frame(i):
            for _ in range(every):
                t.net.learn(t.X, t.Y, learning_rate=args.learn_rate)
            t.record()
            t.draw()
            return ()
        anim = FuncAnimation(t.fig, frame, frames=args.steps // every,
                             interval=50, blit=False)
        style.OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = style.OUT_DIR / f"{name}.gif"
        anim.save(path, writer=PillowWriter(fps=20))
        print(path)
    elif args.save:
        for _ in range(args.steps):
            t.net.learn(t.X, t.Y, learning_rate=args.learn_rate)
            t.record()
        t.draw()
        print(style.save(t.fig, name, dark=args.dark))
    else:
        t.anim = FuncAnimation(t.fig, t.tick, interval=30,
                               blit=False, cache_frame_data=False)
        plt.show()


if __name__ == "__main__":
    main()
