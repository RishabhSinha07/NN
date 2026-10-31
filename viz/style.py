"""Shared plot styling for every animation in Part 1.

Palette is a validated categorical pair: blue for safe, orange for poisonous.
Both clear colourblind separation and contrast gates against each surface
(checked with the dataviz validator, all-pairs mode).

Class identity is carried by marker shape as well as hue -- circle for safe,
triangle for poisonous -- so the plots survive greyscale printing and the
harsher forms of colour vision deficiency.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent / "out"

LIGHT = {
    "surface": "#fcfcfb",
    "text": "#0b0b0b",
    "muted": "#52514e",
    "grid": "#e3e2de",
    "safe": "#2a78d6",
    "poisonous": "#eb6834",
    "truth": "#52514e",
}

DARK = {
    "surface": "#1a1a19",
    "text": "#ffffff",
    "muted": "#c3c2b7",
    "grid": "#33332f",
    "safe": "#3987e5",
    "poisonous": "#d95926",
    "truth": "#c3c2b7",
}

# Marker shape doubles as class identity. Edge ring is the surface colour so
# overlapping points stay individually readable in dense regions.
MARKERS = {"safe": "o", "poisonous": "^"}


def palette(dark=False):
    return DARK if dark else LIGHT


def use(dark=False):
    """Apply recessive, text-token-driven defaults. Returns the palette."""
    p = palette(dark)
    mpl.rcParams.update(
        {
            "figure.facecolor": p["surface"],
            "axes.facecolor": p["surface"],
            "savefig.facecolor": p["surface"],
            "text.color": p["text"],
            "axes.labelcolor": p["muted"],
            "axes.edgecolor": p["grid"],
            "axes.titlecolor": p["text"],
            "xtick.color": p["muted"],
            "ytick.color": p["muted"],
            "grid.color": p["grid"],
            "grid.linewidth": 0.8,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "normal",
            "axes.titlepad": 10,
            "legend.frameon": False,
            "figure.dpi": 130,
        }
    )
    return p


def save(fig, name, dark=False):
    """Write to viz/out/<name>.png (or <name>-dark.png). Returns the path."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}{'-dark' if dark else ''}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
