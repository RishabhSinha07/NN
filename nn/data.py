"""Fake fruit data for Part 1 of the plan.

Every fruit is three numbers: two measurements and an answer.

    spot size     X[:, 0]   float in 0..1
    spike length  X[:, 1]   float in 0..1
    label         y         0 = safe, 1 = poisonous

Two datasets, chosen for how they fail:

    "linear"  separable by a straight line that does NOT pass through the
              origin -- so weights alone cannot solve it (step 2), but
              weights plus biases can (step 3).

    "blob"    poisonous fruit occupy a round region in the middle, so no
              straight line separates them at all. Needs a hidden layer
              plus a non-linear activation (step 5).

Labels are exact: every point obeys the rule. If hand-tuning can't reach
100% in steps 3 and 5, the fault is the tuning, not the data.
"""

import numpy as np

SAFE = 0
POISONOUS = 1
CLASS_NAMES = ("safe", "poisonous")

FEATURE_NAMES = ("spot size", "spike length")

# --- the ground truth rules ------------------------------------------------
# These are the answer keys. The network never sees them; it only sees
# labelled points and has to reverse-engineer the shape.

# "linear": poisonous where  spot + spike > 1
# Picked for two reasons: it cuts the unit square exactly in half, so the
# classes are balanced, and the line misses the origin, which is what makes
# step 2 fail and step 3 succeed.
LINEAR_INTERCEPT = 1.0

# "blob": poisonous inside a circle, nudged slightly off-centre so the
# network can't exploit any symmetry. Radius chosen to cover ~45% of the
# square, keeping the classes roughly balanced.
BLOB_CENTRE = (0.47, 0.50)
BLOB_RADIUS = 0.38


def label_points(X, kind="linear"):
    """Apply a ground truth rule to points. Public so viz/ can label a grid
    and draw the true boundary alongside whatever the network learned."""
    X = np.asarray(X, dtype=float)
    spot, spike = X[:, 0], X[:, 1]

    if kind == "linear":
        poisonous = spot + spike > LINEAR_INTERCEPT
    elif kind == "blob":
        cx, cy = BLOB_CENTRE
        poisonous = (spot - cx) ** 2 + (spike - cy) ** 2 < BLOB_RADIUS**2
    else:
        raise ValueError(f"unknown kind {kind!r}, expected 'linear' or 'blob'")

    return np.where(poisonous, POISONOUS, SAFE).astype(int)


def make_fruit_data(n, kind="linear", seed=None):
    """Return (X, y) for n fruit scattered uniformly over the unit square.

    X has shape (n, 2), y has shape (n,). Pass a seed when you need the same
    fruit twice -- comparing step 8's training against step 13's is only
    meaningful on identical data.
    """
    rng = np.random.default_rng(seed)
    X = rng.random((n, 2))
    return X, label_points(X, kind)


def class_balance(y):
    """Fraction of each class, plus the accuracy of always guessing the
    majority. Any model that can't beat `baseline` has learned nothing."""
    counts = np.bincount(y, minlength=2)
    fractions = counts / len(y)
    return {
        "counts": dict(zip(CLASS_NAMES, counts.tolist())),
        "fractions": dict(zip(CLASS_NAMES, fractions.round(3).tolist())),
        "baseline": round(float(fractions.max()), 3),
    }

def one_hot(y, num_classes=2):
    """Convert integer labels to one-hot encoding."""
    y = np.asarray(y, dtype=int)
    one_hot = np.zeros((y.size, num_classes), dtype=float)
    one_hot[np.arange(y.size), y] = 1.0
    return one_hot

if __name__ == "__main__":
    for kind in ("linear", "blob"):
        X, y = make_fruit_data(200, kind, seed=1)
        print(f"{kind:7}  X {X.shape} {X.dtype}   y {y.shape} {y.dtype}")
        print(f"         {class_balance(y)}")
