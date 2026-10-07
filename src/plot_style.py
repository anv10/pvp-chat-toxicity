"""Shared matplotlib style so every figure in the report looks consistent."""
import matplotlib.pyplot as plt

# One fixed colour per class, used in every chart (never reassigned by rank).
CLASS_COLORS = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a"}
SOURCE_COLORS = {"lol": "#2a78d6", "dota": "#eb6834"}
SEQUENTIAL_CMAP = "Blues"  # confusion matrices: one hue, light -> dark


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#888888",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e5e5e5",
        "grid.linewidth": 0.8,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "legend.frameon": False,
    })
