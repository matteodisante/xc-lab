"""One style for the figures of the crest report: fonts, a colour per kind of curve, line styles.

Colours are the Okabe-Ito palette, safe for the common forms of colour blindness. A kind keeps its colour and its
line style in every figure; the fitted H of a curve is a dashed horizontal line in the colour of the curve.
"""

import matplotlib.pyplot as plt

TEXT_WIDTH_IN = 6.3  # the width of the text of the report: figures are drawn at this width, so 8 pt stays 8 pt
KIND_COLOR = {"geo": "#000000", "model": "#0072B2", "straight": "#009E73", "random": "#CC79A7", "dyn": "#D55E00", "route": "#E69F00"}
KIND_LABEL = {"geo": "crests", "model": "crest model", "straight": "straight", "random": "random", "dyn": "flights", "route": "flights, circles removed"}
REGION_COLOR = {"alps": "#332288", "pyrenees": "#999933"}  # the two massifs in the crest figures: indigo and olive, unused by any curve kind
ACTIVE_COLOR = {"geo": "#000000", "dyn": "#0072B2", "route": "#E69F00"}  # the three curves of the active report: crests, flights in time, route
GUIDE = "#B8B8B8"  # thin reference lines
LW = 1.2
FIT_DASH = (0, (4, 2))


def rc(size=8):
    """Matplotlib settings with every text at `size` pt (figures are drawn at the final width, so the size is the printed one)."""
    return {"font.size": size, "axes.labelsize": size, "axes.titlesize": size, "legend.fontsize": size, "xtick.labelsize": size,
            "ytick.labelsize": size, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100,
            "savefig.bbox": "tight", "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
            "mathtext.default": "regular"}


def apply(size=8):
    plt.rcParams.update(rc(size))


def letter(ax, s, x=-0.14, y=1.04):
    """Panel letter in bold, above the upper left corner of the axes."""
    ax.text(x, y, s, transform=ax.transAxes, fontsize=plt.rcParams["font.size"] + 1, fontweight="bold", va="bottom", ha="left")
