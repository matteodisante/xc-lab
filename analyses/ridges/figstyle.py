"""One style for the figures of the crest report: fonts, a colour per kind of curve, line styles.

Colours are the Okabe-Ito palette, safe for the common forms of colour blindness. A kind keeps its colour and its
line style in every figure; the fitted H of a curve is a dashed horizontal line in the colour of the curve.
"""

import matplotlib.pyplot as plt

TEXT_WIDTH_IN = 6.3  # the width of the text of the report: figures are drawn at this width, so 8 pt stays 8 pt
KIND_COLOR = {"geo": "#000000", "model": "#0072B2", "straight": "#009E73", "random": "#CC79A7", "dyn": "#D55E00", "route": "#E69F00"}
KIND_LABEL = {"geo": "crests", "model": "crest model", "straight": "straight", "random": "random", "dyn": "flights", "route": "flights, circles removed"}
REGION_COLOR = {"alps": "#000000", "pyrenees": "#8C8C8C"}  # the crests of the two massifs: black and grey
GUIDE = "#B8B8B8"  # thin reference lines
LW = 1.2
FIT_DASH = (0, (4, 2))


def apply():
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 8, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100,
                         "savefig.bbox": "tight", "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "mathtext.default": "regular"})


def letter(ax, s, x=-0.14, y=1.04):
    """Panel letter in bold, above the upper left corner of the axes."""
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")
