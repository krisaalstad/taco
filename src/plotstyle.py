import logging

import matplotlib as mpl

# Set to True to render text with a real LaTeX installation (slower, needs latex
# on the PATH). The default uses matplotlib's built-in mathtext with Computer
# Modern fonts, which looks close to MATLAB's 'Interpreter','Latex'.
USE_TEX = False


def set_style():
    # Silence harmless fontTools timestamp warnings when embedding cmr10 in PDFs.
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    mpl.rcParams.update({
        "text.usetex": USE_TEX,
        "font.family": "serif",
        "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "legend.fancybox": False,
        "legend.edgecolor": "black",
        "legend.framealpha": 1.0,
        "pdf.fonttype": 42,
    })


def style_axes(ax, fs, ticklength=4):
    """Rough equivalent of the repeated set(gca, ...) calls in the MATLAB scripts."""
    ax.tick_params(direction="out", width=2, length=ticklength, labelsize=0.8 * fs)
    for s in ax.spines.values():
        s.set_linewidth(2)
    ax.xaxis.get_offset_text().set_fontsize(0.8 * fs)
    ax.yaxis.get_offset_text().set_fontsize(0.8 * fs)
