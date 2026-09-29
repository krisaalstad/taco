import numpy as np


def percplot(ax, x, Y, c, tr, lst="-", ec=None):
    """Shade a percentile range (port of percplot.m).

    Y may be (2, N), (N, 2), (3, N) or (N, 3); with three rows/columns the
    first and last are used as the lower and upper bounds.
    Unlike MATLAB's alpha(tr), the transparency applies to the face only, so
    the edge stays opaque (which matches how the MATLAB figures look).
    """
    x = np.asarray(x).ravel()
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 2 and Y.shape[1] == 3:
        Y = Y[:, [0, 2]]
    elif Y.ndim == 2 and Y.shape[0] == 3:
        Y = Y[[0, 2], :]
    if Y.shape[0] != 2:
        Y = Y.T
        if Y.shape[0] != 2:
            raise ValueError("Expected Y to contain a percentile range: "
                             "size(Y)=2(3)xN or Nx2(3)")

    xf = np.concatenate([x, x[::-1]])
    Yf = np.concatenate([Y[1], Y[0][::-1]])
    lw = 0.5 if np.sum(c) == 3 else 1.0
    (pp,) = ax.fill(xf, Yf, facecolor=(*c, tr), edgecolor=ec if ec is not None else c,
                    linestyle=lst, linewidth=lw)
    return pp
