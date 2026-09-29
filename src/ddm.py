import numpy as np


def ddmvs(t, P, T, a, b, c, D0=None):
    """A very simple degree-day seasonal snow model (port of DDMvs.m).

    Runs a potentially large ensemble for a single point/cell.

    Parameters
    ----------
    t : (Nt,) array, time stamps (only the length is used).
    P : (Nt,) array, precipitation.
    T : (Nt,) array, air temperature [K].
    a, b, c : (Ne,) arrays, melt factor, temperature bias, snowfall multiplier.
    D0 : scalar or (Ne,) array, initial SWE. None/empty means 0.

    Returns
    -------
    D : (Nt, Ne) array of SWE.
    """
    # Hyperparameters
    Tr = 3.0     # Temperature threshold for rainfall (degrees C)
    Ts = -1.0    # Temperature threshold for snowfall (degrees C)
    Tm = 0.0     # Temperature threshold for snowmelt (degrees C)
    k2c = 273.15

    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    c = np.asarray(c, dtype=float).ravel()
    P = np.asarray(P, dtype=float).ravel()
    T = np.asarray(T, dtype=float).ravel()

    Nt = np.size(t)
    Ne = b.size
    if D0 is None or np.size(D0) == 0:
        Dold = 0.0
    else:
        Dold = np.asarray(D0, dtype=float)
    D = np.zeros((Nt, Ne))
    dTrs = Tr - Ts

    for j in range(Nt):
        Tj = T[j] - k2c - b
        ddj = Tj - Tm
        Mj = np.maximum(a * ddj, 0.0)
        frj = np.clip((Tj - Ts) / dTrs, 0.0, 1.0)
        fsj = 1.0 - frj
        Sj = (c * fsj) * P[j]
        NAj = Sj - Mj

        # Update
        Dj = np.maximum(Dold + NAj, 0.0)
        D[j, :] = Dj
        Dold = Dj

    return D
