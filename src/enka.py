import numpy as np


def enka(X, y, Yp, alpha, R, dostoch, rng=None, svthresh=0.99):
    """Efficient implementation of the ensemble Kalman analysis step (port of EnKA.m).

    Based on Evensen et al. (2022) and Emerick (2016).
    Assumes R is diagonal, given as a scalar, an (No,) vector, or an (No, No) matrix.

    Parameters
    ----------
    X  : (Ns, Ne) prior ensemble of states/parameters.
    y  : (No,) observations (a scalar is fine for No = 1).
    Yp : (No, Ne) predicted observations (a 1-D array is treated as No = 1).
    alpha : ES-MDA inflation coefficient.
    R  : observation error variance(s).
    dostoch : bool, stochastic EnKF if True, otherwise DEnKF.
    rng : numpy Generator, only used when dostoch is True.
    svthresh : cumulative singular-value fraction retained in the No > Ne branch.

    Returns
    -------
    Xu : (Ns, Ne) updated ensemble.
    """
    if rng is None:
        rng = np.random.default_rng()

    X = np.asarray(X, dtype=float)
    y = np.atleast_1d(np.asarray(y, dtype=float)).reshape(-1, 1)
    Yp = np.atleast_2d(np.asarray(Yp, dtype=float))
    No = y.shape[0]
    Ne = X.shape[1]

    R = np.asarray(R, dtype=float)
    if R.size == 1:
        r = np.full(No, R.item())
        Rm = np.diag(r)
    elif R.size == No:
        r = R.ravel()
        Rm = np.diag(r)
    elif R.shape == (No, No):
        r = np.diag(R).copy()
        Rm = R
    else:
        raise ValueError("Check R")

    Xm = X.mean(axis=1, keepdims=True)
    Xa = X - Xm
    Ypm = Yp.mean(axis=1, keepdims=True)
    Ypa = Yp - Ypm

    if No > Ne:
        ris = (1.0 / np.sqrt(r))[:, None]
        Z = ris * Ypa                    # Normalised predicted anomalies.
        _, sigmas, Vh = np.linalg.svd(Z, full_matrices=False)
        V = Vh.T
        tv = sigmas.sum()
        if tv == 0:  # Complete ensemble collapse, e.g. EnKF for FSCA after melt.
            return X.copy()
        svr = np.cumsum(sigmas) / tv
        Nr = int(np.argmax(svr >= svthresh)) + 1
        Vr = V[:, :Nr]
        Sr2 = sigmas[:Nr] ** 2
        D = 1.0 / (Sr2 + alpha * Ne)     # (Nr,)

        if dostoch:  # Classic stochastic EnKF
            Ew = np.sqrt(alpha) * rng.standard_normal((No, Ne))
            innoes = ris * (y - Yp) + Ew
            X1 = Vr.T @ (Z.T @ innoes)
            W = Vr @ (D[:, None] * X1)
            Xu = X + Xa @ W
        else:  # DEnKF
            # Update mean
            innoms = ris * (y - Ypm)
            X1 = Vr.T @ (Z.T @ innoms)
            wm = Vr @ (D[:, None] * X1)
            Xum = Xm + Xa @ wm
            # Update anomaly
            Wa = Vr @ ((D * Sr2)[:, None] * Vr.T)
            Xua = Xa - 0.5 * (Xa @ Wa)
            Xu = Xum + Xua
    else:
        C_XY = (Xa @ Ypa.T) / Ne
        C_YY = (Ypa @ Ypa.T) / Ne
        # K = C_XY / (C_YY + alpha R)  (MATLAB right division)
        K = np.linalg.solve((C_YY + alpha * Rm).T, C_XY.T).T

        if dostoch:
            perts = np.sqrt(alpha) * np.sqrt(Rm) @ rng.standard_normal((No, Ne))
            Xu = X + K @ ((y + perts) - Yp)
        else:
            Xum = Xm + K @ (y - Ypm)
            Xua = Xa - 0.5 * (K @ Ypa)
            Xu = Xum + Xua

    return Xu
