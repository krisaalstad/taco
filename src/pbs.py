import numpy as np
from scipy.special import logsumexp


def pbs(HX, Y, R):
    """Efficient implementation of the Particle Batch Smoother (port of PBS.m).

    As presented in Margulis et al. (2015; JHM). Observation errors are
    assumed uncorrelated (diagonal R) and Gaussian.

    Parameters
    ----------
    HX : (No, Ne) array of predicted observations. A 1-D array of length Ne
         is treated as a single observation (No = 1).
    Y  : (No,) array (or scalar) of unperturbed observations.
    R  : scalar, or (No,) array of observation error variances.

    Returns
    -------
    w : (Ne,) array of normalised posterior weights (prior weights are 1/Ne).

    Code by Kristoffer Aalstad (Feb. 2019), ported to Python.
    """
    HX = np.atleast_2d(np.asarray(HX, dtype=float))
    Y = np.atleast_1d(np.asarray(Y, dtype=float)).ravel()
    No = Y.size
    if HX.shape[0] != No:
        raise ValueError(f"HX has {HX.shape[0]} rows but there are {No} observations")

    # Diagonal of the inverse obs. error covariance.
    R = np.asarray(R, dtype=float)
    if R.size == 1:
        Rinv = np.full(No, 1.0 / R.item())
    elif R.size == No:
        Rinv = 1.0 / R.ravel()
    else:
        raise ValueError("Expected R to be a scalar or to have No entries")

    # Log-likelihood; the Gaussian normalising constant drops out.
    Inn = Y[:, None] - HX              # Innovation, (No, Ne)
    EObj = Rinv @ (Inn ** 2)           # (Ne,) ensemble objective function
    LLH = -0.5 * EObj

    # Normalise with log-sum-exp to avoid underflow for small R / many obs.
    logw = LLH - logsumexp(LLH)
    return np.exp(logw)
