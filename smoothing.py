"""Particle filter vs particle batch smoother experiment (port of smoothing.m).

Recreates Figure 4 in the paper (PF_PBS.pdf).
Input data originally from https://zenodo.org/records/22938686,
courtesy of NVE for the Filefjell site.
"""
from datetime import date
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src import ddmvs, pbs, percplot, set_style, style_axes

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(12)
Ne = 10_000
set_style()


def datenum(y, m, d):
    """MATLAB datenum for a calendar date."""
    return date(y, m, d).toordinal() + 366


def to_mpl(dn):
    """MATLAB datenum -> matplotlib date number (days since 1970-01-01)."""
    return np.asarray(dn, dtype=float) - datenum(1970, 1, 1)


datadir = HERE / "input" / "Snow" / "FF"
f = sio.loadmat(datadir / "FF_forcing.mat", simplify_cells=True)["f"]
obs = sio.loadmat(datadir / "FF_obs.mat", simplify_cells=True)["obs"]

wyear = 2022
tstart = datenum(wyear - 1, 9, 1)
tend = datenum(wyear, 9, 1) - 1
thesef = (f["t"] >= tstart) & (f["t"] <= tend)
T = f["T"][thesef]
P = f["P"][thesef]
t = f["t"][thesef].astype(np.int64)

theseo = (obs["t"] >= tstart) & (obs["t"] <= tend)
Dt = obs["D"][theseo]
tt = obs["t"][theseo].astype(np.int64)

# Observation times: first of Jan..Jun in the water year
mos = np.arange(1, 7)
to = np.array([datenum(wyear if m < 9 else wyear - 1, m, 1) for m in mos])

# Monthly x-ticks: Sep (wyear-1) ... Sep (wyear)
tam = list(range(9, 13)) + list(range(1, 10))
tax = np.array([datenum(wyear - 1 if j < 4 else wyear, m, 1) for j, m in enumerate(tam)])

# Synthetic observations from the nearest ground-truth day
sigy = 10.0
Do = np.zeros(to.size)
for j in range(to.size):
    herej = np.argmin(np.abs(tt - to[j]))   # first (earliest) minimiser, as min(ttj)
    Do[j] = max(Dt[herej] + sigy * rng.standard_normal(), 0.0)

hereo = np.isin(t, to)

# Prior parameter ensemble
meda, siga = 4.0, 0.2
a = np.exp(np.log(meda) + siga * rng.standard_normal(Ne))
mub, sigb = 0.0, 1.0
b = mub + sigb * rng.standard_normal(Ne)
medc, sigc = 1.0, 0.5
c = np.exp(np.log(medc) + sigc * rng.standard_normal(Ne))

# %% Filtering experiment
af, bf, cf = a.copy(), b.copy(), c.copy()
Dfpri, Dfpost = [], []
tw = np.append(to, t[-1])  # End of each window
for j in range(tw.size):
    if j == 0:
        D0 = 0.0
        twold = t[0] - 1
    else:
        D0 = Dj[-1, :]
        twold = tw[j - 1]
    herej = (t > twold) & (t <= tw[j])
    Dj = ddmvs(t[herej], P[herej], T[herej], af, bf, cf, D0)
    Dfpri.append(Dj)

    # NB: as in the MATLAB original, the final window (after the last
    # observation) re-uses w from the previous window. Since the particles
    # were already resampled, this amounts to one extra resampling step with
    # weights that no longer belong to the particles.
    if j < to.size:
        w = pbs(Dj[-1, :], Do[j], sigy**2)

    reinds = rng.choice(Ne, size=Ne, replace=True, p=w)

    # For illustration, running without jitter, just resampling.
    # Causes degeneracy.
    dopert = False  # Jitter switch.
    if dopert:
        af = np.exp(np.log(af[reinds]) + 0.1 * rng.standard_normal(Ne))
        bf = bf[reinds] + 0.1 * rng.standard_normal(Ne)
        cf = np.exp(np.log(cf[reinds]) + 0.1 * rng.standard_normal(Ne))
    else:
        af, bf, cf = af[reinds], bf[reinds], cf[reinds]
    Dj = Dj[:, reinds]  # Update Dj for next window (also prior)
    Dfpost.append(Dj)
Dfpri = np.vstack(Dfpri)
Dfpost = np.vstack(Dfpost)

# %% Smoothing experiment (single window covering the whole water year)
as_, bs, cs = a.copy(), b.copy(), c.copy()
Dj = ddmvs(t, P, T, as_, bs, cs, 0.0)
Dspri = Dj
w = pbs(Dj[hereo, :], Do, sigy**2)
reinds = rng.choice(Ne, size=Ne, replace=True, p=w)
as_, bs, cs = as_[reinds], bs[reinds], cs[reinds]
Dspost = Dj[:, reinds]

# %% Figure 4
pcts = [2.5, 97.5]
fs = 30
lwt = 0.5
red, blue = (0.8, 0, 0), (0, 0, 0.8)
tm, ttm, tom = to_mpl(t), to_mpl(tt), to_mpl(to)


def prctile(X):
    # MATLAB's prctile corresponds to numpy's 'hazen' method.
    return np.percentile(X, pcts, axis=1, method="hazen")


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 11), sharex=True,
                               layout="constrained")

for ax, Dpri, Dpost, title, wins, ftr, label in (
        (ax1, Dfpri, Dfpost, "Particle Filter", tom, 0.2, "Filtering"),
        (ax2, Dspri, Dspost, "Particle Batch Smoother", tom[-1:], 0.1, "Smoothing")):
    for x in wins:
        win = ax.plot([x, x], [0, 500], color=(0.5, 0.5, 0.5, 0.5), lw=1, ls="-")[0]
    percplot(ax, tm, prctile(Dpri), red, ftr, "-", red)
    percplot(ax, tm, prctile(Dpost), blue, ftr, "-", blue)
    p1 = ax.plot(tm, Dpri.mean(axis=1), color=(*red, 0.6), lw=2)[0]
    p2 = ax.plot(tm, Dpost.mean(axis=1), color=(*blue, 0.6), lw=2)[0]

    postm = Dpost.mean(axis=1)
    rmse = np.sqrt(np.mean((postm - Dt) ** 2))
    print(f"{label} rmse {rmse:4.2f}")

    p3 = ax.plot(ttm, Dt, color="k", lw=4 * lwt, ls=":")[0]
    p4 = ax.scatter(tom, Do, s=150, marker="o", edgecolors="k",
                    facecolors=(0.95, 0.8, 0.2), linewidths=2, zorder=5,
                    clip_on=False)
    ax.set_ylim(0, 500)
    ax.set_title(title, fontsize=fs)
    ax.set_ylabel(r"SWE, $m$ [kg m$^{-2}$]", fontsize=fs)
    style_axes(ax, fs, ticklength=3)

ax1.legend([p1, p2, p3, p4, win],
           ["Prior", "Posterior", "Ground truth", "Observations", "Windows"],
           loc="upper left", fontsize=fs)

ax2.set_xticks(to_mpl(tax))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax2.set_xlim(to_mpl([datenum(wyear - 1, 10, 1), datenum(wyear, 7, 1)]))
ax1.tick_params(labelbottom=False)

fig.savefig(HERE / "PF_PBS_py.pdf", bbox_inches="tight")
plt.close(fig)
