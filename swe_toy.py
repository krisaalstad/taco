"""SWE / snow depth / density toy example (port of SWEtoy.m).

Produces Figure 3 (joint.jpg: joint prior and posterior with EnKF and particle
posteriors) and Figure 2 (Bayes.pdf: marginal Bayes for SWE and snow depth).
"""
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from src import enka, pbs, set_style, style_axes

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(123)
set_style()


def H(M, r):
    """Observation operator: snow depth from SWE and density."""
    return M / r


dtrue = 1.2            # True snow depth
rtrue = 300.0          # True density
Mtrue = rtrue * dtrue  # True SWE

sigd = 0.1             # Snow depth measurement error standard deviation (m)
dobs = dtrue - 1 * sigd

# Prior sampling
Ne = 30
Mm, Ms = 500.0, 100.0
M = Mm + Ms * rng.standard_normal(Ne)
rm, rs = 250.0, 50.0
r = rm + rs * rng.standard_normal(Ne)
dpred = H(M, r)

xls = [200, 800]  # 3 sigma range
yls = [100, 400]  # 3 sigma range

# %% Grid approximation
dr = 2.5
ra = dr * np.arange(1, round(500 / dr) + 1)      # dr:dr:500
dM = 2.5
Ma = dM * np.arange(1, round(1e3 / dM) + 1)      # dM:dM:1e3

Mg, rg = np.meshgrid(Ma, ra)                     # rows: density, cols: SWE
Dg = H(Mg, rg)

logpriMg = -0.5 * ((Mg - Mm) ** 2) / Ms**2 - 0.5 * np.log(2 * np.pi * Ms**2)
logprirg = -0.5 * ((rg - rm) ** 2) / rs**2 - 0.5 * np.log(2 * np.pi * rs**2)
logprig = logpriMg + logprirg

loglikeg = -0.5 * ((dobs - Dg) ** 2) / sigd**2 - 0.5 * np.log(2 * np.pi * sigd**2)
logjointg = logprig + loglikeg

evig = trapezoid(trapezoid(np.exp(logjointg), ra, axis=0), Ma)
logpostg = logjointg - np.log(evig)

# Control calculation to verify everything integrates to 1
I = trapezoid(trapezoid(np.exp(logprig), ra, axis=0), Ma)
print(f"Prior integrates to {I:.6f}")

# %% Figure 3
batlowW = sio.loadmat(HERE / "input" / "batlowW.mat")["batlowW"]
cmap = ListedColormap(np.flipud(batlowW))

fs = 25
thresh = 0
extent = [Ma[0] - dM / 2, Ma[-1] + dM / 2, ra[0] - dr / 2, ra[-1] + dr / 2]


def imagesc(ax, Z):
    """imagesc(...,'AlphaData',Z>thresh); axis xy"""
    Zm = np.ma.masked_where(Z <= thresh, Z)
    ax.imshow(Zm, origin="lower", extent=extent, aspect="auto", cmap=cmap,
              interpolation="nearest")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), sharex=True,
                               layout="constrained")

prig = np.exp(logprig)
imagesc(ax1, prig)
ax1.set_xlabel(r"Snow water equivalent (SWE), $m$ [kg m$^{-2}$]", fontsize=fs)
spri1 = ax1.scatter(M, r, s=250, marker="o", edgecolors="k",
                    facecolors=(0.8, 0, 0, 0.6), label="Prior ensemble", zorder=3)
spri2 = ax1.scatter(Mtrue, rtrue, s=350, marker="*", edgecolors="k",
                    facecolors="w", label="Truth", zorder=4)
ax1.set_title(r"(a) Joint prior $p(m,\rho)$", fontsize=fs)
ax1.legend(handles=[spri1, spri2], loc="upper left", fontsize=fs, markerscale=0.5)

postg = np.exp(logpostg)
imagesc(ax2, postg)
ax2.set_xlabel(r"Snow water equivalent (SWE), $m$ [kg m$^{-2}$]", fontsize=fs)

# %% EnKA
thetapri = np.vstack([M, r])
Na = 1
alpha = Na
dostoch = False
for _ in range(Na):
    thetapost = enka(thetapri, dobs, dpred[None, :], alpha, sigd**2, dostoch, rng=rng)
    dpred = H(thetapost[0], thetapost[1])
    thetapri = thetapost

spost3 = ax2.scatter(thetapost[0], thetapost[1], s=250, marker="D",
                     facecolors=(0, 0, 0.8, 0.6), edgecolors="k",
                     label="EnKF posterior", zorder=3)

# %% PBS
thetapri = np.vstack([M, r])
dpred = H(thetapri[0], thetapri[1])
w = pbs(dpred, dobs, sigd**2)
reinds = rng.choice(Ne, size=Ne, replace=True, p=w)

spost1 = ax2.scatter(thetapri[0], thetapri[1], s=250, marker="o", edgecolors="k",
                     facecolors=(0.8, 0, 0, 0.2), label="Prior ensemble", zorder=3)
spost4 = ax2.scatter(thetapri[0, reinds], thetapri[1, reinds], s=250, marker="s",
                     facecolors=(0, 0.8, 0.8, 0.6), edgecolors="k",
                     label="Particle posterior", zorder=4)
spost2 = ax2.scatter(Mtrue, rtrue, s=350, marker="*", edgecolors="k",
                     facecolors="w", label="Truth", zorder=5)
ax2.set_title(r"(b) Joint posterior $p(m,\rho \mid d^o)$", fontsize=fs)
ax2.legend(handles=[spost1, spost2, spost3, spost4], loc="lower left", fontsize=fs,
           markerscale=0.5)

for ax in (ax1, ax2):
    ax.set_xlim(xls)
    ax.set_ylim(yls)
    ax.set_box_aspect(1)
    style_axes(ax, fs)
ax1.set_ylabel(r"Snow density, $\rho$ [m w.e.]", fontsize=fs)

fig.savefig(HERE / "joint_py.jpg", dpi=600, bbox_inches="tight")
plt.close(fig)

# %% Bayes' for SWE = Figure 2
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), layout="constrained")
lw = 3
al = 0.6

priMg = trapezoid(prig, ra, axis=0)
pt4 = ax1.plot([Mtrue, Mtrue], [0, 8e-3], lw=lw, color="k", ls="-.",
               label=r"True SWE $m^\star$")[0]
pt1 = ax1.plot(Ma, priMg, color=(0.8, 0, 0, al), lw=lw, label=r"Prior $p(m)$")[0]
ax1.fill_between(Ma, priMg, color=(0.8, 0, 0), alpha=0.25 * al, lw=0)

# integrated likelihood \int p(y,phi|theta) dphi = int p(y|theta,phi)p(phi) dphi = p(y|theta)
ilike = trapezoid(np.exp(loglikeg + logprirg), ra, axis=0)
ev = trapezoid(ilike * priMg, Ma)

postMg = trapezoid(np.exp(logpostg), ra, axis=0)

nc = trapezoid(ilike, Ma)
pt2 = ax1.plot(Ma, ilike / nc, color=(0.3, 0.3, 0.3, al), lw=lw,
               label=r"Likelihood* $p(d^o\mid m)$")[0]
ax1.fill_between(Ma, ilike / nc, color=(0.3, 0.3, 0.3), alpha=0.25 * al, lw=0)

pt3 = ax1.plot(Ma, postMg, color=(0, 0, 0.8, al), lw=lw,
               label=r"Posterior $p(m\mid d^o)$")[0]
ax1.fill_between(Ma, postMg, color=(0, 0, 0.8), alpha=0.25 * al, lw=0)

ax1.set_ylabel(r"Probability density [m$^2$ kg$^{-1}$]", fontsize=fs)
ax1.set_xlabel(r"Snow water equivalent (SWE) [kg m$^{-2}$]", fontsize=fs)
ax1.legend(handles=[pt1, pt2, pt3, pt4], loc="upper right", fontsize=fs)
ax1.set_xlim(0, 900)
ax1.set_ylim(0, 8e-3)
ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax1.set_title("(a) Hidden state: SWE", fontsize=fs)

va = ra
da = 0.01 * np.arange(1, 401)  # Same size as M (but not same coordinate due to dependence)
dg, vg = np.meshgrid(da, va)

Mgre = dg * vg

logpriMgre = -0.5 * ((Mgre - Mm) ** 2) / Ms**2 - 0.5 * np.log(2 * np.pi * Ms**2)
logprirvg = -0.5 * ((vg - rm) ** 2) / rs**2 - 0.5 * np.log(2 * np.pi * rs**2)
logprigre = logpriMgre + logprirg + np.log(vg)  # joint p(d,v)
loglikegre = -0.5 * ((dobs - dg) ** 2) / sigd**2 - 0.5 * np.log(2 * np.pi * sigd**2)
logjointgre = logprigre + loglikegre
evigre = trapezoid(trapezoid(np.exp(logjointgre), va, axis=0), da)
logpostgre = logjointgre - np.log(evigre)

prid = trapezoid(np.exp(logprigre), ra, axis=0)
# p(d,v) = p(M,r) |det(J)| = p(M,r) |r| if v=r. Units m^2 kg^-1

pt4 = ax2.plot([dtrue, dtrue], [0, 10], lw=lw, color="k", ls="-.",
               label=r"True depth $d^\star$")[0]
pt1 = ax2.plot(da, prid, color=(0.8, 0, 0, al), lw=lw,
               label=r"Prior $p\left(\hat{d}\right)$")[0]
ax2.fill_between(da, prid, color=(0.8, 0, 0), alpha=0.25 * al, lw=0)

ilikere = trapezoid(np.exp(loglikegre + logprirg), va, axis=0)
postdg = trapezoid(np.exp(logpostgre), va, axis=0)

pt5 = ax2.plot([dobs, dobs], [0, 10], lw=lw, color=(0.3, 0.3, 0.3, al), ls=":",
               label=r"Observed depth $d^o$")[0]
ax2.set_ylabel(r"Probability density [m$^{-1}$]", fontsize=fs)
nc = trapezoid(ilikere, da)
pt2 = ax2.plot(da, ilikere / nc, color=(0.3, 0.3, 0.3, al), lw=lw,
               label=r"Likelihood* $p\left(d^o\mid \hat{d}\right)$")[0]
ax2.fill_between(da, ilikere / nc, color=(0.3, 0.3, 0.3), alpha=0.25 * al, lw=0)

pt3 = ax2.plot(da, postdg, color=(0, 0, 0.8, al), lw=lw,
               label=r"Posterior $p\left(\hat{d}\mid d^o\right)$")[0]
ax2.fill_between(da, postdg, color=(0, 0, 0.8), alpha=0.25 * al, lw=0)

ax2.set_xlabel("Snow depth [m]", fontsize=fs)
ax2.legend(handles=[pt1, pt2, pt3, pt4, pt5], loc="upper right", fontsize=fs)
ax2.set_xlim(0, 4)
ax2.set_ylim(0, 5)
ax2.set_title("(b) Observable: Snow depth", fontsize=fs)

for ax in (ax1, ax2):
    ax.set_box_aspect(1)
    style_axes(ax, fs)

fig.savefig(HERE / "Bayes_py.pdf", bbox_inches="tight")
plt.close(fig)
