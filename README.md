<p align="center">
  <img src="assets/logo.png" alt="TACO Logo" width="220"/>
</p>

<h1 align="center">taco</h1>

<p align="center">
  <strong>Toy models for Assimilating Cryospheric Observations</strong>
</p>

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.22938686"><img src="https://img.shields.io/badge/DOI-Dataset-blue.svg" alt="DOI - Dataset"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
</p>

**taco** contains two toy-model examples of cryospheric data assimilation from the review manuscript *Synthesizing data assimilation for snow, glaciers, and permafrost*. Each example is available in both MATLAB and Python.

## Examples

### 1. `SWEtoy`: SWE from snow depth (Figures 2 and 3)

- **Scripts:** `SWEtoy.m` / `SWEtoy.py`
- **Outputs:** `Bayes.pdf` (Figure 2), `joint.jpg` (Figure 3)

A twin (synthetic) experiment in which we infer snow water equivalent (SWE) $m$ from a single noisy snow depth observation $d^o$, with snow density $\rho$ as a nuisance variable. This implies a nonlinear mapping from the hidden state to the noisy observable:

$$d = \frac{m}{\rho} + \epsilon, \qquad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

We approximate the Bayesian posterior $p(m,\rho\mid d^o)$ by assimilating the observed depth (i.e., conditioning on $d=d^o$) using three methods:

- **Grid approximation:** a costly brute-force solution, shown as classical 1D Bayesian updating in Figure 2 and used as the reference posterior for the 2D joint updating in Figure 3.
- **Particle approximation:** importance weighting and resampling of prior samples.
- **Ensemble Kalman approximation:** a deterministic ensemble Kalman analysis, whose linear-Gaussian update is challenged by the nonlinear observation model.

### 2. `smoothing`: particle filtering and batch smoothing (Figure 4)

- **Scripts:** `smoothing.m` / `smoothing.py`
- **Output:** `PF_PBS.pdf` (Figure 4)

In this experiment we use daily SWE measurements from NVE's Filefjell station for the hydrological year 2022, obtained from a curated dataset on [Zenodo](https://doi.org/10.5281/zenodo.22938686), as the reference truth. From this truth, we generate noisy monthly observations for January–June and assimilate them into a degree-day model with uncertain melt factor, temperature bias, and snowfall multiplier using:

- **Particle filter (PF):** weighting and resampling at each observation time.
- **Particle batch smoother (PBS):** a single weighting of full-season trajectories against all observations at once.

For both schemes we use the simplest implementation, with no rejuvenation or proposal adaptation. As a result, even with $10^4$ particles, the filter struggles during the ablation season and the smoother during the accumulation season. The purpose of this exercise is partly to motivate the need to sometimes move beyond simple particle schemes in cryospheric data assimilation. For example, even simple jittering of the parameters greatly improves the filter's performance, as can be verified by setting `dopert` to true in the code.

## How to run

Clone the repository and move into it:

```bash
git clone https://github.com/kristaal/taco.git
cd taco
```

### MATLAB

Requires MATLAB R2020a or newer with the Statistics and Machine Learning Toolbox. From the repository root, run:

```matlab
SWEtoy      % Figures 2 and 3
smoothing   % Figure 4
```

### Python

With [uv](https://docs.astral.sh/uv/) (recommended), which installs the dependencies and a suitable Python version automatically:

```bash
uv sync
uv run SWEtoy.py
uv run smoothing.py
```

Figures are written to the repository root. The Python versions get a `_py` suffix (e.g. `Bayes_py.pdf`) so they don't overwrite the MATLAB output.

## Notes

*Programming languages*: The code was originally written by the authors in MATLAB and translated to Python with the help of a large language model ([Claude](https://claude.ai)). The core functions (degree-day model, particle batch smoother, ensemble Kalman analysis) and the grid calculations were checked against the original MATLAB code (run in GNU Octave) and agree to machine precision. Because MATLAB and NumPy use different random number generators, the ensemble-based results should agree statistically but not draw-for-draw.

*Name*: This repository is named after the unofficial national dish of Norway, the humble taco 🌮, for no particular reason other than that we are fans of both tacos and cryospheric data assimilation.

*Real problems*: To move beyond toy models to real experiments with more complex cryospheric models, we recommend the Multiple Snow Data Assimilation system [MuSA](https://github.com/ealonsogzl/MuSA), which is under continuous development and supports multiple snow models, observables, and assimilation schemes. Related data assimilation tools also exist within the [CryoGrid](https://github.com/CryoGrid) model ecosystem for terrestrial cryospheric (especially permafrost) modeling and are being adopted within the Python Glacier Evolution Model [PyGEM](https://github.com/PyGEM-Community/PyGEM). Published (or in review) example applications from MuSA, CryoGrid, and PyGEM are included as additional figures in the manuscript.

*Logo*: The **taco** logo was generated with Nano Banana via [Google Gemini](https://gemini.google/), adjusted by Claude, and then edited by the authors using [GIMP](https://www.gimp.org).

