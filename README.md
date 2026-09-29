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

<!--
---
## Overview

-->



**taco** contains two new 'toy model' cryospheric data assimilation examples from the manuscript *Synthesizing data assimilation for snow, glaciers, and permafrost*. 

The **first** example is contained in the `SWEtoy` scripts (either `.m` or `.py`) that reproduce Figure 2 and Figure 3 in the aforementioned manuscript. This is a twin (snythetic) experiment where we infer snow water equivalent (SWE) $m$ using a single noisy snow depth observation $d^o$ with snow density $\rho$ as a nuisance variable. Note that this implies a nonlinear mapping from the hidden state to the noisy observables of the form $d=m/\rho+\epsilon$ where $\epsilon$ is the(uncertain) observation error. On the one hand, we obtain a brute-force solution to the problem via a costly **grid approximation** as visualized in the classical 1D Bayesian updating in Figure 2 (called `Bayes.pdf` herein) and also as a reference for the 2D joint Bayesian updating in Figure 3 (called `joint.jpg` herein) where we compare the grid approximation to a **particle approximation** and an **ensemble Kalman approximation**. 

The **second** example is contained in the `smoothing` that reproduce Figure 4 in the aforementioned manuscript. In this experiment, we use daily SWE measurements from Filefjell in 2022 as a reference truth, generate monthly noisy observations from these for the period January-June, and then assimilate the observations into a degree day model with a **particle filter** and a **particle batch smoother**. For both schemes, we use their most simple implementation with no rejuvenation or proposal adaptation. As such, even with a large ensemble of $10^4$ particles, these particular implementations of particle fitlering and smoothing struggle in the ablation and accumulation season, respectively. The purpose of this exercise is partly to help motivate the need to sometimes move beyond simple particle schemes in cryospheric data assimilation. For example, even adding jitter greatly improves the perofrmance of the particle filter as can be verified by turning on this switch in the code.

*Programming languages*: The code was originally written by the authors (i.e., humans) in MATLAB but has with the aid of a large language model (LLM), particularly [Claude](https://claude.ai/) from Anthropic, been translated to Python. The latter Python implementation has been verified to reproduce the same results as the original MATLAB implementation. Any errors in the code are likely the fault of the authors who coded the original scripts and not the machine translation. 


*Name*: This repository is named after the infamous national dish of Norway, the humble taco, for no particular reason other than that we are fans of both tacos and cryospheric data assimilation.

*Real problems*: To move beyond toy models to real experiments with more complex cryospheric models, we recommend you check out the Multiple Snow Data Assimilation system [MuSA](https://github.com/ealonsogzl/MuSA) which is undergoing continuous development while supporting multiple: snow models, observables, and assimilation schemes. Related tools data assimilation also exist within the [CryoGrid](https://github.com/CryoGrid) model ecosystem for terrestrial cryospheric (especially permafrost) modeling and are being adopted within the Python Glacier Evolution model [PyGEM](https://github.com/PyGEM-Community/PyGEM). Published (or in review) example applications from MuSA, CryoGrid, and PyGEM are all included as additional Figures in the manuscript.