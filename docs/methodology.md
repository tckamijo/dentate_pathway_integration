# Methodology

This document describes the modeling framework, parameters, and analysis methods used in the simulations.

## 1. Synapse Model
All simulations are based on the **Tsodyks–Markram short-term plasticity model**:
- `tau_rec`: recovery time constant
- `tau_facil`: facilitation time constant
- `U`: utilization of synaptic efficacy

## 2. Pathway Definitions
- **DD (Distal Dendritic, LPP):** facilitation-dominant synapses
- **MD (Medial Dendritic, MPP):** depression-dominant synapses
- **PD (Proximal Dendritic):** intermediate/moderate depression

## 3. Simulation Protocol
- Input: trains of 5 pulses at frequencies from 0.1 Hz to 40 Hz
- Output: normalized fEPSP amplitudes
- Baseline duration: 500 ms, stimulation duration: 2000 ms

## 4. Parameter Fitting
- Experimental PD pathway data were fitted using nonlinear least squares.
- Key parameters: τ_rec, τ_facil, U.
- High-frequency facilitation (30–40 Hz) was emphasized.

## 5. Competitive Learning Model
- Implemented Hebbian plasticity with Tsodyks–Markram dynamics.
- Long-term weight adaptation demonstrates PD-dominance in competitive environments.

## 6. Figure Generation
- Publication-quality plots were generated with Matplotlib.
- Figures are automatically saved in `/figures/` for reproducibility.

---

This methodology ensures full reproducibility of all results presented in the main text and supplementary materials.
