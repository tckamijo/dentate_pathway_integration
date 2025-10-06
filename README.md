# Frequency-Dependent Nonlinear Integration of Three-Pathway Synaptic Inputs in Dentate Granule Cells

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-pending-orange)](https://github.com/tckamijo/dentate_pathway_integration)

Computational analysis and experimental validation of frequency-dependent integration mechanisms across three major synaptic input pathways to dentate gyrus granule cells.

## Overview

This repository contains all code, data, and analysis scripts for reproducing the computational figures in:

**Kamijo, T.C., Nakajima, N., Aihara, T., Hoshi, H., Takayanagi, M., & Sato, F. (2025)**  
*Frequency-Dependent Nonlinear Integration of Three-Pathway Synaptic Inputs in Dentate Granule Cells: Extension of Lateral Inhibition Framework*  
Cognitive Neurodynamics (submitted)

### Key Findings

- **First comprehensive characterization** of proximal dendrite (PD) pathway synaptic dynamics with 20 Hz critical transition
- **Lateral inhibition mechanism** producing graded frequency-dependent suppression (-7.6% at theta to -11.6% at high gamma)
- **Near-linear integration** under physiological conditions across theta, beta, and gamma frequency bands
- **Complete reproducibility** with publicly available code and experimental parameters

---

## Installation

### Requirements

- Python >= 3.8
- NumPy >= 1.22
- SciPy >= 1.8
- Matplotlib >= 3.5
- Pillow >= 9.0

### Quick Start

```bash
# Clone repository
git clone https://github.com/tckamijo/dentate_pathway_integration.git
cd dentate_pathway_integration

# Install dependencies
pip install -r requirements.txt
```

---

## Reproducible Figures

All computational figures from the manuscript can be reproduced using the scripts in the `main/` directory. Each script generates publication-ready TIFF files (600 dpi, LZW compression).

### Figure Generation Pipeline

| Figure | Description | Script | Output File |
|--------|-------------|--------|-------------|
| **Figure 1** | Frequency-dependent response characteristics across DD/MD/PD pathways | `figure1-2_create_figures_threshold_only.py` | `Figure1_Threshold_Summary_YYYYMMDD_HHMMSS.tiff` |
| **Figure 2** | Pathway integration heatmap showing frequency-dependent dominance | `figure1-2_create_figures_threshold_only.py` | `Figure2_Threshold_Detail_YYYYMMDD_HHMMSS.tiff` |
| **Figure 3** | Validation through reproduction of Hayakawa et al. (2014) | `figure3_reproduction_hayakawa2014.py` | `Figure3_Hayakawa2014_Reproduction_YYYYMMDD_HHMMSS.tiff` |
| **Figure 5** | PD pathway parameter validation with 20 Hz threshold | `figure5_pd_validation_20hz_threshold.py` | `Figure5_PD_Validation_20Hz_YYYYMMDD_HHMMSS.tiff` |
| **Figure 6** | Three-pathway integration reveals near-linear summation | `figure6_three_pathway_integration_20hz.py` | `Figure6_Three_Pathway_Integration_20Hz_YYYYMMDD_HHMMSS.tiff` |
| **Figure 7** | Mechanism analysis of resource competition and temporal dynamics | `figure7_mechanism_analysis_20hz.py` | `Figure7_Mechanism_Analysis_20Hz_YYYYMMDD_HHMMSS.tiff` |
| **Figure 8** | Natural input pattern analysis validates linear integration | `figure8_natural_patterns_20hz.py` | `Figure8_Natural_Patterns_20Hz_YYYYMMDD_HHMMSS.tiff` |
| **Figure 9** | Lateral inhibition effects on three-pathway integration | `figure9_lateral_inhibition_validated.py` | `Figure9_Lateral_Inhibition_YYYYMMDD_HHMMSS.tiff` |
| **Figure 10** | Physiological frequency summary across theta/beta/gamma bands | `figure10_physiological_frequencies.py` | `Figure10_Physiological_Frequencies_YYYYMMDD_HHMMSS.tiff` |

**Note:** Figure 4 (experimental validation data) is not included in this repository as it contains primary experimental recordings subject to institutional data sharing policies.

### Generate All Figures

```bash
# Generate Figures 1-2
python main/figure1-2_create_figures_threshold_only.py

# Generate Figure 3
python main/figure3_reproduction_hayakawa2014.py

# Generate Figure 5
python main/figure5_pd_validation_20hz_threshold.py

# Generate Figure 6
python main/figure6_three_pathway_integration_20hz.py

# Generate Figure 7
python main/figure7_mechanism_analysis_20hz.py

# Generate Figure 8
python main/figure8_natural_patterns_20hz.py

# Generate Figure 9
python main/figure9_lateral_inhibition_validated.py

# Generate Figure 10
python main/figure10_physiological_frequencies.py
```

All figures will be saved in the `Figures/` directory with timestamps.

---

## Repository Structure

```
dentate_pathway_integration/
├── main/                          # Core analysis scripts for manuscript figures
│   ├── figure1-2_create_figures_threshold_only.py
│   ├── figure3_reproduction_hayakawa2014.py
│   ├── figure5_pd_validation_20hz_threshold.py
│   ├── figure6_three_pathway_integration_20hz.py
│   ├── figure7_mechanism_analysis_20hz.py
│   ├── figure8_natural_patterns_20hz.py
│   ├── figure9_lateral_inhibition_validated.py
│   └── figure10_physiological_frequencies.py
├── Figures/                       # Generated output (TIFF, 600 dpi)
├── data/                          # Experimental parameters and model configurations
├── docs/                          # Additional documentation
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Computational Methods

### Tsodyks-Markram Synaptic Dynamics

All simulations use the Tsodyks-Markram dynamic synapse model with pathway-specific parameters:

- **DD (Distal Dendrite / LPP)**: τ_rec = 248 ms, τ_facil = 133 ms, U = 0.20 (facilitation)
- **MD (Middle Dendrite / MPP)**: τ_rec = 3977 ms, τ_facil = 27 ms, U = 0.30 (depression)
- **PD (Proximal Dendrite)**: 
  - Low-frequency (<20 Hz): τ_rec = 460 ms, τ_facil = 20 ms, U = 0.32
  - High-frequency (≥20 Hz): τ_rec = 184 ms, τ_facil = 52.5 ms, U = 0.2135

### Lateral Inhibition Implementation

Lateral inhibition from DD to MD pathways was modeled based on Nakajima et al. (2024) with activity-dependent suppression and exponential decay dynamics (τ_decay = 5-15 ms).

### Frequency Analysis

Simulations span physiologically relevant frequency ranges:
- **Theta**: 8 Hz
- **Beta**: 15 Hz
- **Low Gamma**: 30 Hz
- **High Gamma**: 40 Hz

---

## Citation

If you use this code or reproduce any figures, please cite:

```bibtex
@article{kamijo2025frequency,
  title={Frequency-Dependent Nonlinear Integration of Three-Pathway Synaptic Inputs in Dentate Granule Cells: Extension of Lateral Inhibition Framework},
  author={Kamijo, Tadanobu C. and Nakajima, Naoki and Aihara, Takeshi and Hoshi, Hideo and Takayanagi, Masaaki and Sato, Fumi},
  journal={Cognitive Neurodynamics},
  year={2025},
  note={Submitted}
}
```

---

## Related Publications

This work extends previous findings published in *Cognitive Neurodynamics*:

- Hayakawa, H., et al. (2015). Spatial information enhanced by non-spatial information in hippocampal granule cells. *Cogn Neurodyn*, 9(1), 1-12.
- Kamijo, T.C., et al. (2014). Input integration around the dendritic branches in hippocampal dentate granule cells. *Cogn Neurodyn*, 8(4), 267-276.
- Nakajima, N., et al. (2024). Modification of temporal pattern sensitivity for inputs from medial entorhinal cortex by lateral inputs in hippocampal granule cells. *Cogn Neurodyn*, 18(3), 1047-1059.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

**Tadanobu C. Kamijo, Ph.D.**  
Department of Systems Physiology  
Graduate School of Medicine, University of the Ryukyus  
1076 Kiyuna, Ginowan-city, Okinawa 901-2720, Japan  
Email: tadanobu@cs.u-ryukyu.ac.jp

For questions about the code or analysis: Open an issue on GitHub

---

## Acknowledgments

This work was supported by KAKENHI Grant-in-Aid for Scientific Research (C) 23K10504 and internal support from Toho University. We thank the University Research Administrator office at Toho University for assistance with manuscript preparation.
