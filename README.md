# Frequency-Dependent Nonlinear Integration of Three-Pathway Synaptic Inputs in Dentate Granule Cells

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17276481.svg)](https://doi.org/10.5281/zenodo.17276481)
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
- **Rigorous statistical validation** with 30-trial averaging and biological noise modeling (CV=0.20)

---

## Installation

### Requirements

- Python >= 3.8
- NumPy >= 1.22
- SciPy >= 1.8
- Matplotlib >= 3.5
- Pandas >= 1.3
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

All computational figures from the manuscript can be reproduced using the scripts in the `main/` directory. Each script generates publication-ready TIFF files (600 dpi, LZW compression) with **30-trial averaging and error bars (Mean ± SEM)**.

### Figure Generation Pipeline

| Figure | Description | Script | Output Files |
|--------|-------------|--------|-------------|
| **Figure 1** | Frequency-dependent response characteristics across DD/MD/PD pathways | `figure1-2_with_variability_30trials.py` | `Figure1_variability_*.png/tiff` |
| **Figure 2** | Pathway integration heatmap showing frequency-dependent dominance | `figure1-2_with_variability_30trials.py` | `Figure2_variability_*.png/tiff` |
| **Figure 3** | Validation through reproduction of Hayakawa et al. (2014) | `figure3_with_variability_30trials.py` | `Figure3_Hayakawa2014_variability_*.png/tiff` |
| **Figure 5** | PD pathway parameter validation with 20 Hz threshold | `figure5_with_variability_30trials.py` | `Figure5_PD_validation_variability_*.png/tiff` |
| **Figure 6** | Three-pathway integration reveals near-linear summation | `figure6_with_variability_30trials.py` | `Figure6_Integration_variability_*.png/tiff` |
| **Figure 7** | Mechanism analysis of resource competition and temporal dynamics | `figure7_with_variability_30trials.py` | `Figure7_Mechanism_variability_*.png/tiff` |
| **Figure 8** | Natural input pattern analysis validates linear integration | `figure8_with_variability_30trials.py` | `Figure8_patterns_variability_*.png/tiff` |
| **Figure 9** | Lateral inhibition effects on three-pathway integration | `figure9_with_variability_30trials.py` | `Figure9_inhibition_variability_*.png/tiff` |
| **Figure 10** | Physiological frequency summary across theta/beta/gamma bands | `figure10_physiological_30trials.py` | `Figure10_Physiological_variability_*.png/tiff` |

**Note:** Figure 4 (experimental validation data) is not included in this repository as it contains primary experimental recordings subject to institutional data sharing policies.

### Generate All Figures

```bash
# Generate Figures 1-2 (30 trials, CV=0.20)
python main/figure1-2_with_variability_30trials.py

# Generate Figure 3
python main/figure3_with_variability_30trials.py

# Generate Figure 5
python main/figure5_with_variability_30trials.py

# Generate Figure 6
python main/figure6_with_variability_30trials.py

# Generate Figure 7
python main/figure7_with_variability_30trials.py

# Generate Figure 8
python main/figure8_with_variability_30trials.py

# Generate Figure 9
python main/figure9_with_variability_30trials.py

# Generate Figure 10
python main/figure10_physiological_30trials.py
```

All figures will be saved in the `Figures/` directory with timestamps. Each execution also generates:
- **CSV data files**: Raw numerical results in `data/`
- **Log files**: Execution details and parameters in `data/`

---

## Repository Structure

```
dentate_pathway_integration/
├── main/                          # Core analysis scripts for manuscript figures
│   ├── figure1-2_with_variability_30trials.py
│   ├── figure3_with_variability_30trials.py
│   ├── figure5_with_variability_30trials.py
│   ├── figure6_with_variability_30trials.py
│   ├── figure7_with_variability_30trials.py
│   ├── figure8_with_variability_30trials.py
│   ├── figure9_with_variability_30trials.py
│   └── figure10_physiological_30trials.py
├── Figures/                       # Generated output (PNG + TIFF, 600 dpi)
│   ├── Figure1_variability_*.png/tiff
│   ├── Figure2_variability_*.png/tiff
│   ├── Figure3_Hayakawa2014_variability_*.png/tiff
│   ├── Figure5_PD_validation_variability_*.png/tiff
│   ├── Figure6_Integration_variability_*.png/tiff
│   ├── Figure7_Mechanism_variability_*.png/tiff
│   ├── Figure8_patterns_variability_*.png/tiff
│   ├── Figure9_inhibition_variability_*.png/tiff
│   └── Figure10_Physiological_variability_*.png/tiff
├── data/                          # Simulation results and logs
│   ├── Figure*_data_variability_*.csv
│   └── figure*_variability_*.log
├── docs/                          # Additional documentation
│   └── methodology.md             # Detailed methods and implementation
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT License
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

### Biological Noise and Statistical Analysis

**NEW IN THIS VERSION:**

All simulations now include:
- **30 trials per condition**: Ensures statistical robustness
- **Biological noise**: Multiplicative Gaussian noise with coefficient of variation (CV) = 0.20
- **Statistical measures**: Mean ± Standard Error of Mean (SEM)
- **Error bars**: Displayed on all figures for transparency
- **Reproducibility**: Random seed control for exact replication

#### Noise Implementation

```python
response_noisy = response_base * np.random.normal(1.0, cv)
where cv = 0.20 (20% trial-to-trial variability)
```

This noise level is consistent with experimental observations in hippocampal synapses.

### Lateral Inhibition Implementation

Lateral inhibition from DD to MD pathways was modeled based on Nakajima et al. (2024) with:
- Activity-dependent suppression
- Exponential decay dynamics (τ_decay = 10 ms)
- Frequency-dependent strength:
  - <20 Hz: 0.3
  - ≥20 Hz: 0.3 + 0.2 × (frequency/40)
- Shunting factor: 2.0 (divisive inhibition)

### Frequency Analysis

Simulations span physiologically relevant frequency ranges:
- **Theta**: 8 Hz
- **Beta**: 15 Hz  
- **Low Gamma**: 30 Hz
- **High Gamma**: 40 Hz
- **Extended range**: 5-40 Hz (5 Hz steps)

### Phase Relationships

Inter-pathway timing based on anatomical connectivity:
- **DD**: 0 ms (reference)
- **MD**: 3 ms (delayed, longer pathway)
- **PD**: 1.5 ms (intermediate)

---

## Output Specifications

### Figure Files

Each script generates three types of output:

1. **PNG images**: High-resolution (600 DPI) for quick preview
2. **TIFF images**: Publication-ready (600 DPI, LZW compression)
3. **CSV data**: Raw numerical results with trial-by-trial data

### Data Format

CSV files include:
- Condition labels (frequency, pathway, model type)
- Mean responses
- Standard Error of Mean (SEM)
- Individual trial data (for reanalysis)

### Log Files

Detailed execution logs contain:
- All simulation parameters
- Timestamp information
- Statistical summaries
- Runtime performance metrics

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

@software{kamijo2025code,
  title={Dentate Pathway Integration: 30-Trial Simulation Code},
  author={Kamijo, Tadanobu C.},
  year={2025},
  publisher={Zenodo},
  doi={10.5281/zenodo.17276481},
  url={https://github.com/tckamijo/dentate_pathway_integration}
}
```

---

## Related Publications

This work extends previous findings published in *Cognitive Neurodynamics*:

- Hayakawa, H., et al. (2015). Spatial information enhanced by non-spatial information in hippocampal granule cells. *Cogn Neurodyn*, 9(1), 1-12.
- Kamijo, T.C., et al. (2014). Input integration around the dendritic branches in hippocampal dentate granule cells. *Cogn Neurodyn*, 8(4), 267-276.
- Nakajima, N., et al. (2024). Modification of temporal pattern sensitivity for inputs from medial entorhinal cortex by lateral inputs in hippocampal granule cells. *Cogn Neurodyn*, 18(3), 1047-1059.

### Key Model References

- Tsodyks, M., & Markram, H. (1997). The neural code between neocortical pyramidal neurons depends on neurotransmitter release probability. *PNAS*, 94(2), 719-723.

---

## Version History

### v2.0.0 (Current) - 30-Trial Statistical Validation
- ✅ 30 trials per condition for all figures
- ✅ Biological noise model (CV=0.20)
- ✅ Mean ± SEM calculations
- ✅ Error bars on all plots
- ✅ Enhanced statistical rigor
- ✅ Complete reproducibility with random seed control

### v1.0.0 - Initial Release
- Single-trial simulations
- Basic figure generation
- Core model implementation

---

## Documentation

For detailed methodology, model equations, and implementation details, see:
- [docs/methodology.md](docs/methodology.md) - Complete mathematical framework
- [GitHub Issues](https://github.com/tckamijo/dentate_pathway_integration/issues) - Questions and bug reports

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

For questions about the code or analysis: [Open an issue on GitHub](https://github.com/tckamijo/dentate_pathway_integration/issues)

---

## Acknowledgments

This work was supported by:
- KAKENHI Grant-in-Aid for Scientific Research (C) 23K10504
- Internal support from Toho University

We thank the University Research Administrator office at Toho University for assistance with manuscript preparation.

---

## Contributing

We welcome contributions! Please feel free to:
- Report bugs via GitHub Issues
- Submit pull requests for bug fixes
- Suggest improvements to documentation
- Share your use cases and extensions

For major changes, please open an issue first to discuss proposed modifications.

---

**Last Updated**: October 2025  
**Repository**: https://github.com/tckamijo/dentate_pathway_integration  
**Zenodo DOI**: 10.5281/zenodo.17276481
