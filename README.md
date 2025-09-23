# Dentate Gyrus Three-Pathway Integration Analysis

Code for computational analysis of three-pathway synaptic dynamics in dentate gyrus granule cells.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Generate Main Figures
```bash
# Frequency response characteristics (Figures 1-2)
python main/create_figures_threshold_only.py

# Model validation (Figure 3)
python main/fig3_reproduction_hayakawa2014.py

# Parameter fitting analysis (Figures 4-5)
python main/pd_highfreq_parameter_fitting.py
```

## Repository Structure

```
├── main/                 # Core analysis scripts (Paper Figures 1-5)
│   ├── create_figures_threshold_only.py
│   ├── fig3_reproduction_hayakawa2014.py
│   └── pd_highfreq_parameter_fitting.py
├── supplementary/        # Validation and exploratory analyses
│   ├── supp_enhanced_cooperative_learning.py
│   ├── supp_robustness_validation_study.py
│   ├── supp_pd_highfreq_parameter_fitting.py
│   ├── supp_three_pathway_physiological_simulation.py
│   └── supp_fig5_reproduction_hayakawa2014.py
├── data/                 # Experimental parameters
├── Figures/              # Generated figures
└── docs/                 # Documentation
```

### Main Analysis (Published Results)
The `main/` directory contains code that generates Figures 1-5 in the published paper.

### Supplementary Analysis (Validation & Development)
The `supplementary/` directory contains additional analyses performed during method development and validation:

- **Enhanced Cooperative Learning**: Advanced learning algorithm achieving SuM distance <0.3
- **Robustness Validation**: Reproducibility testing across multiple random seeds
- **Parameter Fitting**: Detailed parameter optimization and validation
- **Physiological Simulation**: Extended validation under realistic conditions
- **Extended Reproduction**: Comprehensive replication of Hayakawa et al. (2014)

## Key Features

- Tsodyks-Markram synaptic dynamics modeling
- Three-pathway integration analysis (DD/MD/PD)
- Frequency response characterization (0.1-40 Hz)
- Parameter validation with experimental data

## Requirements

- Python >= 3.7
- NumPy >= 1.19
- Matplotlib >= 3.3
- SciPy >= 1.5

See `requirements.txt` for complete dependencies.

## Citation

If you use this code, please cite:
[Paper citation when published]

## Contact

For questions about implementation: [email]
