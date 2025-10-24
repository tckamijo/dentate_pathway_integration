# Methodology

This document describes the modeling framework, parameters, statistical methods, and analysis techniques used in the simulations.

---

## 1. Synapse Model: Tsodyks-Markram Dynamics

All simulations are based on the **Tsodyks-Markram short-term plasticity model** (Tsodyks & Markram, 1997), which describes dynamic changes in synaptic efficacy through two state variables:

### 1.1 Model Equations

**Between spikes (recovery phase):**
```
dx/dt = (1 - x) / τ_rec
du/dt = (U - u) / τ_facil
```

**At each spike (release and facilitation):**
```
Response = u × x
x_new = x × (1 - u)
u_new = u + U × (1 - u)
```

### 1.2 Model Parameters

- **τ_rec** (recovery time constant): Controls the rate of resource replenishment (depression)
- **τ_facil** (facilitation time constant): Controls the decay of facilitation
- **U** (utilization of synaptic efficacy): Baseline release probability
- **x** (available resources): Fraction of neurotransmitter available for release (0 ≤ x ≤ 1)
- **u** (utilization parameter): Current release probability (0 ≤ u ≤ 1)

### 1.3 Pathway-Specific Implementation

#### DD Pathway (Distal Dendrite / LPP)
```python
tau_rec = 248 ms
tau_facil = 133 ms
U = 0.20
```
**Characteristics**: Facilitation-dominant with moderate recovery

#### MD Pathway (Middle Dendrite / MPP)
```python
tau_rec = 3977 ms
tau_facil = 27 ms
U = 0.30
```
**Characteristics**: Depression-dominant with very slow recovery

#### PD Pathway (Proximal Dendrite)

**Frequency-dependent dual-mode operation:**

**Low-frequency mode (<20 Hz):**
```python
tau_rec = 460 ms
tau_facil = 20 ms
U = 0.32
```

**High-frequency mode (≥20 Hz):**
```python
tau_rec = 184 ms
tau_facil = 52.5 ms
U = 0.2135
```

**Transition**: Sharp threshold at 20 Hz, validated against experimental data (Figure 5)

---

## 2. Biological Noise and Statistical Framework

### 2.1 Noise Implementation

To simulate biological variability, **multiplicative Gaussian noise** is applied to each synaptic response:

```python
response_noisy = response_base × noise_factor
where noise_factor ~ N(1.0, CV)
```

**Parameters:**
- **CV (Coefficient of Variation)**: 0.20 (20% trial-to-trial variability)
- **Distribution**: Gaussian, truncated at 0.01 to prevent negative values
- **Application**: Independent for each stimulus in each trial

**Rationale:**
- CV = 0.20 is consistent with experimental observations in hippocampal synapses
- Multiplicative noise preserves relative response dynamics across frequencies

### 2.2 Multi-Trial Averaging

**All simulations use 30 trials per condition:**

```python
n_trials = 30
results = [simulate_single_trial(cv=0.20) for _ in range(n_trials)]
```

**Statistical measures:**
- **Mean response**: `μ = np.mean(results)`
- **Standard Error of Mean**: `SEM = scipy.stats.sem(results)`
- **95% Confidence Interval**: `CI = μ ± 1.96 × SEM`

### 2.3 Error Propagation

For derived quantities (e.g., interaction coefficient):

**Independent measurements:**
```
error_combined = √(error_a² + error_b²)
```

**Ratios:**
```
relative_error = √((error_a/a)² + (error_b/b)²)
```

---

## 3. Lateral Inhibition Model

### 3.1 Implementation (Nakajima et al., 2024)

Lateral inhibition from DD pathway onto MD pathway using **divisive shunting inhibition**:

```python
inhibited_response = response / (1 + inhibition × shunting_factor)
```

### 3.2 Inhibition Calculation

```python
inhibition = Σ(dd_strength × strength × exp(-time_diff / τ_decay))
```

**Parameters:**
- **τ_decay**: 10 ms (time constant of inhibitory conductance)
- **Delay**: 0.8 ms (synaptic transmission delay)
- **Shunting factor**: 2.0 (divisive inhibition strength)

### 3.3 Frequency Dependence

**Inhibitory strength varies with stimulation frequency:**

```python
if frequency < 20:
    strength = 0.3
else:
    strength = 0.3 + 0.2 × (frequency / 40)
```

**Temporal window**: Inhibition computed for DD activity within 5τ_decay (50 ms) before MD stimulus

---

## 4. Phase Relationships

### 4.1 Inter-Pathway Timing

Based on anatomical pathway lengths and conduction velocities:

```python
phase_offsets = {
    'DD': 0 ms,      # Reference (longest pathway)
    'MD': 3 ms,      # Delayed (intermediate pathway)
    'PD': 1.5 ms     # Intermediate (shortest pathway)
}
```

### 4.2 Stimulation Protocol

For frequency f (Hz):
```python
ISI = 1000 / f  # Inter-stimulus interval (ms)
time_pathway = pulse_number × ISI + phase_offset
```

---

## 5. Pathway Integration Analysis

### 5.1 Individual Pathway Responses

Each pathway stimulated independently:
```python
for pathway in ['DD', 'MD', 'PD']:
    synapse.reset()
    response = sum([synapse.stimulate(time_i) for i in range(n_pulses)])
```

### 5.2 Linear Summation

Expected response assuming independent summation:
```python
linear_sum = response_DD + response_MD + response_PD
```

### 5.3 Integrated Response

Actual response with temporal interactions:
```python
# All pathways with phase offsets
integrated_response = simulate_with_all_pathways(
    DD_times, MD_times, PD_times,
    lateral_inhibition=True
)
```

### 5.4 Interaction Coefficient

Quantifies nonlinearity:
```python
interaction_coeff = ((integrated - linear_sum) / linear_sum) × 100
```

**Interpretation:**
- **> 0**: Supralinear integration (facilitation)
- **= 0**: Linear summation
- **< 0**: Sublinear integration (suppression)

---

## 6. Simulation Protocol

### 6.1 Frequency Range

**Standard analysis:**
```python
frequencies = [5, 10, 15, 20, 25, 30, 35, 40]  # Hz
```

**Physiological bands:**
```python
frequency_bands = {
    'Theta': 8,
    'Beta': 15,
    'Low_Gamma': 30,
    'High_Gamma': 40
}
```

### 6.2 Pulse Train Configuration

- **Number of pulses**: 10 per train
- **Analysis window**: Full train duration
- **Baseline**: 100 ms pre-stimulus
- **Normalization**: First pulse response = 100%

### 6.3 Natural Stimulation Patterns

**Theta-nested gamma (Figure 8):**

```python
# Theta component (8 Hz)
theta_phase = 2π × 8 × time

# Gamma envelope (modulated by theta)
gamma_amplitude = 1 + 0.5 × cos(theta_phase)

# Gamma frequency varies with theta phase
gamma_freq = 30 + 10 × cos(theta_phase)  # 20-40 Hz range
```

---

## 7. Parameter Fitting and Validation

### 7.1 PD Pathway Parameter Optimization

**Experimental data source**: Hayakawa et al. (2014)

**Fitting method**: Nonlinear least squares minimization
```python
from scipy.optimize import minimize

def error_function(params):
    model_response = simulate(params)
    experimental_response = load_experimental_data()
    return np.sum((model_response - experimental_response)²)

optimal_params = minimize(error_function, initial_guess)
```

**Target features**:
- Frequency response curves (5-40 Hz)
- Layer-specific responses (Layer II/III vs. Layer V)
- 20 Hz transition behavior

### 7.2 Validation Metrics

- **Root Mean Square Error (RMSE)**: `√(Σ(model - experiment)² / n)`
- **Pearson correlation**: `r = cov(model, experiment) / (σ_model × σ_experiment)`
- **Normalized difference**: `|model - experiment| / experiment`

---

## 8. Figure Generation

### 8.1 Output Specifications

**Image formats:**
- **PNG**: 600 DPI (for quick preview)
- **TIFF**: 600 DPI, LZW compression (publication-ready)

**Color scheme:**
- Grayscale-compatible for print
- Pathway distinction via line styles and markers:
  - DD: dark gray + circles (○)
  - MD: medium gray + squares (■)
  - PD: light gray + triangles (△)

### 8.2 Error Bar Display

All multi-trial simulations display:
```python
plt.errorbar(x, mean, yerr=sem, 
             capsize=3, linewidth=1.5,
             marker='o', markersize=8)
```

### 8.3 Statistical Annotations

Significant differences indicated by:
- **p < 0.05**: *
- **p < 0.01**: **
- **p < 0.001**: ***

(Computed using paired t-tests or ANOVA where appropriate)

---

## 9. Computational Implementation

### 9.1 Numerical Methods

**Time discretization:**
- Δt = 0.1 ms for continuous dynamics
- Exact timing for discrete spike events

**Integration method:**
- Exponential relaxation (analytical solution between spikes)
- Event-driven updates at spike times

### 9.2 Performance Optimization

**Single trial:** ~1-10 ms
**30 trials:** ~30-300 ms per condition
**Full analysis:** Minutes to hours (depending on parameter space)

### 9.3 Reproducibility

**Random seed control:**
```python
np.random.seed(42)  # Fixed for exact replication
```

**Version control:**
- All code tracked via Git
- Parameters logged in CSV files
- Timestamps in all output files

---

## 10. Data Output

### 10.1 CSV Data Files

**Structure:**
```csv
frequency,pathway,model,mean_response,sem_response,trial_1,trial_2,...,trial_30
8,DD,Complete,-8.76,0.89,value1,value2,...,value30
```

**Contents:**
- Condition labels (frequency, pathway, model configuration)
- Summary statistics (mean, SEM)
- Individual trial data (for reanalysis)

### 10.2 Log Files

**Contents:**
```
[2025-10-23 10:38:26] INFO: Starting Figure 10 analysis
[2025-10-23 10:38:26] INFO: Parameters: n_trials=30, CV=0.20
[2025-10-23 10:38:26] INFO: Processing Theta (8Hz)
[2025-10-23 10:38:26] INFO:   Complete model: -8.76 ± 0.89%
[2025-10-23 10:38:27] INFO: Analysis complete
```

**Information logged:**
- All simulation parameters
- Execution timestamps
- Statistical summaries
- Runtime performance metrics
- Warnings or errors

---

## 11. Quality Control

### 11.1 Validation Checks

**Pre-simulation:**
- Parameter range verification
- Physical plausibility checks (0 ≤ x, u ≤ 1)

**Post-simulation:**
- Convergence verification (trial-to-trial stability)
- Outlier detection (> 3σ from mean)
- Conservation checks (resource balance)

### 11.2 Code Testing

**Unit tests:**
- Synapse model accuracy
- Noise distribution properties
- Statistical calculation correctness

**Integration tests:**
- End-to-end figure generation
- Data format validation
- Reproducibility verification

---

## 12. Software Dependencies

### 12.1 Core Libraries

```python
numpy >= 1.22      # Numerical computations
scipy >= 1.8       # Statistical functions
pandas >= 1.3      # Data management
matplotlib >= 3.5  # Visualization
Pillow >= 9.0      # Image processing
```

### 12.2 Development Tools

```python
pytest >= 6.2      # Testing framework
black >= 21.0      # Code formatting
flake8 >= 3.9      # Style checking
```

---

## 13. Limitations and Assumptions

### 13.1 Model Assumptions

1. **Point neuron approximation**: Spatial structure simplified
2. **Deterministic dynamics**: Between noise applications
3. **Independent pathways**: Except for lateral inhibition
4. **Linear summation**: Of individual pathway outputs (as null hypothesis)

### 13.2 Noise Model Limitations

- **Gaussian distribution**: May not capture all biological variability
- **Independent trials**: No correlation between successive trials
- **Constant CV**: Across all frequencies and pathways

### 13.3 Temporal Resolution

- **Discrete time steps**: 0.1 ms minimum resolution
- **Spike timing**: Precise to nearest time step
- **Phase offsets**: Fixed values (not stochastic)

---

## 14. Extensions and Future Work

### 14.1 Potential Enhancements

1. **Dendritic compartments**: Spatially extended models
2. **Calcium dynamics**: Include postsynaptic integration
3. **Spike timing**: Full spiking neuron models
4. **Network effects**: Population-level dynamics

### 14.2 Additional Analyses

1. **Information theory**: Mutual information between pathways
2. **Optimal stimulation**: Parameter space exploration
3. **Pathology models**: Disease state simulations
4. **Learning rules**: Plasticity and adaptation

---

## 15. References

### 15.1 Primary Model

- Tsodyks, M., & Markram, H. (1997). The neural code between neocortical pyramidal neurons depends on neurotransmitter release probability. *PNAS*, 94(2), 719-723.

### 15.2 Experimental Validation

- Hayakawa, H., et al. (2014). Layer-dependent synaptic processing in the rat entorhinal cortex. *Hippocampus*, 24(5), 571-584.
- Nakajima, N., et al. (2024). Modification of temporal pattern sensitivity for inputs from medial entorhinal cortex by lateral inputs in hippocampal granule cells. *Cogn Neurodyn*, 18(3), 1047-1059.

### 15.3 Statistical Methods

- Efron, B., & Tibshirani, R. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Zar, J.H. (2010). *Biostatistical Analysis* (5th ed.). Pearson.

---

## 16. Data Availability

All simulation code, raw data, and analysis scripts are publicly available at:

**GitHub Repository**: https://github.com/tckamijo/dentate_pathway_integration  
**Zenodo Archive**: https://doi.org/10.5281/zenodo.17276481

---

## 17. Contact for Methodology Questions

For technical questions about the implementation:

**Tadanobu C. Kamijo, Ph.D.**  
Email: tadanobu@cs.u-ryukyu.ac.jp  
GitHub Issues: https://github.com/tckamijo/dentate_pathway_integration/issues

---

**Last Updated**: October 2025  
**Version**: 2.0.0 (30-trial statistical validation)

---

This comprehensive methodology ensures full reproducibility of all results presented in the manuscript and supplementary materials.
