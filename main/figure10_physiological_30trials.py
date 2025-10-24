"""
Figure 10 - Three-Pathway Integration at Physiological Frequencies
===================================================================
Testing at theta (8Hz), beta (15Hz), low gamma (30Hz), and high gamma (40Hz)
WITH 30 TRIALS AND BIOLOGICAL NOISE (CV=0.20)

Filename: figure10_physiological_30trials.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
from datetime import datetime
from scipy import stats
from PIL import Image

def setup_logging():
    """Setup logging for final analysis"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"figure10_physiological_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*70)
    logging.info("FIGURE 10: PHYSIOLOGICAL FREQUENCY ANALYSIS (30 TRIALS)")
    logging.info("Testing at theta, beta, and gamma frequencies")
    logging.info("="*70)
    
    return log_filename

class TsodyksMarkramSynapse:
    """Standard TM synapse with noise option"""
    
    def __init__(self, tau_rec, tau_facil, U, name="synapse"):
        self.tau_rec = float(tau_rec)
        self.tau_facil = float(tau_facil)
        self.U = float(U)
        self.name = name
        self.reset()
        
    def reset(self):
        self.x = 1.0
        self.u = self.U
        self.last_time = 0.0
    
    def stimulate(self, time, cv=0.0):
        """
        Stimulate synapse with optional biological noise
        
        Parameters:
        -----------
        time : float
            Stimulation time (ms)
        cv : float
            Coefficient of variation for noise (default 0.0)
        """
        dt = time - self.last_time
        
        if dt > 0:
            self.x = 1 - (1 - self.x) * np.exp(-dt / self.tau_rec)
            self.u = self.U + (self.u - self.U) * np.exp(-dt / self.tau_facil)
        
        response = self.u * self.x
        
        # Add biological noise if cv > 0
        if cv > 0:
            noise_factor = np.random.normal(1.0, cv)
            noise_factor = max(0.01, noise_factor)  # Prevent negative responses
            response *= noise_factor
        
        self.x = self.x * (1 - self.u)
        self.u = self.u + self.U * (1 - self.u)
        self.last_time = time
        
        return response

class DecomposedPDSynapse:
    """PD with continuous frequency response and noise"""
    
    def __init__(self, frequency):
        self.frequency = frequency
        self.components = {
            'SuM': TsodyksMarkramSynapse(350.0, 45.0, 0.28, "SuM"),
            'MS': TsodyksMarkramSynapse(180.0, 85.0, 0.18, "MS"),
            'MC': TsodyksMarkramSynapse(480.0, 25.0, 0.38, "MC")
        }
        self.weights = self.get_frequency_weights(frequency)
        
    def get_frequency_weights(self, frequency):
        weights = {}
        
        # Continuous functions instead of sharp threshold
        weights['SuM'] = np.exp(-((frequency - 8)**2) / (2 * 5**2))
        weights['MS'] = 0.2 + 0.8 * (1 / (1 + np.exp(-(frequency - 25) / 5)))
        weights['MC'] = 0.3 + 0.7 * min(frequency / 40, 1.0)
        
        total = sum(weights.values())
        for key in weights:
            weights[key] /= total
            
        return weights
    
    def reset(self):
        for comp in self.components.values():
            comp.reset()
    
    def stimulate(self, time, cv=0.0):
        response = 0.0
        for comp_name, synapse in self.components.items():
            comp_response = synapse.stimulate(time, cv=cv)
            response += self.weights[comp_name] * comp_response
        return response

class LateralInhibition:
    """Frequency-dependent lateral inhibition"""
    
    def __init__(self, frequency):
        self.decay_time = 10.0
        self.delay = 0.8
        
        # Smooth frequency dependence
        self.strength = 0.3 + 0.2 * (frequency / 40.0)
        self.shunting_factor = 2.0
        
    def apply(self, response, time, dd_activity):
        inhibition = 0.0
        
        for dd_time, dd_strength in dd_activity:
            time_diff = time - dd_time - self.delay
            
            if 0 < time_diff < 5 * self.decay_time:
                temporal_factor = np.exp(-time_diff / self.decay_time)
                inhibition += dd_strength * self.strength * temporal_factor
        
        return response / (1 + inhibition * self.shunting_factor)

def simulate_integration_single_trial(frequency, model_config, n_pulses=10, cv=0.0):
    """Simulate single trial of three-pathway integration"""
    
    # Optimal phase offsets from validation
    phase_offsets = {'DD': 0, 'MD': 3, 'PD': 1.5}
    
    # Create synapses
    synapses = {
        'DD': TsodyksMarkramSynapse(248.0, 133.0, 0.20, "DD"),
        'MD': TsodyksMarkramSynapse(3977.0, 27.0, 0.30, "MD")
    }
    
    # PD synapse selection
    if model_config['decomposed_PD']:
        synapses['PD'] = DecomposedPDSynapse(frequency)
    else:
        # Simple PD with smooth transition
        tau_rec = 460 - (460 - 184) * (frequency / 40)
        tau_facil = 20 + (52.5 - 20) * (frequency / 40)
        U = 0.32 - (0.32 - 0.2135) * (frequency / 40)
        synapses['PD'] = TsodyksMarkramSynapse(tau_rec, tau_facil, U, "PD_simple")
    
    # Lateral inhibition
    lateral_inh = LateralInhibition(frequency) if model_config['lateral_inhibition'] else None
    
    # Generate stimuli
    all_stimuli = []
    dd_activity = []
    
    for pathway in ['DD', 'MD', 'PD']:
        isi = 1000.0 / frequency
        phase = phase_offsets[pathway]
        
        for pulse in range(n_pulses):
            time = pulse * isi + phase
            all_stimuli.append((time, pathway))
            
            if pathway == 'DD':
                dd_activity.append((time, 1.0))
    
    all_stimuli.sort(key=lambda x: x[0])
    
    # Calculate individual responses
    individual_responses = {}
    for pathway in ['DD', 'MD', 'PD']:
        synapses[pathway].reset()
        response = 0.0
        for pulse in range(n_pulses):
            time = pulse * (1000.0 / frequency) + phase_offsets[pathway]
            response += synapses[pathway].stimulate(time, cv=cv)
        individual_responses[pathway] = response
    
    linear_sum = sum(individual_responses.values())
    
    # Reset and calculate integrated response
    for synapse in synapses.values():
        synapse.reset()
    
    integrated_response = 0.0
    
    for time, pathway in all_stimuli:
        response = synapses[pathway].stimulate(time, cv=cv)
        
        if pathway == 'MD' and lateral_inh:
            response = lateral_inh.apply(response, time, dd_activity)
        
        integrated_response += response
    
    # Calculate interaction
    interaction_coeff = ((integrated_response - linear_sum) / linear_sum * 100) if linear_sum > 0 else 0
    
    return {
        'individual': individual_responses,
        'linear_sum': linear_sum,
        'integrated': integrated_response,
        'interaction': interaction_coeff
    }

def simulate_integration_multiple_trials(frequency, model_config, n_pulses=10, n_trials=30, cv=0.20):
    """
    Simulate multiple trials and return statistics
    
    Parameters:
    -----------
    frequency : float
        Stimulation frequency (Hz)
    model_config : dict
        Model configuration
    n_pulses : int
        Number of pulses per trial
    n_trials : int
        Number of trials (default 30)
    cv : float
        Coefficient of variation for noise (default 0.20)
    """
    all_results = []
    
    for trial in range(n_trials):
        result = simulate_integration_single_trial(frequency, model_config, n_pulses, cv)
        all_results.append(result)
    
    # Calculate statistics
    interactions = [r['interaction'] for r in all_results]
    linear_sums = [r['linear_sum'] for r in all_results]
    integrated_responses = [r['integrated'] for r in all_results]
    
    return {
        'frequency': frequency,
        'interaction_mean': np.mean(interactions),
        'interaction_sem': stats.sem(interactions),
        'linear_sum_mean': np.mean(linear_sums),
        'integrated_mean': np.mean(integrated_responses)
    }

def run_physiological_analysis(n_trials=30, cv=0.20):
    """Test at physiologically relevant frequencies with multiple trials"""
    print(f"\nTesting at physiological frequencies ({n_trials} trials, CV={cv})...")
    logging.info(f"Starting analysis with {n_trials} trials, CV={cv}")
    
    test_frequencies = {
        'Theta': 8,
        'Beta': 15,
        'Low_Gamma': 30,
        'High_Gamma': 40
    }
    
    model_configs = {
        'Simple': {'decomposed_PD': False, 'lateral_inhibition': False},
        'Decomposed_PD': {'decomposed_PD': True, 'lateral_inhibition': False},
        'Lateral_Inh': {'decomposed_PD': False, 'lateral_inhibition': True},
        'Complete': {'decomposed_PD': True, 'lateral_inhibition': True}
    }
    
    results = []
    
    for freq_name, freq in test_frequencies.items():
        print(f"\n{freq_name} ({freq}Hz):")
        logging.info(f"Processing {freq_name} ({freq}Hz)")
        
        for model_name, config in model_configs.items():
            result = simulate_integration_multiple_trials(freq, config, n_pulses=10, 
                                                         n_trials=n_trials, cv=cv)
            
            results.append({
                'frequency_band': freq_name,
                'frequency': freq,
                'model': model_name,
                'interaction_mean': result['interaction_mean'],
                'interaction_sem': result['interaction_sem'],
                'linear_sum': result['linear_sum_mean'],
                'integrated': result['integrated_mean']
            })
            
            print(f"  {model_name}: {result['interaction_mean']:.2f} ± {result['interaction_sem']:.2f}%")
            logging.info(f"  {model_name}: {result['interaction_mean']:.2f} ± {result['interaction_sem']:.2f}%")
    
    logging.info(f"Completed {len(results)} conditions")
    return results

def create_physiological_figure(results, n_trials=30, cv=0.20):
    """Create figure for physiological frequencies with error bars"""
    print("\nCreating Figure 10...")
    logging.info("Generating Figure 10")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Figure 10: Three-Pathway Integration at Physiological Frequencies\nMean ± SEM (n={n_trials} trials, CV={cv})', 
                 fontsize=14, fontweight='bold')
    
    df = pd.DataFrame(results)
    
    frequencies = [8, 15, 30, 40]
    freq_labels = ['Theta\n(8Hz)', 'Beta\n(15Hz)', 'Low Gamma\n(30Hz)', 'High Gamma\n(40Hz)']
    models = ['Simple', 'Decomposed_PD', 'Lateral_Inh', 'Complete']
    model_colors = {'Simple': '#252525', 'Decomposed_PD': '#636363', 
                   'Lateral_Inh': '#969696', 'Complete': '#d9d9d9'}
    
    # Panel A: All models comparison WITH ERROR BARS
    ax1 = axes[0, 0]
    
    x = np.arange(len(frequencies))
    width = 0.2
    
    # Define hatching patterns for models
    model_hatches = {'Simple': '', 'Decomposed_PD': '///', 
                     'Lateral_Inh': '\\\\\\', 'Complete': 'xxx'}
    
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        values = [model_data[model_data['frequency'] == f]['interaction_mean'].values[0] 
                 for f in frequencies]
        errors = [model_data[model_data['frequency'] == f]['interaction_sem'].values[0] 
                 for f in frequencies]
        
        bars = ax1.bar(x + i*width - 1.5*width, values, width, yerr=errors,
                      label=model.replace('_', ' '), color=model_colors[model], 
                      alpha=0.8, edgecolor='black', linewidth=1.5, capsize=3, error_kw={'linewidth': 1.5})
        
        # Apply hatching pattern
        for bar in bars:
            bar.set_hatch(model_hatches[model])
    
    ax1.set_xlabel('Frequency Band', fontsize=11)
    ax1.set_ylabel('Interaction Coefficient (%)', fontsize=11)
    ax1.set_title('A. Model Comparison Across Frequencies', fontweight='bold', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(freq_labels)
    ax1.legend(loc='lower left', fontsize=9)
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Panel B: Complete model detailed WITH ERROR BARS
    ax2 = axes[0, 1]
    
    complete_data = df[df['model'] == 'Complete']
    
    interactions = [complete_data[complete_data['frequency'] == f]['interaction_mean'].values[0] 
                   for f in frequencies]
    errors = [complete_data[complete_data['frequency'] == f]['interaction_sem'].values[0] 
             for f in frequencies]
    
    freq_colors = ['#252525', '#636363', '#969696', '#d9d9d9']
    freq_hatches = ['', '///', '\\\\\\', 'xxx']  # Theta, Beta, Low Gamma, High Gamma
    
    bars = ax2.bar(range(len(frequencies)), interactions, yerr=errors,
                   color=freq_colors, alpha=0.8, edgecolor='black', linewidth=1.5, 
                   capsize=3, error_kw={'linewidth': 1.5})
    
    # Apply hatching patterns
    for bar, hatch in zip(bars, freq_hatches):
        bar.set_hatch(hatch)
    
    for bar, val, err in zip(bars, interactions, errors):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + err,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('Frequency Band', fontsize=11)
    ax2.set_ylabel('Interaction Coefficient (%)', fontsize=11)
    ax2.set_title('B. Complete Model Response', fontweight='bold', fontsize=12)
    ax2.set_xticks(range(len(frequencies)))
    ax2.set_xticklabels(freq_labels)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Panel C: Component contributions
    ax3 = axes[1, 0]
    
    contributions = []
    contribution_errors = []
    
    for freq in frequencies:
        simple = df[(df['model'] == 'Simple') & (df['frequency'] == freq)]['interaction_mean'].values[0]
        decomp = df[(df['model'] == 'Decomposed_PD') & (df['frequency'] == freq)]['interaction_mean'].values[0]
        lateral = df[(df['model'] == 'Lateral_Inh') & (df['frequency'] == freq)]['interaction_mean'].values[0]
        complete = df[(df['model'] == 'Complete') & (df['frequency'] == freq)]['interaction_mean'].values[0]
        
        # Get SEMs for error propagation
        simple_sem = df[(df['model'] == 'Simple') & (df['frequency'] == freq)]['interaction_sem'].values[0]
        decomp_sem = df[(df['model'] == 'Decomposed_PD') & (df['frequency'] == freq)]['interaction_sem'].values[0]
        lateral_sem = df[(df['model'] == 'Lateral_Inh') & (df['frequency'] == freq)]['interaction_sem'].values[0]
        complete_sem = df[(df['model'] == 'Complete') & (df['frequency'] == freq)]['interaction_sem'].values[0]
        
        contributions.append({
            'PD_decomp': decomp - simple,
            'Lateral_inh': lateral - simple,
            'Complete': complete
        })
        
        # Error propagation for differences
        contribution_errors.append({
            'PD_decomp': np.sqrt(decomp_sem**2 + simple_sem**2),
            'Lateral_inh': np.sqrt(lateral_sem**2 + simple_sem**2),
            'Complete': complete_sem
        })
    
    x = np.arange(len(frequencies))
    width = 0.25
    
    pd_contrib = [c['PD_decomp'] for c in contributions]
    inh_contrib = [c['Lateral_inh'] for c in contributions]
    complete_contrib = [c['Complete'] for c in contributions]
    
    pd_errors = [e['PD_decomp'] for e in contribution_errors]
    inh_errors = [e['Lateral_inh'] for e in contribution_errors]
    complete_errors = [e['Complete'] for e in contribution_errors]
    
    bars1 = ax3.bar(x - width, pd_contrib, width, yerr=pd_errors, label='PD decomposition', 
                    color='#636363', alpha=0.8, edgecolor='black', linewidth=1.5, capsize=3)
    bars2 = ax3.bar(x, inh_contrib, width, yerr=inh_errors, label='Lateral inhibition', 
                    color='#969696', alpha=0.8, edgecolor='black', linewidth=1.5, capsize=3)
    bars3 = ax3.bar(x + width, complete_contrib, width, yerr=complete_errors,
                    label='Complete', color='#d9d9d9', alpha=0.8, edgecolor='black', linewidth=1.5, capsize=3)
    
    # Add hatching patterns
    for bar in bars1:
        bar.set_hatch('///')
    for bar in bars2:
        bar.set_hatch('\\\\\\')
    for bar in bars3:
        bar.set_hatch('xxx')
    
    ax3.set_xlabel('Frequency Band', fontsize=11)
    ax3.set_ylabel('Effect Size (%)', fontsize=11)
    ax3.set_title('C. Mechanism Contributions', fontweight='bold', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(freq_labels)
    ax3.legend(fontsize=9)
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel D: Frequency band effects (heatmap)
    ax4 = axes[1, 1]
    
    # Create heatmap data
    models_ordered = ['Simple', 'Decomposed_PD', 'Lateral_Inh', 'Complete']
    heatmap_data = np.zeros((len(models_ordered), len(frequencies)))
    
    for i, model in enumerate(models_ordered):
        for j, freq in enumerate(frequencies):
            val = df[(df['model'] == model) & (df['frequency'] == freq)]['interaction_mean'].values[0]
            heatmap_data[i, j] = val
    
    im = ax4.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', vmin=-15, vmax=5)
    
    ax4.set_xticks(range(len(frequencies)))
    ax4.set_xticklabels(freq_labels)
    ax4.set_yticks(range(len(models_ordered)))
    ax4.set_yticklabels([m.replace('_', ' ') for m in models_ordered])
    ax4.set_xlabel('Frequency Band', fontsize=11)
    ax4.set_ylabel('Model Type', fontsize=11)
    ax4.set_title('D. Interaction Coefficient Heatmap', fontweight='bold', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Interaction (%)', rotation=270, labelpad=15)
    
    # Add text annotations
    for i in range(len(models_ordered)):
        for j in range(len(frequencies)):
            text = ax4.text(j, i, f'{heatmap_data[i, j]:.1f}',
                          ha="center", va="center", 
                          color="white" if abs(heatmap_data[i, j]) > 7 else "black",
                          fontsize=9)
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_filename = f'Figure10_Physiological_variability_{timestamp}.png'
    tiff_filename = f'Figure10_Physiological_variability_{timestamp}.tiff'
    csv_filename = f'Figure10_data_variability_{timestamp}.csv'
    
    fig.savefig(png_filename, dpi=600, bbox_inches='tight')
    print(f"Saved: {png_filename}")
    logging.info(f"Saved: {png_filename}")
    
    # Convert to TIFF
    Image.open(png_filename).save(tiff_filename, dpi=(600, 600), compression='tiff_lzw')
    print(f"Saved: {tiff_filename}")
    logging.info(f"Saved: {tiff_filename}")
    
    plt.close(fig)
    
    # Save data
    df.to_csv(csv_filename, index=False)
    print(f"Saved: {csv_filename}")
    logging.info(f"Saved: {csv_filename}")
    
    return df, png_filename, tiff_filename, csv_filename

def main():
    """Main execution"""
    print("="*70)
    print("FIGURE 10: PHYSIOLOGICAL FREQUENCY ANALYSIS (30 TRIALS)")
    print("Avoiding problematic 20Hz transition")
    print("="*70)
    
    log_file = setup_logging()
    print(f"Log file: {log_file}")
    
    try:
        # Run analysis with 30 trials and CV=0.20
        results = run_physiological_analysis(n_trials=30, cv=0.20)
        
        # Create figure
        df, png_file, tiff_file, csv_file = create_physiological_figure(results, n_trials=30, cv=0.20)
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        complete_data = df[df['model'] == 'Complete']
        
        for _, row in complete_data.iterrows():
            print(f"{row['frequency_band']:12s} ({row['frequency']:2d}Hz): "
                  f"{row['interaction_mean']:+.2f} ± {row['interaction_sem']:.2f}%")
        
        max_effect = complete_data['interaction_mean'].min()
        print(f"\nMaximum effect: {max_effect:.2f}%")
        
        if abs(max_effect) > 5:
            print("SUCCESS: Significant nonlinear integration at physiological frequencies")
        else:
            print("NOTE: Moderate effects observed, consistent with balanced integration")
        
        print("="*70)
        print(f"\nFiles saved:")
        print(f"  PNG: {png_file}")
        print(f"  TIFF: {tiff_file}")
        print(f"  CSV: {csv_file}")
        print(f"  LOG: {log_file}")
        print("="*70)
        
        logging.info("="*70)
        logging.info("ANALYSIS COMPLETE")
        logging.info(f"Files: {png_file}, {tiff_file}, {csv_file}")
        logging.info("="*70)
        
        return results
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        logging.error(error_msg, exc_info=True)
        raise

if __name__ == "__main__":
    results = main()
