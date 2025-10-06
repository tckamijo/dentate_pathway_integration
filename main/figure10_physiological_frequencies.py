"""
Figure 10 - Three-Pathway Integration at Physiological Frequencies
===================================================================
Testing at theta (8Hz), beta (15Hz), low gamma (30Hz), and high gamma (40Hz)
Avoiding problematic 20Hz transition region

Filename: figure10_physiological_frequencies.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
from datetime import datetime

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
    logging.info("FIGURE 10: PHYSIOLOGICAL FREQUENCY ANALYSIS")
    logging.info("Testing at theta, beta, and gamma frequencies")
    logging.info("="*70)
    
    return log_filename

class TsodyksMarkramSynapse:
    """Standard TM synapse"""
    
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
    
    def stimulate(self, time):
        dt = time - self.last_time
        
        if dt > 0:
            self.x = 1 - (1 - self.x) * np.exp(-dt / self.tau_rec)
            self.u = self.U + (self.u - self.U) * np.exp(-dt / self.tau_facil)
        
        response = self.u * self.x
        
        self.x = self.x * (1 - self.u)
        self.u = self.u + self.U * (1 - self.u)
        self.last_time = time
        
        return response

class DecomposedPDSynapse:
    """PD with continuous frequency response"""
    
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
    
    def stimulate(self, time):
        response = 0.0
        for comp_name, synapse in self.components.items():
            comp_response = synapse.stimulate(time)
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

def simulate_integration(frequency, model_config, n_pulses=10):
    """Simulate three-pathway integration at specific frequency"""
    
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
            response += synapses[pathway].stimulate(time)
        individual_responses[pathway] = response
    
    linear_sum = sum(individual_responses.values())
    
    # Reset and calculate integrated response
    for synapse in synapses.values():
        synapse.reset()
    
    integrated_response = 0.0
    
    for time, pathway in all_stimuli:
        response = synapses[pathway].stimulate(time)
        
        if pathway == 'MD' and lateral_inh:
            response = lateral_inh.apply(response, time, dd_activity)
        
        integrated_response += response
    
    # Calculate interaction
    interaction_coeff = ((integrated_response - linear_sum) / linear_sum * 100) if linear_sum > 0 else 0
    
    return {
        'frequency': frequency,
        'individual': individual_responses,
        'linear_sum': linear_sum,
        'integrated': integrated_response,
        'interaction': interaction_coeff
    }

def run_physiological_analysis():
    """Test at physiologically relevant frequencies"""
    print("\nTesting at physiological frequencies...")
    
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
        
        for model_name, config in model_configs.items():
            result = simulate_integration(freq, config)
            
            results.append({
                'frequency_band': freq_name,
                'frequency': freq,
                'model': model_name,
                'interaction': result['interaction'],
                'linear_sum': result['linear_sum'],
                'integrated': result['integrated']
            })
            
            print(f"  {model_name}: {result['interaction']:.2f}%")
    
    return results

def create_physiological_figure(results):
    """Create figure for physiological frequencies"""
    print("\nCreating Figure 10...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Figure 10: Three-Pathway Integration at Physiological Frequencies', 
                 fontsize=14, fontweight='bold')
    
    df = pd.DataFrame(results)
    
    frequencies = [8, 15, 30, 40]
    freq_labels = ['Theta\n(8Hz)', 'Beta\n(15Hz)', 'Low Gamma\n(30Hz)', 'High Gamma\n(40Hz)']
    models = ['Simple', 'Decomposed_PD', 'Lateral_Inh', 'Complete']
    model_colors = {'Simple': 'gray', 'Decomposed_PD': 'lightblue', 
                   'Lateral_Inh': 'lightcoral', 'Complete': 'purple'}
    
    # Panel A: All models comparison
    ax1 = axes[0, 0]
    
    x = np.arange(len(frequencies))
    width = 0.2
    
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        values = [model_data[model_data['frequency'] == f]['interaction'].values[0] 
                 for f in frequencies]
        ax1.bar(x + i*width - 1.5*width, values, width, 
               label=model.replace('_', ' '), color=model_colors[model], alpha=0.7)
    
    ax1.set_xlabel('Frequency Band')
    ax1.set_ylabel('Interaction Coefficient (%)')
    ax1.set_title('A. Model Comparison Across Frequencies', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(freq_labels)
    ax1.legend(loc='lower left', fontsize=9)
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Complete model detailed
    ax2 = axes[0, 1]
    
    complete_data = df[df['model'] == 'Complete']
    
    interactions = [complete_data[complete_data['frequency'] == f]['interaction'].values[0] 
                   for f in frequencies]
    
    bars = ax2.bar(range(len(frequencies)), interactions, 
                   color=['green', 'blue', 'orange', 'red'], alpha=0.7)
    
    for bar, val in zip(bars, interactions):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%', ha='center', va='bottom' if height >= 0 else 'top')
    
    ax2.set_xlabel('Frequency Band')
    ax2.set_ylabel('Interaction Coefficient (%)')
    ax2.set_title('B. Complete Model Response', fontweight='bold')
    ax2.set_xticks(range(len(frequencies)))
    ax2.set_xticklabels(freq_labels)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Component contributions
    ax3 = axes[1, 0]
    
    contributions = []
    for freq in frequencies:
        simple = df[(df['model'] == 'Simple') & (df['frequency'] == freq)]['interaction'].values[0]
        decomp = df[(df['model'] == 'Decomposed_PD') & (df['frequency'] == freq)]['interaction'].values[0]
        lateral = df[(df['model'] == 'Lateral_Inh') & (df['frequency'] == freq)]['interaction'].values[0]
        complete = df[(df['model'] == 'Complete') & (df['frequency'] == freq)]['interaction'].values[0]
        
        contributions.append({
            'PD_decomp': decomp - simple,
            'Lateral_inh': lateral - simple,
            'Complete': complete
        })
    
    x = np.arange(len(frequencies))
    width = 0.25
    
    pd_contrib = [c['PD_decomp'] for c in contributions]
    inh_contrib = [c['Lateral_inh'] for c in contributions]
    
    ax3.bar(x - width, pd_contrib, width, label='PD decomposition', color='blue', alpha=0.7)
    ax3.bar(x, inh_contrib, width, label='Lateral inhibition', color='red', alpha=0.7)
    ax3.bar(x + width, [c['Complete'] for c in contributions], width, 
           label='Complete', color='purple', alpha=0.7)
    
    ax3.set_xlabel('Frequency Band')
    ax3.set_ylabel('Effect Size (%)')
    ax3.set_title('C. Mechanism Contributions', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(freq_labels)
    ax3.legend()
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax3.grid(True, alpha=0.3)
    
    # Panel D: Frequency band effects
    ax4 = axes[1, 1]
    
    # Create heatmap data
    models_ordered = ['Simple', 'Decomposed_PD', 'Lateral_Inh', 'Complete']
    heatmap_data = np.zeros((len(models_ordered), len(frequencies)))
    
    for i, model in enumerate(models_ordered):
        for j, freq in enumerate(frequencies):
            val = df[(df['model'] == model) & (df['frequency'] == freq)]['interaction'].values[0]
            heatmap_data[i, j] = val
    
    im = ax4.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', vmin=-15, vmax=5)
    
    ax4.set_xticks(range(len(frequencies)))
    ax4.set_xticklabels(freq_labels)
    ax4.set_yticks(range(len(models_ordered)))
    ax4.set_yticklabels(models_ordered)
    ax4.set_xlabel('Frequency Band')
    ax4.set_ylabel('Model Type')
    ax4.set_title('D. Interaction Coefficient Heatmap', fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Interaction (%)', rotation=270, labelpad=15)
    
    # Add text annotations
    for i in range(len(models_ordered)):
        for j in range(len(frequencies)):
            text = ax4.text(j, i, f'{heatmap_data[i, j]:.1f}',
                          ha="center", va="center", 
                          color="white" if abs(heatmap_data[i, j]) > 7 else "black")
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'Figure10_Physiological_{timestamp}.png'
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    
    plt.show()
    
    return df

def main():
    """Main execution"""
    print("="*70)
    print("FIGURE 10: PHYSIOLOGICAL FREQUENCY ANALYSIS")
    print("Avoiding problematic 20Hz transition")
    print("="*70)
    
    log_file = setup_logging()
    print(f"Log file: {log_file}")
    
    try:
        # Run analysis
        results = run_physiological_analysis()
        
        # Create figure
        df = create_physiological_figure(results)
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        complete_data = df[df['model'] == 'Complete']
        
        for _, row in complete_data.iterrows():
            print(f"{row['frequency_band']:12s} ({row['frequency']:2d}Hz): {row['interaction']:+.2f}%")
        
        max_effect = complete_data['interaction'].min()
        print(f"\nMaximum effect: {max_effect:.2f}%")
        
        if abs(max_effect) > 5:
            print("SUCCESS: Significant nonlinear integration at physiological frequencies")
        else:
            print("NOTE: Moderate effects observed, consistent with balanced integration")
        
        print("="*70)
        
        return results
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        logging.error(error_msg, exc_info=True)
        raise

if __name__ == "__main__":
    results = main()
