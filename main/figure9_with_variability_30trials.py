"""
Figure 9 - Lateral Inhibition with Biological Variability (FIXED VERSION)
==========================================================================
30 trials per condition with CV = 0.20
Enhanced lateral inhibition with temporal dynamics and frequency dependence

BIOLOGICAL VARIABILITY REFERENCES:
- Nusser et al., 2001: mIPSC variability ~30%
- Smith et al., 2003: mEPSC amplitude CV = 0.69-0.75
- Implemented as CV = 0.20 (20%, conservative estimate)

Filename: figure9_with_variability_30trials_fixed.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
from datetime import datetime
from scipy import stats
from PIL import Image

def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"figure9_variability_{timestamp}.log"
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                       handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])
    logging.info("="*70)
    logging.info("FIGURE 9: LATERAL INHIBITION (30 trials, CV=0.20)")
    logging.info("="*70)
    return log_filename

plt.rcParams.update({"font.family": "Arial", "font.size": 12, "axes.linewidth": 1.2})

class TsodyksMarkramSynapse:
    def __init__(self, tau_rec, tau_facil, U, name="synapse"):
        self.base_tau_rec, self.base_tau_facil, self.base_U = float(tau_rec), float(tau_facil), float(U)
        self.name = name
        self.reset()
    
    def reset(self):
        self.x, self.u, self.last_time = 1.0, self.base_U, 0.0
        self.tau_rec, self.tau_facil, self.U = self.base_tau_rec, self.base_tau_facil, self.base_U
        self.response_history = []
    
    def set_trial_parameters(self, cv=0.20):
        self.tau_rec = max(np.random.normal(self.base_tau_rec, self.base_tau_rec * cv), self.base_tau_rec * 0.5)
        self.tau_facil = max(np.random.normal(self.base_tau_facil, self.base_tau_facil * cv), self.base_tau_facil * 0.5)
        self.U = np.clip(np.random.normal(self.base_U, self.base_U * cv), 0.05, 0.95)
    
    def stimulate(self, time):
        dt = time - self.last_time
        if dt > 0:
            self.x = 1 - (1 - self.x) * np.exp(-dt / self.tau_rec)
            self.u = self.U + (self.u - self.U) * np.exp(-dt / self.tau_facil)
        response = self.u * self.x
        self.x, self.u = self.x * (1 - self.u), self.u + self.U * (1 - self.u)
        self.last_time = time
        self.response_history.append((time, response))
        return response

class DecomposedPDSynapse:
    """PD with frequency-dependent components"""
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
        weights['SuM'] = np.exp(-((frequency - 8)**2) / (2 * 5**2))
        weights['MS'] = 0.2 if frequency < 20 else 0.2 + 0.8 * min((frequency - 20) / 20, 1.0)
        weights['MC'] = 0.3 + 0.7 * min(frequency / 40, 1.0)
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}
    
    def reset(self):
        for comp in self.components.values():
            comp.reset()
    
    def set_trial_parameters(self, cv=0.20):
        for comp in self.components.values():
            comp.set_trial_parameters(cv)
    
    def stimulate(self, time):
        return sum(self.weights[name] * syn.stimulate(time) 
                  for name, syn in self.components.items())

class EnhancedLateralInhibition:
    """Enhanced lateral inhibition (Nakajima et al. 2022)"""
    def __init__(self, decay_time=10.0, frequency=20.0):
        self.decay_time = decay_time
        self.strength = 0.3 if frequency < 20 else 0.45
        self.delay = 0.8
        self.shunting_factor = 2.0
        self.inhibition_history = []
    
    def apply_inhibition(self, excitatory_response, time, dd_activity, md_activity):
        lateral_inhibition = sum(
            dd_strength * self.strength * np.exp(-(time - dd_time - self.delay) / self.decay_time)
            for dd_time, dd_strength in dd_activity
            if 0 < time - dd_time - self.delay < 5 * self.decay_time
        )
        
        feedforward_inhibition = sum(
            md_strength * 0.1
            for md_time, md_strength in md_activity
            if abs(time - md_time) < 2.0
        )
        
        total_inhibition = lateral_inhibition + feedforward_inhibition
        inhibited_response = excitatory_response / (1 + total_inhibition * self.shunting_factor)
        self.inhibition_history.append((time, total_inhibition, lateral_inhibition, feedforward_inhibition))
        
        return inhibited_response

def simulate_enhanced_integration_single_trial(dd_freq, md_freq, pd_freq, n_pulses=10, 
                                               inhibition_params=None, cv=0.20):
    """Single trial simulation"""
    synapses = {
        'DD': TsodyksMarkramSynapse(248.0, 133.0, 0.20, "DD"),
        'MD': TsodyksMarkramSynapse(3977.0, 27.0, 0.30, "MD"),
        'PD': DecomposedPDSynapse(pd_freq)
    }
    
    for syn in synapses.values():
        if hasattr(syn, 'set_trial_parameters'):
            syn.set_trial_parameters(cv)
    
    if inhibition_params:
        mean_freq = np.mean([dd_freq, md_freq, pd_freq])
        lateral_inh = EnhancedLateralInhibition(decay_time=inhibition_params['decay'], frequency=mean_freq)
    else:
        lateral_inh = None
    
    all_stimuli = []
    dd_activity, md_activity = [], []
    
    for pathway, freq in [('DD', dd_freq), ('MD', md_freq), ('PD', pd_freq)]:
        if freq > 0:
            isi = 1000.0 / freq
            for pulse in range(n_pulses):
                time = pulse * isi
                all_stimuli.append((time, pathway))
                if pathway == 'DD': dd_activity.append((time, 1.0))
                elif pathway == 'MD': md_activity.append((time, 0.5))
    
    all_stimuli.sort()
    
    # Individual sum
    for syn in synapses.values(): syn.reset()
    individual_sum = sum(synapses[pathway].stimulate(time) for time, pathway in all_stimuli)
    
    # Integrated response with trace
    for syn in synapses.values(): syn.reset()
    integrated_response = 0.0
    integrated_trace = []
    
    for time, pathway in all_stimuli:
        response = synapses[pathway].stimulate(time)
        if pathway == 'MD' and lateral_inh is not None:
            response = lateral_inh.apply_inhibition(response, time, dd_activity, md_activity)
        integrated_response += response
        integrated_trace.append((time, pathway, response))
    
    interaction_coeff = (integrated_response - individual_sum) / individual_sum if individual_sum > 0 else 0.0
    
    return {
        'individual_sum': individual_sum, 
        'integrated_response': integrated_response, 
        'interaction_coefficient': interaction_coeff,
        'integrated_trace': integrated_trace
    }

def simulate_enhanced_integration_multiple_trials(dd_freq, md_freq, pd_freq, n_pulses=10,
                                                  inhibition_params=None, n_trials=30, cv=0.20):
    """Multiple trials simulation"""
    all_results = [
        simulate_enhanced_integration_single_trial(dd_freq, md_freq, pd_freq, n_pulses, inhibition_params, cv)
        for _ in range(n_trials)
    ]
    
    return {
        'individual_sum': np.mean([r['individual_sum'] for r in all_results]),
        'integrated_response': np.mean([r['integrated_response'] for r in all_results]),
        'interaction_coefficient': np.mean([r['interaction_coefficient'] for r in all_results]),
        'interaction_sem': stats.sem([r['interaction_coefficient'] for r in all_results]),
        'integrated_trace': all_results[0]['integrated_trace']  # Keep first trial for Panel D
    }

def run_enhanced_analysis(n_trials=30, cv=0.20):
    """Run analysis with variability"""
    print(f"Running lateral inhibition analysis ({n_trials} trials)...")
    logging.info(f"Starting analysis with {n_trials} trials, CV={cv}")
    
    decay_times = [0, 5, 10, 15]
    frequencies = [(5,5,5), (10,10,10), (15,15,15), (20,20,20), (25,25,25), (30,30,30), (35,35,35), (40,40,40)]
    
    results = []
    
    for decay_time in decay_times:
        for dd_freq, md_freq, pd_freq in frequencies:
            inhibition_params = {'decay': decay_time} if decay_time > 0 else None
            
            result = simulate_enhanced_integration_multiple_trials(
                dd_freq, md_freq, pd_freq, n_pulses=10, 
                inhibition_params=inhibition_params, n_trials=n_trials, cv=cv
            )
            
            results.append({
                'decay_time': decay_time,
                'DD_freq': dd_freq, 'MD_freq': md_freq, 'PD_freq': pd_freq,
                'mean_freq': np.mean([dd_freq, md_freq, pd_freq]),
                'interaction_coefficient': result['interaction_coefficient'],
                'interaction_sem': result['interaction_sem'],
                'individual_sum': result['individual_sum'],
                'integrated_response': result['integrated_response'],
                'trace_data': result['integrated_trace'] if decay_time == 10 else None
            })
    
    logging.info(f"Completed {len(results)} conditions")
    return results

def create_enhanced_figure(results):
    """Create 2x2 figure with correct Panel D"""
    print("Creating Figure 9...")
    logging.info("Generating Figure 9")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    n_trials = 30
    cv = 0.20
    
    fig.suptitle(f'Figure 9: Enhanced Lateral Inhibition Effects on Three-Pathway Integration\nMean ± SEM (n={n_trials} trials, CV={cv})', 
                fontsize=14, fontweight='bold')
    
    df = pd.DataFrame(results)
    
    # Panel A: Frequency-dependent effects
    ax1 = axes[0, 0]
    colors_map = {0: '#252525', 5: '#636363', 10: '#969696', 15: '#d9d9d9'}
    markers_map = {0: 'o', 5: 's', 10: '^', 15: 'D'}
    
    for decay_time in [0, 5, 10, 15]:
        subset = df[df['decay_time'] == decay_time]
        grouped = subset.groupby('mean_freq')['interaction_coefficient']
        freq_means = grouped.mean() * 100
        freq_sem = subset.groupby('mean_freq')['interaction_sem'].mean() * 100
        
        label = 'No inhibition' if decay_time == 0 else f'{decay_time}ms decay'
        ax1.errorbar(freq_means.index, freq_means.values, yerr=freq_sem.values,
                    marker=markers_map[decay_time], markersize=8, linewidth=2, capsize=3,
                    label=label, color=colors_map[decay_time], alpha=0.8)
    
    ax1.axvline(x=20, color='gray', linestyle=':', alpha=0.5, label='20Hz threshold')
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Mean Frequency (Hz)')
    ax1.set_ylabel('Interaction Coefficient (%)')
    ax1.set_title('A. Frequency-Dependent Inhibition', fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Decay time effects
    ax2 = axes[0, 1]
    decay_data, positions, colors = [], [], []
    color_map = ['#252525', '#636363', '#969696', '#d9d9d9']
    
    for i, freq in enumerate([10, 20, 30, 40]):
        for j, decay in enumerate([0, 5, 10, 15]):
            subset = df[(df['decay_time'] == decay) & (df['mean_freq'] == freq)]
            if not subset.empty:
                decay_data.append(subset['interaction_coefficient'].values[0] * 100)
                positions.append(i * 4 + j)
                colors.append(color_map[j])
    
    ax2.bar(positions, decay_data, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Frequency (Hz) / Decay Time (ms)')
    ax2.set_ylabel('Interaction Coefficient (%)')
    ax2.set_title('B. Frequency × Decay Time', fontweight='bold')
    ax2.set_xticks([1.5, 5.5, 9.5, 13.5])
    ax2.set_xticklabels(['10Hz', '20Hz', '30Hz', '40Hz'])
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3, axis='y')
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#252525', edgecolor='black', label='None'),
                      Patch(facecolor='#636363', edgecolor='black', label='5ms'),
                      Patch(facecolor='#969696', edgecolor='black', label='10ms'),
                      Patch(facecolor='#d9d9d9', edgecolor='black', label='15ms')]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    # Panel C: Effect size
    ax3 = axes[1, 0]
    no_inh = df[df['decay_time'] == 0].groupby('mean_freq')['interaction_coefficient'].mean() * 100
    colors_map = {5: '#636363', 10: '#969696', 15: '#d9d9d9'}
    markers_map = {5: 's', 10: '^', 15: 'D'}
    
    for decay in [5, 10, 15]:
        with_inh = df[df['decay_time'] == decay].groupby('mean_freq')['interaction_coefficient'].mean() * 100
        effect_size = no_inh - with_inh
        ax3.plot(effect_size.index, effect_size.values, 
                marker=markers_map[decay], markersize=8, linewidth=2,
                color=colors_map[decay], label=f'{decay}ms', alpha=0.8)
    
    ax3.axvline(x=20, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Inhibition Effect Size (%)')
    ax3.set_title('C. Magnitude of Inhibition Effect', fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Panel D: Temporal dynamics (FIXED - from original code)
    ax4 = axes[1, 1]
    
    # Show example at 20Hz with 10ms decay
    example_data = df[(df['decay_time'] == 10) & (df['mean_freq'] == 20)]
    
    if not example_data.empty and example_data.iloc[0]['trace_data'] is not None:
        trace = example_data.iloc[0]['trace_data']
        
        # Separate by pathway
        dd_times = [t for t, p, _ in trace if p == 'DD']
        dd_responses = [r for t, p, r in trace if p == 'DD']
        
        md_times = [t for t, p, _ in trace if p == 'MD']
        md_responses = [r for t, p, r in trace if p == 'MD']
        
        pd_times = [t for t, p, _ in trace if p == 'PD']
        pd_responses = [r for t, p, r in trace if p == 'PD']
        
        ax4.scatter(dd_times, dd_responses, c='red', s=60, alpha=0.7, label='DD', marker='o')
        ax4.scatter(md_times, md_responses, c='blue', s=60, alpha=0.7, label='MD (inhibited)', marker='s')
        ax4.scatter(pd_times, pd_responses, c='green', s=60, alpha=0.7, label='PD', marker='^')
        
        ax4.set_xlabel('Time (ms)')
        ax4.set_ylabel('Synaptic Response')
        ax4.set_title('D. Temporal Dynamics at 20Hz (10ms inhibition)', fontweight='bold')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(-10, max(dd_times + md_times + pd_times) + 10)
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_filename = f'Figure9_inhibition_variability_{timestamp}.png'
    tiff_filename = f'Figure9_inhibition_variability_{timestamp}.tiff'
    csv_filename = f'Figure9_data_variability_{timestamp}.csv'
    
    fig.savefig(png_filename, dpi=600, bbox_inches='tight')
    Image.open(png_filename).save(tiff_filename, dpi=(600, 600), compression='tiff_lzw')
    plt.close(fig)
    
    pd.DataFrame(results).to_csv(csv_filename, index=False)
    
    logging.info(f"Saved: {png_filename}, {tiff_filename}, {csv_filename}")
    return png_filename, tiff_filename, csv_filename

def main():
    print("="*70)
    print("FIGURE 9: LATERAL INHIBITION (30 trials, CV=0.20)")
    print("="*70)
    
    log_file = setup_logging()
    
    try:
        results = run_enhanced_analysis(n_trials=30, cv=0.20)
        png_file, tiff_file, csv_file = create_enhanced_figure(results)
        
        logging.info("="*70)
        logging.info("COMPLETE")
        logging.info(f"Files: {png_file}, {tiff_file}, {csv_file}")
        logging.info("="*70)
        
        print(f"\nLog: {log_file}")
        print("Success!")
        
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
