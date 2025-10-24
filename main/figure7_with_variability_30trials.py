"""
Figure 7 - Mechanism Analysis with Biological Variability
==========================================================
30 trials per condition with CV = 0.20
Analysis of resource competition, temporal dynamics, and frequency effects

BIOLOGICAL VARIABILITY REFERENCES:
- Nusser et al., 2001: mIPSC variability ~30%
- Smith et al., 2003: mEPSC amplitude CV = 0.69-0.75
- Implemented as CV = 0.20 (20%, conservative estimate)

Filename: figure7_with_variability_30trials.py
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
    log_filename = f"figure7_variability_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*70)
    logging.info("FIGURE 7: MECHANISM ANALYSIS WITH VARIABILITY")
    logging.info("30 trials per condition, CV = 0.20")
    logging.info("="*70)
    
    return log_filename

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "axes.linewidth": 1.2,
})

class TsodyksMarkramSynapse:
    """Enhanced synapse with history tracking and variability"""
    
    def __init__(self, tau_rec, tau_facil, U, pathway_name="synapse"):
        self.base_tau_rec = float(tau_rec)
        self.base_tau_facil = float(tau_facil)
        self.base_U = float(U)
        self.pathway_name = pathway_name
        self.reset()
    
    def reset(self):
        self.x = 1.0
        self.u = self.base_U
        self.last_time = 0.0
        self.tau_rec = self.base_tau_rec
        self.tau_facil = self.base_tau_facil
        self.U = self.base_U
        self.history = {'times': [], 'x_values': [], 'u_values': [], 'responses': []}
    
    def set_trial_parameters(self, cv=0.20):
        self.tau_rec = max(np.random.normal(self.base_tau_rec, self.base_tau_rec * cv),
                          self.base_tau_rec * 0.5)
        self.tau_facil = max(np.random.normal(self.base_tau_facil, self.base_tau_facil * cv),
                            self.base_tau_facil * 0.5)
        self.U = np.clip(np.random.normal(self.base_U, self.base_U * cv), 0.05, 0.95)
    
    def stimulate(self, time, record_history=False):
        dt = time - self.last_time
        
        if dt > 0:
            self.x = 1 - (1 - self.x) * np.exp(-dt / self.tau_rec)
            self.u = self.U + (self.u - self.U) * np.exp(-dt / self.tau_facil)
        
        response = self.u * self.x
        
        if record_history:
            self.history['times'].append(time)
            self.history['x_values'].append(self.x)
            self.history['u_values'].append(self.u)
            self.history['responses'].append(response)
        
        self.x = self.x * (1 - self.u)
        self.u = self.u + self.U * (1 - self.u)
        self.last_time = time
        
        return response

def get_pd_parameters(frequency):
    """Get PD parameters with 20Hz threshold"""
    if frequency < 20.0:
        return 460.0, 20.0, 0.32
    else:
        return 184.0, 52.5, 0.2135

def analyze_resource_competition_single_trial(frequencies, n_pulses=5, cv=0.20):
    """Analyze resource competition for single trial"""
    dd_freq, md_freq, pd_freq = frequencies
    
    # Create synapses for individual responses
    synapses_individual = {
        'DD': TsodyksMarkramSynapse(248.0, 133.0, 0.20, "DD"),
        'MD': TsodyksMarkramSynapse(3977.0, 27.0, 0.30, "MD"),
        'PD': TsodyksMarkramSynapse(*get_pd_parameters(pd_freq), "PD")
    }
    
    for syn in synapses_individual.values():
        syn.set_trial_parameters(cv)
    
    # Individual responses
    individual_responses = {}
    for pathway, freq in zip(['DD', 'MD', 'PD'], frequencies):
        isi = 1000.0 / freq
        syn = synapses_individual[pathway]
        total = sum(syn.stimulate(i * isi, record_history=True) for i in range(n_pulses))
        individual_responses[pathway] = total
    
    # Create synapses for integrated response
    synapses_integrated = {
        'DD': TsodyksMarkramSynapse(248.0, 133.0, 0.20, "DD"),
        'MD': TsodyksMarkramSynapse(3977.0, 27.0, 0.30, "MD"),
        'PD': TsodyksMarkramSynapse(*get_pd_parameters(pd_freq), "PD")
    }
    
    for syn in synapses_integrated.values():
        syn.set_trial_parameters(cv)
    
    # Generate combined stimulus schedule
    all_stimuli = []
    for pathway, freq in zip(['DD', 'MD', 'PD'], frequencies):
        isi = 1000.0 / freq
        for pulse in range(n_pulses):
            all_stimuli.append((pulse * isi, pathway))
    all_stimuli.sort()
    
    # Integrated response
    integrated_response = sum(synapses_integrated[pw].stimulate(t, record_history=True) 
                             for t, pw in all_stimuli)
    
    # Calculate metrics
    individual_sum = sum(individual_responses.values())
    interaction_coefficient = ((integrated_response - individual_sum) / individual_sum 
                              if individual_sum > 0 else 0.0)
    
    # Resource utilization
    resource_utilization = {
        pathway: 1.0 - (syn.history['x_values'][-1] if syn.history['x_values'] else 1.0)
        for pathway, syn in synapses_integrated.items()
    }
    
    return {
        'individual_responses': individual_responses,
        'integrated_response': integrated_response,
        'interaction_coefficient': interaction_coefficient,
        'resource_utilization': resource_utilization
    }

def analyze_resource_competition_multiple_trials(frequencies, n_pulses=5, 
                                                n_trials=30, cv=0.20):
    """Analyze with multiple trials"""
    all_results = [analyze_resource_competition_single_trial(frequencies, n_pulses, cv) 
                  for _ in range(n_trials)]
    
    return {
        'frequencies': frequencies,
        'n_pulses': n_pulses,
        'individual_responses': {
            'DD': np.mean([r['individual_responses']['DD'] for r in all_results]),
            'MD': np.mean([r['individual_responses']['MD'] for r in all_results]),
            'PD': np.mean([r['individual_responses']['PD'] for r in all_results])
        },
        'integrated_response': np.mean([r['integrated_response'] for r in all_results]),
        'interaction_coefficient': np.mean([r['interaction_coefficient'] for r in all_results]),
        'interaction_sem': stats.sem([r['interaction_coefficient'] for r in all_results]),
        'resource_utilization': {
            'DD': np.mean([r['resource_utilization']['DD'] for r in all_results]),
            'MD': np.mean([r['resource_utilization']['MD'] for r in all_results]),
            'PD': np.mean([r['resource_utilization']['PD'] for r in all_results])
        }
    }

def run_mechanism_analysis(n_trials=30, cv=0.20):
    """Run comprehensive mechanism analysis"""
    print(f"Starting mechanism analysis ({n_trials} trials per condition)...")
    logging.info(f"Beginning mechanism analysis with {n_trials} trials, CV={cv}")
    
    test_conditions = [
        ([5, 5, 5], "Equal_5Hz"),
        ([10, 10, 10], "Equal_10Hz"),
        ([15, 15, 15], "Equal_15Hz"),
        ([20, 20, 20], "Equal_20Hz"),
        ([30, 30, 30], "Equal_30Hz"),
        ([5, 10, 15], "Mixed_Low"),
        ([15, 20, 30], "Mixed_High"),
        ([30, 5, 5], "DD_Dominant"),
        ([5, 30, 5], "MD_Dominant"),
        ([5, 5, 30], "PD_Dominant"),
        ([10, 10, 15], "PD_Below_Thresh"),
        ([10, 10, 25], "PD_Above_Thresh"),
    ]
    
    results = []
    
    for i, (frequencies, condition_name) in enumerate(test_conditions):
        progress = (i + 1) / len(test_conditions) * 100
        print(f"Testing {condition_name} ({progress:.0f}%)")
        
        for n_pulses in [5, 10, 15]:
            result = analyze_resource_competition_multiple_trials(
                frequencies, n_pulses, n_trials, cv
            )
            result['condition_name'] = condition_name
            result['mean_frequency'] = np.mean(frequencies)
            result['frequency_spread'] = max(frequencies) - min(frequencies)
            results.append(result)
    
    # Frequency sweep
    logging.info("Performing frequency sweep")
    base_freqs = [1, 5, 10, 15, 20, 30, 40]
    
    for dd_freq in [10, 20]:
        for md_freq in [10, 20]:
            for pd_freq in base_freqs:
                frequencies = [dd_freq, md_freq, pd_freq]
                result = analyze_resource_competition_multiple_trials(
                    frequencies, n_pulses=10, n_trials=n_trials, cv=cv
                )
                result['condition_name'] = f"Sweep_DD{dd_freq}_MD{md_freq}_PD{pd_freq}"
                result['mean_frequency'] = np.mean(frequencies)
                result['frequency_spread'] = max(frequencies) - min(frequencies)
                results.append(result)
    
    logging.info(f"Completed {len(results)} test conditions")
    return results

def create_mechanism_figure(results):
    """Create mechanism analysis figure with 6 panels"""
    print("Creating Figure 7...")
    logging.info("Generating Figure 7")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    n_trials = 30
    cv = 0.20
    
    fig.suptitle(f'Figure 7: Mechanism Analysis\n'
                f'Mean ± SEM (n={n_trials} trials, CV={cv})', 
                fontsize=16, fontweight='bold')
    
    interaction_coeffs = [r['interaction_coefficient'] * 100 for r in results]
    mean_frequencies = [r['mean_frequency'] for r in results]
    
    # Panel A: Pathway competition matrix
    ax1 = axes[0, 0]
    pathway_pairs = ['DD-MD', 'DD-PD', 'MD-PD']
    competition_scores = []
    
    for pair in pathway_pairs:
        pair_interactions = []
        for r in results:
            freqs = r['frequencies']
            if pair == 'DD-MD' and freqs[0] > 0 and freqs[1] > 0:
                pair_interactions.append(abs(r['interaction_coefficient']))
            elif pair == 'DD-PD' and freqs[0] > 0 and freqs[2] > 0:
                pair_interactions.append(abs(r['interaction_coefficient']))
            elif pair == 'MD-PD' and freqs[1] > 0 and freqs[2] > 0:
                pair_interactions.append(abs(r['interaction_coefficient']))
        competition_scores.append(np.mean(pair_interactions) * 100 if pair_interactions else 0)
    
    ax1.bar(pathway_pairs, competition_scores, 
           color=['#636363', '#969696', '#cccccc'], alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Mean Interaction Strength (%)')
    ax1.set_title('A. Pathway Competition Matrix', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Interaction vs frequency
    ax2 = axes[0, 1]
    ax2.scatter(mean_frequencies, interaction_coeffs, 
               facecolors='none', edgecolors='black', linewidths=1.5, alpha=0.8, s=60)
    
    if len(mean_frequencies) > 1:
        slope, intercept, r_value, _, _ = stats.linregress(mean_frequencies, interaction_coeffs)
        line_x = np.array([min(mean_frequencies), max(mean_frequencies)])
        line_y = slope * line_x + intercept
        ax2.plot(line_x, line_y, 'r--', linewidth=2, alpha=0.8, 
                label=f'Trend (r²={r_value**2:.3f})')
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.axvline(x=20, color='purple', linestyle=':', alpha=0.5, label='20Hz threshold')
    ax2.set_xlabel('Mean Frequency (Hz)')
    ax2.set_ylabel('Interaction Coefficient (%)')
    ax2.set_title('B. Frequency-Dependent Competition', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Resource utilization
    ax3 = axes[0, 2]
    pathway_resource_usage = {'DD': [], 'MD': [], 'PD': []}
    
    for r in results:
        for pathway in ['DD', 'MD', 'PD']:
            pathway_resource_usage[pathway].append(r['resource_utilization'][pathway])
    
    pathway_names = ['DD', 'MD', 'PD']
    mean_usage = [np.mean(pathway_resource_usage[p]) * 100 for p in pathway_names]
    std_usage = [np.std(pathway_resource_usage[p]) * 100 for p in pathway_names]
    
    ax3.bar(pathway_names, mean_usage, yerr=std_usage, capsize=5,
           color=['#636363', '#969696', '#cccccc'], alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Mean Resource Utilization (%)')
    ax3.set_title('C. Resource Depletion by Pathway', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Panel D: PD frequency domain effects
    ax4 = axes[1, 0]
    pd_low_interactions = []
    pd_high_interactions = []
    
    for r in results:
        pd_freq = r['frequencies'][2]
        interaction = r['interaction_coefficient'] * 100
        
        if pd_freq < 20:
            pd_low_interactions.append(interaction)
        else:
            pd_high_interactions.append(interaction)
    
    bp = ax4.boxplot([pd_low_interactions, pd_high_interactions], 
                     labels=['PD <20Hz\n(Theta-Beta)', 'PD ≥20Hz\n(Gamma)'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('#636363')
    bp['boxes'][1].set_facecolor('#636363')
    
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_ylabel('Interaction Coefficient (%)')
    ax4.set_title('D. PD Frequency Domain Effects', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    if pd_low_interactions and pd_high_interactions:
        from scipy.stats import ttest_ind
        t_stat, p_val = ttest_ind(pd_low_interactions, pd_high_interactions)
        ax4.text(0.5, 0.95, f'p = {p_val:.4f}', transform=ax4.transAxes, 
                ha='center', va='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel E: Mechanism decomposition
    ax5 = axes[1, 1]
    mechanism_categories = {
        'Resource\nCompetition': [],
        'Facilitation\nEffects': [],
        'Temporal\nOverlap': []
    }
    
    for r in results:
        interaction = r['interaction_coefficient'] * 100
        mean_freq = r['mean_frequency']
        freq_spread = r['frequency_spread']
        
        if abs(interaction) > 0.5:
            if mean_freq > 20:
                mechanism_categories['Resource\nCompetition'].append(interaction)
            elif freq_spread > 10:
                mechanism_categories['Temporal\nOverlap'].append(interaction)
            else:
                mechanism_categories['Facilitation\nEffects'].append(interaction)
    
    valid_categories = {k: v for k, v in mechanism_categories.items() if v}
    
    if valid_categories:
        bp2 = ax5.boxplot(valid_categories.values(), labels=list(valid_categories.keys()), 
                         patch_artist=True)
        colors = ['#636363', '#636363', '#636363']
        for i, patch in enumerate(bp2['boxes']):
            patch.set_facecolor(colors[i % len(colors)])
    
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax5.set_ylabel('Interaction Coefficient (%)')
    ax5.set_title('E. Mechanism Classification', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Panel F: Integration vs linear sum
    ax6 = axes[1, 2]
    individual_sums = [sum(r['individual_responses'].values()) for r in results]
    integrated_responses = [r['integrated_response'] for r in results]
    
    ax6.scatter(individual_sums, integrated_responses, 
               facecolors='none', edgecolors='black', linewidths=1.5, alpha=0.8, s=60)
    
    min_val = min(min(individual_sums), min(integrated_responses))
    max_val = max(max(individual_sums), max(integrated_responses))
    ax6.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
            label='Perfect Linear')
    
    if len(individual_sums) > 1:
        slope, intercept, r_value, _, _ = stats.linregress(individual_sums, integrated_responses)
        line_x = np.array([min_val, max_val])
        line_y = slope * line_x + intercept
        ax6.plot(line_x, line_y, 'g-', linewidth=2, alpha=0.8,
                label=f'Observed (r²={r_value**2:.4f})')
    
    ax6.set_xlabel('Individual Pathway Sum')
    ax6.set_ylabel('Integrated Response')
    ax6.set_title('F. Integration vs Linear Sum', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_filename = f'Figure7_Mechanism_variability_{timestamp}.png'
    tiff_filename = f'Figure7_Mechanism_variability_{timestamp}.tiff'
    csv_filename = f'Figure7_data_variability_{timestamp}.csv'
    
    fig.savefig(png_filename, dpi=600, bbox_inches='tight')
    img = Image.open(png_filename)
    img.save(tiff_filename, dpi=(600, 600), compression='tiff_lzw')
    plt.close(fig)
    
    df = pd.DataFrame(results)
    df.to_csv(csv_filename, index=False)
    
    logging.info(f"Saved: {png_filename}, {tiff_filename}, {csv_filename}")
    
    return png_filename, tiff_filename, csv_filename

def main():
    print("="*70)
    print("FIGURE 7: MECHANISM ANALYSIS (30 trials, CV=0.20)")
    print("="*70)
    
    log_file = setup_logging()
    
    try:
        results = run_mechanism_analysis(n_trials=30, cv=0.20)
        png_file, tiff_file, csv_file = create_mechanism_figure(results)
        
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
