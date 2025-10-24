"""
Figure 6 - Three-Pathway Integration Analysis with Biological Variability
==========================================================================
30 trials per condition with CV = 0.20
Comprehensive analysis of three-pathway synaptic integration

BIOLOGICAL VARIABILITY REFERENCES:
- Nusser et al., 2001: mIPSC variability ~30%
- Smith et al., 2003: mEPSC amplitude CV = 0.69-0.75
- Implemented as CV = 0.20 (20%, conservative estimate)

Filename: figure6_with_variability_30trials.py
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
    log_filename = f"figure6_variability_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*70)
    logging.info("FIGURE 6: THREE-PATHWAY INTEGRATION WITH VARIABILITY")
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
    """TM synapse with biological variability"""
    
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
    
    def set_trial_parameters(self, cv=0.20):
        """Apply biological variability (Nusser et al., 2001)"""
        self.tau_rec = max(np.random.normal(self.base_tau_rec, self.base_tau_rec * cv),
                          self.base_tau_rec * 0.5)
        self.tau_facil = max(np.random.normal(self.base_tau_facil, self.base_tau_facil * cv),
                            self.base_tau_facil * 0.5)
        self.U = np.clip(np.random.normal(self.base_U, self.base_U * cv), 0.05, 0.95)
    
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

def get_pd_parameters(frequency):
    """Get PD parameters with 20Hz threshold"""
    if frequency < 20.0:
        return 460.0, 20.0, 0.32  # Low-medium freq
    else:
        return 184.0, 52.5, 0.2135  # High freq

def simulate_individual_pathway_single_trial(pathway, frequency, num_pulses=5, cv=0.20):
    """Simulate individual pathway for single trial"""
    if pathway == 'PD':
        tau_rec, tau_facil, U = get_pd_parameters(frequency)
        synapse = TsodyksMarkramSynapse(tau_rec, tau_facil, U, f"PD_{frequency}Hz")
    elif pathway == 'DD':
        synapse = TsodyksMarkramSynapse(248.0, 133.0, 0.20, "DD")
    elif pathway == 'MD':
        synapse = TsodyksMarkramSynapse(3977.0, 27.0, 0.30, "MD")
    
    synapse.set_trial_parameters(cv)
    
    isi = 1000.0 / frequency
    responses = [synapse.stimulate(i * isi) for i in range(num_pulses)]
    
    return sum(responses)

def simulate_integrated_response_single_trial(dd_freq, md_freq, pd_freq, num_pulses=5, cv=0.20):
    """Simulate three-pathway integrated response for single trial"""
    
    # Create synapses
    pd_tau_rec, pd_tau_facil, pd_U = get_pd_parameters(pd_freq)
    synapses = {
        'DD': TsodyksMarkramSynapse(248.0, 133.0, 0.20, "DD"),
        'MD': TsodyksMarkramSynapse(3977.0, 27.0, 0.30, "MD"),
        'PD': TsodyksMarkramSynapse(pd_tau_rec, pd_tau_facil, pd_U, "PD")
    }
    
    for syn in synapses.values():
        syn.set_trial_parameters(cv)
    
    # Generate combined stimulus schedule
    all_stimuli = []
    frequencies = {'DD': dd_freq, 'MD': md_freq, 'PD': pd_freq}
    
    for pathway, freq in frequencies.items():
        isi = 1000.0 / freq
        for pulse in range(num_pulses):
            all_stimuli.append((pulse * isi, pathway))
    
    all_stimuli.sort(key=lambda x: x[0])
    
    # Simulate integrated response
    total_response = sum(synapses[pathway].stimulate(time) 
                        for time, pathway in all_stimuli)
    
    return total_response

def run_integration_test_multiple_trials(dd_freq, md_freq, pd_freq, num_pulses=5, 
                                        n_trials=30, cv=0.20):
    """Run integration test with multiple trials"""
    
    all_dd_responses = []
    all_md_responses = []
    all_pd_responses = []
    all_individual_sums = []
    all_integrated_responses = []
    all_interaction_coeffs = []
    
    for trial in range(n_trials):
        # Individual responses
        dd_resp = simulate_individual_pathway_single_trial('DD', dd_freq, num_pulses, cv)
        md_resp = simulate_individual_pathway_single_trial('MD', md_freq, num_pulses, cv)
        pd_resp = simulate_individual_pathway_single_trial('PD', pd_freq, num_pulses, cv)
        
        individual_sum = dd_resp + md_resp + pd_resp
        
        # Integrated response
        integrated_resp = simulate_integrated_response_single_trial(
            dd_freq, md_freq, pd_freq, num_pulses, cv
        )
        
        # Interaction coefficient
        if individual_sum > 0:
            interaction = (integrated_resp - individual_sum) / individual_sum
        else:
            interaction = 0.0
        
        all_dd_responses.append(dd_resp)
        all_md_responses.append(md_resp)
        all_pd_responses.append(pd_resp)
        all_individual_sums.append(individual_sum)
        all_integrated_responses.append(integrated_resp)
        all_interaction_coeffs.append(interaction)
    
    return {
        'dd_mean': np.mean(all_dd_responses),
        'md_mean': np.mean(all_md_responses),
        'pd_mean': np.mean(all_pd_responses),
        'individual_sum_mean': np.mean(all_individual_sums),
        'integrated_mean': np.mean(all_integrated_responses),
        'interaction_mean': np.mean(all_interaction_coeffs),
        'interaction_sem': stats.sem(all_interaction_coeffs),
        'interaction_std': np.std(all_interaction_coeffs)
    }

def run_comprehensive_integration_analysis(n_trials=30, cv=0.20):
    """Run comprehensive integration analysis with variability"""
    print(f"Starting integration analysis ({n_trials} trials per condition)...")
    logging.info(f"Starting integration analysis with {n_trials} trials, CV={cv}")
    
    frequencies = [1, 5, 10, 15, 20, 30, 40]
    num_pulses_options = [3, 5, 7]
    
    results = []
    total_tests = len(frequencies) ** 3 * len(num_pulses_options)
    test_count = 0
    
    logging.info(f"Total test conditions: {total_tests}")
    
    for num_pulses in num_pulses_options:
        for dd_freq in frequencies:
            for md_freq in frequencies:
                for pd_freq in frequencies:
                    test_count += 1
                    
                    if test_count % 50 == 0:
                        progress = (test_count / total_tests) * 100
                        print(f"Progress: {progress:.1f}% ({test_count}/{total_tests})")
                    
                    # Run multiple trials for this condition
                    trial_results = run_integration_test_multiple_trials(
                        dd_freq, md_freq, pd_freq, num_pulses, n_trials, cv
                    )
                    
                    result = {
                        'DD_freq': dd_freq,
                        'MD_freq': md_freq,
                        'PD_freq': pd_freq,
                        'num_pulses': num_pulses,
                        'DD_response': trial_results['dd_mean'],
                        'MD_response': trial_results['md_mean'],
                        'PD_response': trial_results['pd_mean'],
                        'individual_sum': trial_results['individual_sum_mean'],
                        'integrated_response': trial_results['integrated_mean'],
                        'interaction_coefficient': trial_results['interaction_mean'],
                        'interaction_sem': trial_results['interaction_sem'],
                        'interaction_std': trial_results['interaction_std'],
                        'mean_frequency': np.mean([dd_freq, md_freq, pd_freq]),
                        'n_trials': n_trials,
                        'cv': cv
                    }
                    
                    results.append(result)
    
    logging.info(f"Completed {len(results)} integration tests")
    return results

def create_integration_figure(results):
    """Create comprehensive integration figure with variability"""
    print("Creating Figure 6...")
    logging.info("Generating Figure 6")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    n_trials = results[0]['n_trials']
    cv = results[0]['cv']
    
    fig.suptitle(f'Figure 6: Three-Pathway Integration Analysis\n'
                f'Mean ± SEM (n={n_trials} trials, CV={cv})', 
                fontsize=16, fontweight='bold')
    
    df = pd.DataFrame(results)
    
    # Panel A: Interaction coefficient distribution
    ax1 = axes[0, 0]
    interaction_coeffs = df['interaction_coefficient'] * 100
    
    ax1.hist(interaction_coeffs, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Linear Summation')
    mean_val = np.mean(interaction_coeffs)
    ax1.axvline(x=mean_val, color='green', linestyle='-', linewidth=2, 
                label=f'Mean: {mean_val:.2f}%')
    
    ax1.set_xlabel('Interaction Coefficient (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('A. Integration Effect Distribution', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-50, 50)
    
    # Panel B: Linear summation test
    ax2 = axes[0, 1]
    
    ax2.scatter(df['individual_sum'], df['integrated_response'], alpha=0.6, s=20)
    
    min_val = min(df['individual_sum'].min(), df['integrated_response'].min())
    max_val = max(df['individual_sum'].max(), df['integrated_response'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Linear')
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df['individual_sum'], df['integrated_response']
    )
    line_x = np.array([min_val, max_val])
    line_y = slope * line_x + intercept
    ax2.plot(line_x, line_y, 'g-', linewidth=2, alpha=0.8, 
            label=f'Observed (r²={r_value**2:.4f})')
    
    ax2.set_xlabel('Individual Pathway Sum')
    ax2.set_ylabel('Integrated Response')
    ax2.set_title('B. Linear Summation Test', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Frequency dependence
    ax3 = axes[0, 2]
    
    freq_groups = df.groupby('mean_frequency').agg({
        'interaction_coefficient': ['mean', 'sem']
    }).reset_index()
    freq_groups.columns = ['mean_frequency', 'mean_coeff', 'sem_coeff']
    freq_groups['mean_pct'] = freq_groups['mean_coeff'] * 100
    freq_groups['sem_pct'] = freq_groups['sem_coeff'] * 100
    
    ax3.errorbar(freq_groups['mean_frequency'], freq_groups['mean_pct'], 
                yerr=freq_groups['sem_pct'], marker='o', capsize=5, capthick=2)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    ax3.set_xlabel('Mean Frequency (Hz)')
    ax3.set_ylabel('Mean Interaction Coefficient (%)')
    ax3.set_title('C. Frequency-Dependent Integration', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-30, 30)
    
    # Panel D: Pulse number dependence
    ax4 = axes[1, 0]
    
    pulse_groups = df.groupby('num_pulses').agg({
        'interaction_coefficient': ['mean', 'sem']
    }).reset_index()
    pulse_groups.columns = ['num_pulses', 'mean_coeff', 'sem_coeff']
    pulse_groups['mean_pct'] = pulse_groups['mean_coeff'] * 100
    pulse_groups['sem_pct'] = pulse_groups['sem_coeff'] * 100
    
    ax4.bar(pulse_groups['num_pulses'], pulse_groups['mean_pct'], 
           yerr=pulse_groups['sem_pct'], capsize=5, alpha=0.7, 
           color='lightcoral', edgecolor='black')
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    ax4.set_xlabel('Number of Pulses')
    ax4.set_ylabel('Mean Interaction Coefficient (%)')
    ax4.set_title('D. Pulse Number Dependence', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-30, 30)
    
    # Panel E: PD frequency effect with 20Hz threshold
    ax5 = axes[1, 1]
    
    pd_freq_groups = df.groupby('PD_freq').agg({
        'interaction_coefficient': ['mean', 'sem']
    }).reset_index()
    pd_freq_groups.columns = ['PD_freq', 'mean_coeff', 'sem_coeff']
    pd_freq_groups['mean_pct'] = pd_freq_groups['mean_coeff'] * 100
    pd_freq_groups['sem_pct'] = pd_freq_groups['sem_coeff'] * 100
    
    colors = ['red' if freq < 20 else 'blue' for freq in pd_freq_groups['PD_freq']]
    
    ax5.scatter(pd_freq_groups['PD_freq'], pd_freq_groups['mean_pct'], 
               c=colors, s=100, alpha=0.7, edgecolors='black')
    ax5.errorbar(pd_freq_groups['PD_freq'], pd_freq_groups['mean_pct'], 
                yerr=pd_freq_groups['sem_pct'], fmt='none', capsize=3, alpha=0.7)
    
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax5.axvline(x=20, color='purple', linestyle=':', linewidth=2, alpha=0.7, 
                label='Parameter Switch (20Hz)')
    
    ax5.set_xlabel('PD Pathway Frequency (Hz)')
    ax5.set_ylabel('Mean Interaction Coefficient (%)')
    ax5.set_title('E. PD Frequency-Dependent Integration', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(-30, 30)
    
    ax5.text(10, -25, '<20Hz', color='darkred', fontsize=10, ha='center')
    ax5.text(30, -25, '≥20Hz', color='darkblue', fontsize=10, ha='center')
    
    # Panel F: Summary statistics with 20Hz threshold
    ax6 = axes[1, 2]
    
    all_data = df['interaction_coefficient'] * 100
    low_pd_data = df[df['PD_freq'] < 20]['interaction_coefficient'] * 100
    high_pd_data = df[df['PD_freq'] >= 20]['interaction_coefficient'] * 100
    
    box_data = [all_data.values, low_pd_data.values, high_pd_data.values]
    box_labels = ['All\nConditions', 'Low PD\n(<20Hz)', 'High PD\n(≥20Hz)']
    
    bp = ax6.boxplot(box_data, labels=box_labels, patch_artist=True)
    
    colors_box = ['lightblue', 'lightcoral', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax6.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax6.set_ylabel('Interaction Coefficient (%)')
    ax6.set_title('F. Integration Summary', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(-40, 40)
    
    plt.tight_layout()
    
    # Save files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    png_filename = f'Figure6_Integration_variability_{timestamp}.png'
    tiff_filename = f'Figure6_Integration_variability_{timestamp}.tiff'
    csv_filename = f'Figure6_data_variability_{timestamp}.csv'
    
    fig.savefig(png_filename, dpi=600, bbox_inches='tight')
    logging.info(f"Saved PNG: {png_filename}")
    
    img = Image.open(png_filename)
    img.save(tiff_filename, dpi=(600, 600), compression='tiff_lzw')
    logging.info(f"Saved TIFF: {tiff_filename}")
    
    plt.close(fig)
    
    df.to_csv(csv_filename, index=False)
    logging.info(f"Saved CSV: {csv_filename}")
    
    return png_filename, tiff_filename, csv_filename

def main():
    print("="*70)
    print("FIGURE 6: THREE-PATHWAY INTEGRATION (30 trials, CV=0.20)")
    print("="*70)
    
    log_file = setup_logging()
    
    n_trials = 30
    cv = 0.20
    
    try:
        # Run comprehensive analysis
        results = run_comprehensive_integration_analysis(n_trials=n_trials, cv=cv)
        
        # Create figure
        png_file, tiff_file, csv_file = create_integration_figure(results)
        
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
