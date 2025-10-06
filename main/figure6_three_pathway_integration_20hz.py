"""
Figure 6 - Three-Pathway Integration Analysis with 20Hz Threshold
==================================================================
NO FABRICATION - Uses experimental parameters with 20Hz threshold
Based on Figure 4 experimental evidence showing facilitation onset at 20Hz
Comprehensive analysis of three-pathway synaptic integration

Filename: figure6_three_pathway_integration_20hz.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
from datetime import datetime
from scipy import stats
from PIL import Image
import itertools

def setup_logging():
    """Setup comprehensive debug logging"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"figure6_integration_20hz_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*70)
    logging.info("THREE-PATHWAY INTEGRATION ANALYSIS (20Hz THRESHOLD)")
    logging.info("NO FABRICATION - Experimental parameters only")
    logging.info("Threshold at 20Hz based on Figure 4 experimental data")
    logging.info("="*70)
    
    return log_filename

# Publication quality parameters
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.2,
})

class TsodyksMarkramSynapse:
    """Tsodyks-Markram synapse model for integration analysis"""
    
    def __init__(self, tau_rec, tau_facil, U, pathway_name="synapse"):
        self.tau_rec = float(tau_rec)
        self.tau_facil = float(tau_facil)
        self.U = float(U)
        self.pathway_name = pathway_name
        self.reset()
        
        logging.debug(f"Created {pathway_name} synapse: "
                     f"tau_rec={tau_rec}, tau_facil={tau_facil}, U={U}")
    
    def reset(self):
        """Reset synapse to initial state"""
        self.x = 1.0  # Available resources
        self.u = self.U  # Release probability
        self.last_time = 0.0
    
    def stimulate(self, time):
        """Apply stimulation with time-dependent dynamics"""
        dt = time - self.last_time
        
        if dt > 0:
            # Resource recovery
            self.x = 1 - (1 - self.x) * np.exp(-dt / self.tau_rec)
            # Facilitation decay
            self.u = self.U + (self.u - self.U) * np.exp(-dt / self.tau_facil)
        
        # Calculate response
        response = self.u * self.x
        
        # Update state
        self.x = self.x * (1 - self.u)
        self.u = self.u + self.U * (1 - self.u)
        
        self.last_time = time
        return response

def get_pd_parameters(frequency):
    """
    Get PD parameters with 20Hz threshold
    <20Hz: Low-medium frequency (theta-beta)
    >=20Hz: High frequency (gamma)
    """
    if frequency < 20.0:  # Changed from <= 10.0
        tau_rec, tau_facil, U = 460.0, 20.0, 0.32
        logging.debug(f"PD freq {frequency}Hz -> Low-medium params")
    else:
        tau_rec, tau_facil, U = 184.0, 52.5, 0.2135
        logging.debug(f"PD freq {frequency}Hz -> High freq params")
    
    return tau_rec, tau_facil, U

def create_pathway_synapses(frequencies):
    """Create synapses with appropriate parameters"""
    pd_freq = np.mean(frequencies) if frequencies else 15.0
    pd_tau_rec, pd_tau_facil, pd_U = get_pd_parameters(pd_freq)
    
    synapses = {
        'DD': TsodyksMarkramSynapse(248.0, 133.0, 0.20, "DD_LPP"),
        'MD': TsodyksMarkramSynapse(3977.0, 27.0, 0.30, "MD_MPP"),
        'PD': TsodyksMarkramSynapse(pd_tau_rec, pd_tau_facil, pd_U, f"PD_{pd_freq:.0f}Hz")
    }
    
    logging.debug(f"Created synapses for mean freq={pd_freq:.1f}Hz")
    
    return synapses

def simulate_individual_pathway(pathway, frequency, num_pulses=5):
    """Simulate individual pathway response"""
    if pathway == 'PD':
        tau_rec, tau_facil, U = get_pd_parameters(frequency)
        synapse = TsodyksMarkramSynapse(tau_rec, tau_facil, U, f"PD_{frequency}Hz")
    elif pathway == 'DD':
        synapse = TsodyksMarkramSynapse(248.0, 133.0, 0.20, "DD")
    elif pathway == 'MD':
        synapse = TsodyksMarkramSynapse(3977.0, 27.0, 0.30, "MD")
    
    # Generate stimulus times
    isi = 1000.0 / frequency
    stimulus_times = [i * isi for i in range(num_pulses)]
    
    # Simulate response
    responses = []
    for time in stimulus_times:
        response = synapse.stimulate(time)
        responses.append(response)
    
    total_response = sum(responses)
    return total_response, responses

def simulate_integrated_response(dd_freq, md_freq, pd_freq, num_pulses=5):
    """Simulate three-pathway integrated response"""
    
    # Create synapses
    synapses = create_pathway_synapses([dd_freq, md_freq, pd_freq])
    
    # Generate combined stimulus schedule
    all_stimuli = []
    frequencies = {'DD': dd_freq, 'MD': md_freq, 'PD': pd_freq}
    
    for pathway, freq in frequencies.items():
        isi = 1000.0 / freq
        for pulse in range(num_pulses):
            stimulus_time = pulse * isi
            all_stimuli.append((stimulus_time, pathway))
    
    # Sort by time
    all_stimuli.sort(key=lambda x: x[0])
    
    # Simulate integrated response
    total_integrated_response = 0
    
    for time, pathway in all_stimuli:
        synapse = synapses[pathway]
        response = synapse.stimulate(time)
        total_integrated_response += response
    
    return total_integrated_response

def run_comprehensive_integration_analysis():
    """Run comprehensive integration analysis with 20Hz threshold"""
    print("Starting comprehensive three-pathway integration analysis...")
    logging.info("Starting integration analysis with 20Hz threshold")
    
    # Test frequencies including 15Hz to show 20Hz threshold effect
    frequencies = [1, 5, 10, 15, 20, 30, 40]  # Added 15Hz
    num_pulses_options = [3, 5, 7]
    
    results = []
    total_tests = len(frequencies) ** 3 * len(num_pulses_options)
    test_count = 0
    
    logging.info(f"Total test conditions: {total_tests}")
    logging.info(f"Test frequencies: {frequencies}")
    
    for num_pulses in num_pulses_options:
        for dd_freq in frequencies:
            for md_freq in frequencies:
                for pd_freq in frequencies:
                    test_count += 1
                    
                    # Progress reporting
                    if test_count % 50 == 0 or test_count <= 10:
                        progress = (test_count / total_tests) * 100
                        print(f"Progress: {progress:.1f}% ({test_count}/{total_tests})")
                        logging.debug(f"Testing: DD={dd_freq}, MD={md_freq}, PD={pd_freq}, pulses={num_pulses}")
                    
                    # Calculate individual responses
                    dd_response, _ = simulate_individual_pathway('DD', dd_freq, num_pulses)
                    md_response, _ = simulate_individual_pathway('MD', md_freq, num_pulses)
                    pd_response, _ = simulate_individual_pathway('PD', pd_freq, num_pulses)
                    
                    individual_sum = dd_response + md_response + pd_response
                    
                    # Calculate integrated response
                    integrated_response = simulate_integrated_response(dd_freq, md_freq, pd_freq, num_pulses)
                    
                    # Calculate interaction coefficient
                    if individual_sum > 0:
                        interaction_coefficient = (integrated_response - individual_sum) / individual_sum
                    else:
                        interaction_coefficient = 0.0
                    
                    # Store results
                    result = {
                        'DD_freq': dd_freq,
                        'MD_freq': md_freq,
                        'PD_freq': pd_freq,
                        'num_pulses': num_pulses,
                        'DD_response': dd_response,
                        'MD_response': md_response,
                        'PD_response': pd_response,
                        'individual_sum': individual_sum,
                        'integrated_response': integrated_response,
                        'interaction_coefficient': interaction_coefficient,
                        'mean_frequency': np.mean([dd_freq, md_freq, pd_freq]),
                        'frequency_range': max([dd_freq, md_freq, pd_freq]) - min([dd_freq, md_freq, pd_freq])
                    }
                    
                    results.append(result)
                    
                    # Log significant deviations
                    if abs(interaction_coefficient) > 0.1:
                        logging.info(f"Large interaction: {interaction_coefficient:.3f} at "
                                   f"DD={dd_freq}, MD={md_freq}, PD={pd_freq}")
    
    logging.info(f"Completed {len(results)} integration tests")
    return results

def create_integration_figure(results):
    """Create comprehensive integration figure with 20Hz threshold visualization"""
    print("Creating Figure 6: Three-Pathway Integration (20Hz threshold)...")
    logging.info("Generating Figure 6 with 20Hz threshold")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Figure 6: Three-Pathway Integration Analysis (20Hz PD Threshold)', 
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
    
    logging.info(f"Panel A: Mean interaction = {mean_val:.3f}%")
    
    # Panel B: Linear summation test
    ax2 = axes[0, 1]
    
    ax2.scatter(df['individual_sum'], df['integrated_response'], alpha=0.6, s=20)
    
    min_val = min(df['individual_sum'].min(), df['integrated_response'].min())
    max_val = max(df['individual_sum'].max(), df['integrated_response'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Linear')
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['individual_sum'], df['integrated_response'])
    line_x = np.array([min_val, max_val])
    line_y = slope * line_x + intercept
    ax2.plot(line_x, line_y, 'g-', linewidth=2, alpha=0.8, label=f'Observed (r²={r_value**2:.4f})')
    
    ax2.set_xlabel('Individual Pathway Sum')
    ax2.set_ylabel('Integrated Response')
    ax2.set_title('B. Linear Summation Test', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    logging.info(f"Panel B: r² = {r_value**2:.4f}, slope = {slope:.4f}")
    
    # Panel C: Frequency dependence
    ax3 = axes[0, 2]
    
    freq_groups = df.groupby('mean_frequency')['interaction_coefficient'].agg(['mean', 'std']).reset_index()
    freq_groups['mean_pct'] = freq_groups['mean'] * 100
    freq_groups['std_pct'] = freq_groups['std'] * 100
    
    ax3.errorbar(freq_groups['mean_frequency'], freq_groups['mean_pct'], 
                yerr=freq_groups['std_pct'], marker='o', capsize=5, capthick=2)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    ax3.set_xlabel('Mean Frequency (Hz)')
    ax3.set_ylabel('Mean Interaction Coefficient (%)')
    ax3.set_title('C. Frequency-Dependent Integration', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-30, 30)
    
    # Panel D: Pulse number dependence
    ax4 = axes[1, 0]
    
    pulse_groups = df.groupby('num_pulses')['interaction_coefficient'].agg(['mean', 'std']).reset_index()
    pulse_groups['mean_pct'] = pulse_groups['mean'] * 100
    pulse_groups['std_pct'] = pulse_groups['std'] * 100
    
    ax4.bar(pulse_groups['num_pulses'], pulse_groups['mean_pct'], 
           yerr=pulse_groups['std_pct'], capsize=5, alpha=0.7, 
           color='lightcoral', edgecolor='black')
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    ax4.set_xlabel('Number of Pulses')
    ax4.set_ylabel('Mean Interaction Coefficient (%)')
    ax4.set_title('D. Pulse Number Dependence', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-30, 30)
    
    # Panel E: PD frequency effect with 20Hz threshold
    ax5 = axes[1, 1]
    
    pd_freq_groups = df.groupby('PD_freq')['interaction_coefficient'].agg(['mean', 'std']).reset_index()
    pd_freq_groups['mean_pct'] = pd_freq_groups['mean'] * 100
    pd_freq_groups['std_pct'] = pd_freq_groups['std'] * 100
    
    # Color-code by 20Hz threshold
    colors = ['red' if freq < 20 else 'blue' for freq in pd_freq_groups['PD_freq']]
    
    ax5.scatter(pd_freq_groups['PD_freq'], pd_freq_groups['mean_pct'], 
               c=colors, s=100, alpha=0.7, edgecolors='black')
    ax5.errorbar(pd_freq_groups['PD_freq'], pd_freq_groups['mean_pct'], 
                yerr=pd_freq_groups['std_pct'], fmt='none', capsize=3, alpha=0.7)
    
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax5.axvline(x=20, color='purple', linestyle=':', linewidth=2, alpha=0.7, 
                label='Parameter Switch (20Hz)')
    
    ax5.set_xlabel('PD Pathway Frequency (Hz)')
    ax5.set_ylabel('Mean Interaction Coefficient (%)')
    ax5.set_title('E. PD Frequency-Dependent Integration (20Hz Threshold)', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(-30, 30)
    
    # Add text labels for domains
    ax5.text(10, -25, '<20Hz', color='darkred', fontsize=10, ha='center')
    ax5.text(30, -25, '≥20Hz', color='darkblue', fontsize=10, ha='center')
    
    # Panel F: Summary statistics with 20Hz threshold
    ax6 = axes[1, 2]
    
    stats_data = {
        'All\nConditions': interaction_coeffs,
        'Low PD\n(<20Hz)': df[df['PD_freq'] < 20]['interaction_coefficient'] * 100,
        'High PD\n(≥20Hz)': df[df['PD_freq'] >= 20]['interaction_coefficient'] * 100,
    }
    
    box_data = [data.values for data in stats_data.values() if len(data) > 0]
    box_labels = [label for label, data in stats_data.items() if len(data) > 0]
    
    bp = ax6.boxplot(box_data, labels=box_labels, patch_artist=True)
    
    colors_box = ['lightblue', 'lightcoral', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors_box[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax6.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax6.set_ylabel('Interaction Coefficient (%)')
    ax6.set_title('F. Integration Summary (20Hz Threshold)', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(-40, 40)
    
    plt.tight_layout()
    
    # Save files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    png_filename = f'Figure6_Three_Pathway_Integration_20Hz_{timestamp}.png'
    tiff_filename = f'Figure6_Three_Pathway_Integration_20Hz_{timestamp}.tiff'
    csv_filename = f'figure6_integration_data_20hz_{timestamp}.csv'
    
    fig.savefig(png_filename, dpi=300, bbox_inches='tight')
    logging.info(f"Saved PNG: {png_filename}")
    
    img = Image.open(png_filename)
    img.save(tiff_filename, dpi=(600, 600), compression='tiff_lzw')
    logging.info(f"Saved TIFF: {tiff_filename}")
    
    df.to_csv(csv_filename, index=False)
    logging.info(f"Saved CSV: {csv_filename}")
    
    print(f"Saved: {png_filename}, {tiff_filename}, {csv_filename}")
    
    plt.show()
    
    return df

def generate_integration_summary(results_df):
    """Generate summary with 20Hz threshold analysis"""
    print("\n" + "="*70)
    print("THREE-PATHWAY INTEGRATION ANALYSIS SUMMARY (20Hz THRESHOLD)")
    print("="*70)
    
    interaction_coeffs = results_df['interaction_coefficient'] * 100
    
    print(f"Total test conditions: {len(results_df)}")
    print(f"Mean interaction coefficient: {np.mean(interaction_coeffs):.3f}% ± {np.std(interaction_coeffs):.3f}%")
    print(f"Median: {np.median(interaction_coeffs):.3f}%")
    print(f"Range: {np.min(interaction_coeffs):.3f}% to {np.max(interaction_coeffs):.3f}%")
    
    # 20Hz threshold analysis
    low_freq_pd = results_df[results_df['PD_freq'] < 20]['interaction_coefficient'] * 100
    high_freq_pd = results_df[results_df['PD_freq'] >= 20]['interaction_coefficient'] * 100
    
    print(f"\nPD Frequency Domain Analysis (20Hz threshold):")
    print(f"  Low-medium freq PD (<20Hz): {np.mean(low_freq_pd):.3f}% ± {np.std(low_freq_pd):.3f}%")
    print(f"  High freq PD (≥20Hz): {np.mean(high_freq_pd):.3f}% ± {np.std(high_freq_pd):.3f}%")
    
    # Statistical test
    from scipy.stats import ttest_ind
    if len(low_freq_pd) > 0 and len(high_freq_pd) > 0:
        t_stat, p_value = ttest_ind(low_freq_pd, high_freq_pd)
        print(f"  Statistical difference: t={t_stat:.3f}, p={p_value:.4f}")
    
    # Linear summation test
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        results_df['individual_sum'], results_df['integrated_response'])
    print(f"\nLinear summation analysis:")
    print(f"  Correlation: r = {r_value:.4f}")
    print(f"  r² = {r_value**2:.4f}")
    print(f"  Slope: {slope:.4f} (perfect = 1.0)")
    print(f"  p-value: {p_value:.2e}")
    
    # Conclusion
    if abs(np.mean(interaction_coeffs)) < 1.0:
        print(f"\nCONCLUSION: Minimal deviation from linear summation")
        print(f"({abs(np.mean(interaction_coeffs)):.1f}% mean effect)")
    elif np.mean(interaction_coeffs) < -5.0:
        print(f"\nCONCLUSION: Significant sublinear summation")
        print(f"({np.mean(interaction_coeffs):.1f}% suppression)")
    else:
        print(f"\nCONCLUSION: Near-linear summation with minimal interaction")
    
    print(f"\n20Hz threshold successfully implemented")
    print(f"Based on Figure 4 experimental evidence")
    
    logging.info("Summary completed")

def main():
    """Main execution with 20Hz threshold"""
    print("Figure 6: Three-Pathway Integration Analysis (20Hz Threshold)")
    print("=" * 60)
    print("NO FABRICATION - Experimental parameters with 20Hz threshold")
    print("Based on Figure 4 showing facilitation onset at 20Hz")
    print("=" * 60)
    
    log_file = setup_logging()
    
    try:
        # Run analysis
        results = run_comprehensive_integration_analysis()
        
        # Create figure
        results_df = create_integration_figure(results)
        
        # Generate summary
        generate_integration_summary(results_df)
        
        print(f"\nDebug log: {log_file}")
        print("Figure 6 completed successfully!")
        
        logging.info("Figure 6 generation completed")
        
        return results_df
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        logging.error(error_msg)
        raise

if __name__ == "__main__":
    main()
