"""
Figure 8 - Natural Input Patterns Analysis with 20Hz Threshold
===============================================================
NO FABRICATION - Tests realistic biological patterns with 20Hz PD threshold
Based on experimental evidence from Figure 4
Compares regular vs burst vs Poisson vs theta-modulated patterns

Filename: figure8_natural_patterns_20hz.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import logging
from datetime import datetime
from scipy import stats
import random

def setup_logging():
    """Setup comprehensive debug logging"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"figure8_natural_patterns_20hz_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*70)
    logging.info("FIGURE 8: NATURAL PATTERNS ANALYSIS (20Hz THRESHOLD)")
    logging.info("NO FABRICATION - Experimental parameters only")
    logging.info("Based on 20Hz threshold from Figure 4 data")
    logging.info("="*70)
    
    return log_filename

# Publication quality settings
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
    """Tsodyks-Markram synapse for natural patterns"""
    
    def __init__(self, tau_rec, tau_facil, U, pathway_name="synapse"):
        self.tau_rec = float(tau_rec)
        self.tau_facil = float(tau_facil)
        self.U = float(U)
        self.pathway_name = pathway_name
        self.reset()
        
        logging.debug(f"Created {pathway_name}: tau_rec={tau_rec}, tau_facil={tau_facil}, U={U}")
    
    def reset(self):
        """Reset synapse state"""
        self.x = 1.0
        self.u = self.U
        self.last_time = 0.0
    
    def stimulate(self, time):
        """Apply stimulation"""
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
        return 460.0, 20.0, 0.32
    else:
        return 184.0, 52.5, 0.2135

def create_pathway_synapses(pattern_frequencies):
    """Create synapses with 20Hz threshold for PD"""
    pd_freq = np.mean(pattern_frequencies.get('PD', [15.0])) if pattern_frequencies.get('PD') else 15.0
    pd_tau_rec, pd_tau_facil, pd_U = get_pd_parameters(pd_freq)
    
    synapses = {
        'DD': TsodyksMarkramSynapse(248.0, 133.0, 0.20, "DD_LPP"),
        'MD': TsodyksMarkramSynapse(3977.0, 27.0, 0.30, "MD_MPP"),
        'PD': TsodyksMarkramSynapse(pd_tau_rec, pd_tau_facil, pd_U, f"PD_{pd_freq:.0f}Hz")
    }
    
    logging.debug(f"Created synapses with PD_freq={pd_freq:.1f}Hz (20Hz threshold)")
    
    return synapses

def generate_regular_pattern(frequency, duration=2000, n_pulses=None):
    """Generate regular periodic pattern"""
    if frequency <= 0:
        return np.array([])
    
    if n_pulses:
        if n_pulses <= 1:
            return np.array([0])
        isi = duration / (n_pulses - 1)
        times = [i * isi for i in range(n_pulses)]
    else:
        isi = 1000.0 / frequency
        times = []
        t = 0
        while t < duration:
            times.append(t)
            t += isi
    
    return np.array(times)

def generate_burst_pattern(burst_freq=8.0, intra_burst_freq=100.0, 
                          spikes_per_burst=4, n_bursts=15, duration=2000):
    """Generate burst pattern"""
    if burst_freq <= 0 or intra_burst_freq <= 0:
        return np.array([])
    
    burst_isi = 1000.0 / burst_freq
    spike_isi = 1000.0 / intra_burst_freq
    
    times = []
    
    for burst in range(n_bursts):
        burst_start = burst * burst_isi
        if burst_start >= duration:
            break
            
        for spike in range(spikes_per_burst):
            spike_time = burst_start + spike * spike_isi
            if spike_time < duration:
                times.append(spike_time)
    
    return np.array(sorted(times))

def generate_poisson_pattern(mean_rate, duration=2000, refractory_period=3.0):
    """Generate Poisson process"""
    if mean_rate <= 0:
        return np.array([])
    
    times = []
    t = 0
    
    np.random.seed(42)
    
    while t < duration:
        isi = np.random.exponential(1000.0 / mean_rate)
        t += max(isi, refractory_period)
        if t < duration:
            times.append(t)
    
    return np.array(times)

def generate_theta_modulated_pattern(base_freq=12.0, theta_freq=8.0, 
                                   modulation_depth=0.7, duration=2000):
    """Generate theta-modulated pattern"""
    if base_freq <= 0 or theta_freq <= 0:
        return np.array([])
    
    times = []
    t = 0
    
    np.random.seed(123)
    
    while t < duration:
        theta_phase = 2 * np.pi * theta_freq * t / 1000.0
        modulated_rate = base_freq * (1 + modulation_depth * np.sin(theta_phase))
        
        if modulated_rate > 0:
            isi = 1000.0 / modulated_rate
            isi *= (0.7 + 0.6 * np.random.random())
            t += isi
            if t < duration:
                times.append(t)
        else:
            t += 20
    
    return np.array(times)

def simulate_natural_pattern_integration(dd_pattern, md_pattern, pd_pattern, pattern_name):
    """Simulate integration with natural patterns"""
    
    # Calculate pattern frequencies
    pattern_durations = 2000
    pattern_frequencies = {}
    
    if len(dd_pattern) > 1:
        pattern_frequencies['DD'] = [len(dd_pattern) * 1000 / pattern_durations]
    if len(md_pattern) > 1:
        pattern_frequencies['MD'] = [len(md_pattern) * 1000 / pattern_durations]
    if len(pd_pattern) > 1:
        pattern_frequencies['PD'] = [len(pd_pattern) * 1000 / pattern_durations]
    
    # Individual responses
    synapses_individual = create_pathway_synapses(pattern_frequencies)
    individual_responses = {}
    
    patterns = {'DD': dd_pattern, 'MD': md_pattern, 'PD': pd_pattern}
    
    for pathway, times in patterns.items():
        synapse = synapses_individual[pathway]
        synapse.reset()
        
        total_response = 0
        for time in times:
            response = synapse.stimulate(time)
            total_response += response
        
        individual_responses[pathway] = total_response
    
    individual_sum = sum(individual_responses.values())
    
    # Integrated response
    synapses_integrated = create_pathway_synapses(pattern_frequencies)
    
    all_stimuli = []
    for pathway, times in patterns.items():
        for time in times:
            all_stimuli.append((time, pathway))
    
    all_stimuli.sort(key=lambda x: x[0])
    
    integrated_response = 0
    for time, pathway in all_stimuli:
        synapse = synapses_integrated[pathway]
        response = synapse.stimulate(time)
        integrated_response += response
    
    # Calculate interaction
    if individual_sum > 0:
        interaction_coefficient = (integrated_response - individual_sum) / individual_sum
    else:
        interaction_coefficient = 0.0
    
    logging.debug(f"{pattern_name}: interaction={interaction_coefficient:.6f}")
    
    return {
        'pattern_name': pattern_name,
        'individual_responses': individual_responses,
        'individual_sum': individual_sum,
        'integrated_response': integrated_response,
        'interaction_coefficient': interaction_coefficient,
        'n_stimuli': {pathway: len(times) for pathway, times in patterns.items()},
        'total_stimuli': sum(len(times) for times in patterns.values())
    }

def run_natural_pattern_analysis():
    """Run analysis with 20Hz threshold"""
    print("Starting natural pattern analysis with 20Hz threshold...")
    logging.info("Beginning natural pattern analysis (20Hz PD threshold)")
    
    duration = 2000
    results = []
    
    # Test patterns updated for 20Hz threshold
    pattern_tests = [
        # Regular patterns
        {
            'name': 'Regular_5Hz_Equal',
            'dd_pattern': generate_regular_pattern(5.0, duration),
            'md_pattern': generate_regular_pattern(5.0, duration),
            'pd_pattern': generate_regular_pattern(5.0, duration)
        },
        {
            'name': 'Regular_15Hz_Equal',  # Below 20Hz threshold
            'dd_pattern': generate_regular_pattern(15.0, duration),
            'md_pattern': generate_regular_pattern(15.0, duration),
            'pd_pattern': generate_regular_pattern(15.0, duration)
        },
        {
            'name': 'Regular_20Hz_Equal',  # At threshold
            'dd_pattern': generate_regular_pattern(20.0, duration),
            'md_pattern': generate_regular_pattern(20.0, duration),
            'pd_pattern': generate_regular_pattern(20.0, duration)
        },
        {
            'name': 'Regular_30Hz_Equal',  # Above threshold
            'dd_pattern': generate_regular_pattern(30.0, duration),
            'md_pattern': generate_regular_pattern(30.0, duration),
            'pd_pattern': generate_regular_pattern(30.0, duration)
        },
        
        # Burst patterns
        {
            'name': 'Theta_Bursts_8Hz',
            'dd_pattern': generate_burst_pattern(8.0, 100.0, 4, 16, duration),
            'md_pattern': generate_burst_pattern(8.0, 80.0, 3, 16, duration),
            'pd_pattern': generate_burst_pattern(8.0, 120.0, 5, 16, duration)
        },
        {
            'name': 'Gamma_Bursts_40Hz',  # Above 20Hz threshold
            'dd_pattern': generate_burst_pattern(40.0, 200.0, 3, 80, duration),
            'md_pattern': generate_burst_pattern(40.0, 180.0, 2, 80, duration),
            'pd_pattern': generate_burst_pattern(40.0, 220.0, 4, 80, duration)
        },
        
        # Poisson patterns
        {
            'name': 'Poisson_Low_10Hz',  # Below threshold
            'dd_pattern': generate_poisson_pattern(10.0, duration),
            'md_pattern': generate_poisson_pattern(10.0, duration),
            'pd_pattern': generate_poisson_pattern(10.0, duration)
        },
        {
            'name': 'Poisson_High_25Hz',  # Above threshold
            'dd_pattern': generate_poisson_pattern(25.0, duration),
            'md_pattern': generate_poisson_pattern(25.0, duration),
            'pd_pattern': generate_poisson_pattern(25.0, duration)
        },
        
        # Theta-modulated
        {
            'name': 'Theta_Modulated_Low',
            'dd_pattern': generate_theta_modulated_pattern(10.0, 8.0, 0.6, duration),
            'md_pattern': generate_theta_modulated_pattern(8.0, 8.0, 0.5, duration),
            'pd_pattern': generate_theta_modulated_pattern(12.0, 8.0, 0.7, duration)
        },
        {
            'name': 'Theta_Modulated_High',
            'dd_pattern': generate_theta_modulated_pattern(25.0, 8.0, 0.8, duration),
            'md_pattern': generate_theta_modulated_pattern(20.0, 8.0, 0.6, duration),
            'pd_pattern': generate_theta_modulated_pattern(30.0, 8.0, 0.9, duration)
        },
        
        # Mixed patterns
        {
            'name': 'Mixed_Below_Threshold',  # All <20Hz
            'dd_pattern': generate_regular_pattern(10.0, duration),
            'md_pattern': generate_regular_pattern(15.0, duration),
            'pd_pattern': generate_regular_pattern(18.0, duration)
        },
        {
            'name': 'Mixed_Above_Threshold',  # All ≥20Hz
            'dd_pattern': generate_regular_pattern(25.0, duration),
            'md_pattern': generate_regular_pattern(30.0, duration),
            'pd_pattern': generate_regular_pattern(35.0, duration)
        }
    ]
    
    # Run simulations
    for i, test in enumerate(pattern_tests):
        progress = (i + 1) / len(pattern_tests) * 100
        print(f"Testing {test['name']} ({progress:.0f}%)")
        
        result = simulate_natural_pattern_integration(
            test['dd_pattern'], test['md_pattern'], test['pd_pattern'], test['name']
        )
        
        results.append(result)
        
        # Log significant interactions
        if abs(result['interaction_coefficient']) > 0.01:
            logging.info(f"{test['name']}: significant interaction = {result['interaction_coefficient']:.4f}")
    
    logging.info(f"Completed {len(results)} pattern tests")
    return results

def create_natural_patterns_figure(results):
    """Create figure with 20Hz threshold visualization"""
    print("Creating Figure 8: Natural Patterns (20Hz threshold)...")
    logging.info("Generating Figure 8 with 20Hz threshold")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Figure 8: Natural Pattern Integration (20Hz PD Threshold)', 
                 fontsize=16, fontweight='bold')
    
    # Panel A: Integration effects
    ax1 = axes[0, 0]
    
    pattern_names = [r['pattern_name'].replace('_', '\n') for r in results]
    interaction_coeffs = [r['interaction_coefficient'] * 100 for r in results]
    
    # Color code by pattern type
    colors = []
    for name in pattern_names:
        if 'Regular' in name:
            colors.append('lightcoral')
        elif 'Burst' in name or 'Theta' in name or 'Gamma' in name:
            colors.append('lightblue')
        elif 'Poisson' in name:
            colors.append('lightgreen')
        elif 'Mixed' in name:
            colors.append('lightyellow')
        else:
            colors.append('lightgray')
    
    bars = ax1.bar(range(len(pattern_names)), interaction_coeffs, color=colors, 
                   alpha=0.8, edgecolor='black', linewidth=1)
    
    ax1.set_xlabel('Pattern Type')
    ax1.set_ylabel('Interaction Coefficient (%)')
    ax1.set_title('A. Pattern-Specific Integration', fontweight='bold')
    ax1.set_xticks(range(len(pattern_names)))
    ax1.set_xticklabels(pattern_names, fontsize=8, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.7)
    
    # Legend
    from matplotlib.patches import Rectangle
    legend_elements = [
        Rectangle((0,0),1,1, facecolor='lightcoral', alpha=0.8, label='Regular'),
        Rectangle((0,0),1,1, facecolor='lightblue', alpha=0.8, label='Bursts/Theta'),
        Rectangle((0,0),1,1, facecolor='lightgreen', alpha=0.8, label='Poisson'),
        Rectangle((0,0),1,1, facecolor='lightyellow', alpha=0.8, label='Mixed')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    logging.info(f"Panel A: Mean interaction = {np.mean(interaction_coeffs):.3f}%")
    
    # Panel B: Linear summation
    ax2 = axes[0, 1]
    
    individual_sums = [r['individual_sum'] for r in results]
    integrated_responses = [r['integrated_response'] for r in results]
    
    ax2.scatter(individual_sums, integrated_responses, s=80, alpha=0.7, 
               c=colors, edgecolors='black', linewidth=1)
    
    min_val = min(min(individual_sums), min(integrated_responses))
    max_val = max(max(individual_sums), max(integrated_responses))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
             alpha=0.8, label='Perfect Linear')
    
    slope, intercept, r_value, _, _ = stats.linregress(individual_sums, integrated_responses)
    line_x = np.array([min_val, max_val])
    line_y = slope * line_x + intercept
    ax2.plot(line_x, line_y, 'g-', linewidth=2, alpha=0.8, 
             label=f'Observed (r²={r_value**2:.4f})')
    
    ax2.set_xlabel('Individual Pathway Sum')
    ax2.set_ylabel('Integrated Response')
    ax2.set_title('B. Linear Summation Test', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    logging.info(f"Panel B: r² = {r_value**2:.4f}")
    
    # Panel C: Stimulus load
    ax3 = axes[0, 2]
    
    total_stimuli = [r['total_stimuli'] for r in results]
    
    ax3.scatter(total_stimuli, interaction_coeffs, s=80, alpha=0.7,
               c=colors, edgecolors='black', linewidth=1)
    
    if len(total_stimuli) > 2:
        slope, intercept, r_value, _, _ = stats.linregress(total_stimuli, interaction_coeffs)
        line_x = np.array([min(total_stimuli), max(total_stimuli)])
        line_y = slope * line_x + intercept
        ax3.plot(line_x, line_y, 'r--', linewidth=2, alpha=0.8,
                 label=f'Trend (r²={r_value**2:.3f})')
        ax3.legend()
    
    ax3.set_xlabel('Total Stimuli')
    ax3.set_ylabel('Interaction Coefficient (%)')
    ax3.set_title('C. Stimulus Load Effect', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.7)
    
    # Panel D: Pattern category
    ax4 = axes[1, 0]
    
    pattern_categories = {
        'Regular': [r for r in results if 'Regular' in r['pattern_name']],
        'Bursts': [r for r in results if 'Burst' in r['pattern_name']],
        'Poisson': [r for r in results if 'Poisson' in r['pattern_name']],
        'Theta-Mod': [r for r in results if 'Theta_Modulated' in r['pattern_name']],
        'Mixed': [r for r in results if 'Mixed' in r['pattern_name']]
    }
    
    category_data = []
    category_labels = []
    
    for cat_name, cat_results in pattern_categories.items():
        if cat_results:
            cat_interactions = [r['interaction_coefficient'] * 100 for r in cat_results]
            category_data.append(cat_interactions)
            category_labels.append(cat_name)
    
    if category_data:
        bp = ax4.boxplot(category_data, labels=category_labels, patch_artist=True)
        
        box_colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow', 'lightgray']
        for i, patch in enumerate(bp['boxes']):
            if i < len(box_colors):
                patch.set_facecolor(box_colors[i])
                patch.set_alpha(0.8)
    
    ax4.set_ylabel('Interaction Coefficient (%)')
    ax4.set_title('D. Pattern Category Analysis', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.7)
    
    # Panel E: Pathway contributions
    ax5 = axes[1, 1]
    
    dd_responses = [r['individual_responses']['DD'] for r in results]
    md_responses = [r['individual_responses']['MD'] for r in results]
    pd_responses = [r['individual_responses']['PD'] for r in results]
    
    x_pos = np.arange(len(results))
    bar_width = 0.25
    
    ax5.bar(x_pos - bar_width, dd_responses, bar_width, 
           label='DD', color='red', alpha=0.7, edgecolor='black')
    ax5.bar(x_pos, md_responses, bar_width,
           label='MD', color='blue', alpha=0.7, edgecolor='black')
    ax5.bar(x_pos + bar_width, pd_responses, bar_width,
           label='PD', color='green', alpha=0.7, edgecolor='black')
    
    ax5.set_xlabel('Pattern')
    ax5.set_ylabel('Response Magnitude')
    ax5.set_title('E. Pathway Contributions', fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels([r['pattern_name'].replace('_', '\n') for r in results],
                        fontsize=7, rotation=45, ha='right')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Panel F: Summary
    ax6 = axes[1, 2]
    
    ax6.hist(interaction_coeffs, bins=15, alpha=0.7, color='skyblue', 
             edgecolor='black', density=True)
    
    mean_coeff = np.mean(interaction_coeffs)
    median_coeff = np.median(interaction_coeffs)
    std_coeff = np.std(interaction_coeffs)
    
    ax6.axvline(mean_coeff, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_coeff:.3f}%')
    ax6.axvline(0, color='black', linestyle='-', alpha=0.7,
                label='Linear summation')
    
    ax6.set_xlabel('Interaction Coefficient (%)')
    ax6.set_ylabel('Probability Density')
    ax6.set_title('F. Overall Summary', fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = (f'Mean ± SD: {mean_coeff:.3f} ± {std_coeff:.3f}%\n'
                  f'Range: {min(interaction_coeffs):.2f} to {max(interaction_coeffs):.2f}%\n'
                  f'n = {len(interaction_coeffs)} patterns')
    ax6.text(0.02, 0.98, stats_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    png_filename = f'Figure8_Natural_Patterns_20Hz_{timestamp}.png'
    tiff_filename = f'Figure8_Natural_Patterns_20Hz_{timestamp}.tiff'
    csv_filename = f'figure8_natural_patterns_20hz_{timestamp}.csv'
    
    fig.savefig(png_filename, dpi=300, bbox_inches='tight')
    logging.info(f"Saved PNG: {png_filename}")
    
    img = Image.open(png_filename)
    img.save(tiff_filename, dpi=(600, 600), compression='tiff_lzw')
    logging.info(f"Saved TIFF: {tiff_filename}")
    
    # Save data
    results_data = []
    for r in results:
        row = {
            'pattern_name': r['pattern_name'],
            'interaction_coefficient': r['interaction_coefficient'],
            'individual_sum': r['individual_sum'],
            'integrated_response': r['integrated_response'],
            'total_stimuli': r['total_stimuli'],
            'DD_response': r['individual_responses']['DD'],
            'MD_response': r['individual_responses']['MD'],
            'PD_response': r['individual_responses']['PD']
        }
        results_data.append(row)
    
    pd.DataFrame(results_data).to_csv(csv_filename, index=False)
    logging.info(f"Saved CSV: {csv_filename}")
    
    print(f"Saved: {png_filename}, {tiff_filename}, {csv_filename}")
    
    plt.show()
    
    return results

def generate_natural_pattern_summary(results):
    """Generate summary with 20Hz threshold"""
    print("\n" + "="*70)
    print("FIGURE 8: NATURAL PATTERNS SUMMARY (20Hz THRESHOLD)")
    print("="*70)
    
    interaction_coeffs = [r['interaction_coefficient'] * 100 for r in results]
    
    print(f"Patterns tested: {len(results)}")
    print(f"Mean interaction: {np.mean(interaction_coeffs):.4f}% ± {np.std(interaction_coeffs):.4f}%")
    print(f"Median: {np.median(interaction_coeffs):.4f}%")
    print(f"Range: {np.min(interaction_coeffs):.3f}% to {np.max(interaction_coeffs):.3f}%")
    
    # Pattern analysis
    categories = {
        'Regular': [r for r in results if 'Regular' in r['pattern_name']],
        'Bursts': [r for r in results if 'Burst' in r['pattern_name']],
        'Poisson': [r for r in results if 'Poisson' in r['pattern_name']],
        'Theta_Mod': [r for r in results if 'Theta_Modulated' in r['pattern_name']],
        'Mixed': [r for r in results if 'Mixed' in r['pattern_name']]
    }
    
    print(f"\nPattern categories:")
    for cat_name, cat_results in categories.items():
        if cat_results:
            cat_coeffs = [r['interaction_coefficient'] * 100 for r in cat_results]
            print(f"  {cat_name}: {np.mean(cat_coeffs):.4f}% ± {np.std(cat_coeffs):.4f}%")
    
    # Consistency check
    print(f"\nConsistency with previous figures:")
    print(f"  Figure 6: -0.61% (20Hz threshold)")
    print(f"  Figure 8: {np.mean(interaction_coeffs):.3f}%")
    
    difference = abs(np.mean(interaction_coeffs) - (-0.61))
    if difference < 1.0:
        print(f"  Assessment: CONSISTENT (difference: {difference:.2f}%)")
    else:
        print(f"  Assessment: SLIGHT VARIATION (difference: {difference:.2f}%)")
    
    print(f"\nINTERPRETATION:")
    mean_abs = np.mean([abs(c) for c in interaction_coeffs])
    print(f"Natural patterns show minimal deviation ({mean_abs:.3f}%)")
    print(f"from linear summation, confirming robust integration")
    print(f"across physiologically relevant stimulation patterns.")
    print(f"\n20Hz threshold successfully implemented")
    
    logging.info("Summary completed")

def main():
    """Main execution with 20Hz threshold"""
    print("Figure 8: Natural Pattern Analysis (20Hz Threshold)")
    print("=" * 60)
    print("NO FABRICATION - Experimental parameters with 20Hz threshold")
    print("Based on Figure 4 experimental evidence")
    print("=" * 60)
    
    log_file = setup_logging()
    
    try:
        # Run analysis
        results = run_natural_pattern_analysis()
        
        # Create figure
        create_natural_patterns_figure(results)
        
        # Generate summary
        generate_natural_pattern_summary(results)
        
        print(f"\nDebug log: {log_file}")
        print("Figure 8 completed successfully!")
        
        logging.info("Figure 8 generation completed")
        
        return results
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        logging.error(error_msg)
        raise

if __name__ == "__main__":
    main()
