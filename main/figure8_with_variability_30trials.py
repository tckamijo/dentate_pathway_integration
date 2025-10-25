"""
Figure 8 - Natural Input Patterns Analysis with Biological Variability
=======================================================================
30 trials per condition with CV = 0.20
Tests realistic biological patterns (regular, burst, Poisson, theta-modulated)

BIOLOGICAL VARIABILITY REFERENCES:
- Nusser et al., 2001: mIPSC variability ~30%
- Smith et al., 2003: mEPSC amplitude CV = 0.69-0.75
- Implemented as CV = 0.20 (20%, conservative estimate)

Filename: figure8_with_variability_30trials.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import logging
from datetime import datetime
from scipy import stats

def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"figure8_variability_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    
    logging.info("="*70)
    logging.info("FIGURE 8: NATURAL PATTERNS WITH VARIABILITY")
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
    def __init__(self, tau_rec, tau_facil, U, pathway_name="synapse"):
        self.base_tau_rec, self.base_tau_facil, self.base_U = float(tau_rec), float(tau_facil), float(U)
        self.pathway_name = pathway_name
        self.reset()
    
    def reset(self):
        self.x, self.u, self.last_time = 1.0, self.base_U, 0.0
        self.tau_rec, self.tau_facil, self.U = self.base_tau_rec, self.base_tau_facil, self.base_U
    
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
        return response

def get_pd_parameters(frequency):
    return (460.0, 20.0, 0.32) if frequency < 20.0 else (184.0, 52.5, 0.2135)

def create_pathway_synapses(pattern_frequencies):
    pd_pattern = pattern_frequencies.get('PD', [15.0])
    # Check if array is not empty before computing mean
    if len(pd_pattern) > 0:
        pd_freq = np.mean(pd_pattern)
    else:
        pd_freq = 15.0
    
    pd_tau_rec, pd_tau_facil, pd_U = get_pd_parameters(pd_freq)
    
    return {
        'DD': TsodyksMarkramSynapse(248.0, 133.0, 0.20, "DD_LPP"),
        'MD': TsodyksMarkramSynapse(3977.0, 27.0, 0.30, "MD_MPP"),
        'PD': TsodyksMarkramSynapse(pd_tau_rec, pd_tau_facil, pd_U, f"PD_{pd_freq:.0f}Hz")
    }

def generate_regular_pattern(frequency, duration=2000, n_pulses=None):
    if frequency <= 0: return np.array([])
    if n_pulses:
        isi = duration / (n_pulses - 1) if n_pulses > 1 else 0
        return np.array([i * isi for i in range(n_pulses)])
    isi = 1000.0 / frequency
    return np.array([t for t in np.arange(0, duration, isi)])

def generate_burst_pattern(burst_freq=8.0, intra_burst_freq=100.0, spikes_per_burst=4, n_bursts=15, duration=2000):
    if burst_freq <= 0 or intra_burst_freq <= 0: return np.array([])
    burst_isi, spike_isi = 1000.0 / burst_freq, 1000.0 / intra_burst_freq
    times = []
    for burst in range(n_bursts):
        burst_start = burst * burst_isi
        if burst_start >= duration: break
        for spike in range(spikes_per_burst):
            spike_time = burst_start + spike * spike_isi
            if spike_time < duration: times.append(spike_time)
    return np.array(sorted(times))

def generate_poisson_pattern(mean_rate, duration=2000, refractory_period=3.0, seed=42):
    if mean_rate <= 0: return np.array([])
    np.random.seed(seed)
    times, t = [], 0
    while t < duration:
        t += max(np.random.exponential(1000.0 / mean_rate), refractory_period)
        if t < duration: times.append(t)
    return np.array(times)

def generate_theta_modulated_pattern(base_freq=12.0, theta_freq=8.0, modulation_depth=0.7, duration=2000, seed=123):
    if base_freq <= 0 or theta_freq <= 0: return np.array([])
    np.random.seed(seed)
    times, t = [], 0
    while t < duration:
        theta_phase = 2 * np.pi * theta_freq * t / 1000.0
        modulated_rate = base_freq * (1 + modulation_depth * np.sin(theta_phase))
        if modulated_rate > 0:
            t += (1000.0 / modulated_rate) * (0.7 + 0.6 * np.random.random())
            if t < duration: times.append(t)
        else:
            t += 20
    return np.array(times)

def simulate_natural_pattern_integration_single_trial(dd_pattern, md_pattern, pd_pattern, cv=0.20):
    duration = 2000
    pattern_frequencies = {}
    if len(dd_pattern) > 1: pattern_frequencies['DD'] = [len(dd_pattern) * 1000 / duration]
    if len(md_pattern) > 1: pattern_frequencies['MD'] = [len(md_pattern) * 1000 / duration]
    if len(pd_pattern) > 1: pattern_frequencies['PD'] = [len(pd_pattern) * 1000 / duration]
    
    # Individual responses
    synapses_individual = create_pathway_synapses(pattern_frequencies)
    for syn in synapses_individual.values(): syn.set_trial_parameters(cv)
    
    individual_responses = {}
    patterns = {'DD': dd_pattern, 'MD': md_pattern, 'PD': pd_pattern}
    
    for pathway, times in patterns.items():
        syn = synapses_individual[pathway]
        syn.reset()
        total_response = sum(syn.stimulate(time) for time in times)
        individual_responses[pathway] = total_response
    
    individual_sum = sum(individual_responses.values())
    
    # Integrated response
    synapses_integrated = create_pathway_synapses(pattern_frequencies)
    for syn in synapses_integrated.values(): syn.set_trial_parameters(cv)
    
    all_stimuli = [(time, pathway) for pathway, times in patterns.items() for time in times]
    all_stimuli.sort()
    
    integrated_response = sum(synapses_integrated[pathway].stimulate(time) for time, pathway in all_stimuli)
    
    interaction_coefficient = ((integrated_response - individual_sum) / individual_sum 
                              if individual_sum > 0 else 0.0)
    
    return {
        'individual_responses': individual_responses,
        'individual_sum': individual_sum,
        'integrated_response': integrated_response,
        'interaction_coefficient': interaction_coefficient
    }

def simulate_natural_pattern_integration_multiple_trials(dd_pattern, md_pattern, pd_pattern, pattern_name, n_trials=30, cv=0.20):
    all_results = [simulate_natural_pattern_integration_single_trial(dd_pattern, md_pattern, pd_pattern, cv) 
                  for _ in range(n_trials)]
    
    return {
        'pattern_name': pattern_name,
        'individual_responses': {
            pathway: np.mean([r['individual_responses'][pathway] for r in all_results])
            for pathway in ['DD', 'MD', 'PD']
        },
        'individual_sum': np.mean([r['individual_sum'] for r in all_results]),
        'integrated_response': np.mean([r['integrated_response'] for r in all_results]),
        'interaction_coefficient': np.mean([r['interaction_coefficient'] for r in all_results]),
        'interaction_sem': stats.sem([r['interaction_coefficient'] for r in all_results]),
        'total_stimuli': len(dd_pattern) + len(md_pattern) + len(pd_pattern)
    }

def run_natural_patterns_analysis(n_trials=30, cv=0.20):
    print(f"Starting natural patterns analysis ({n_trials} trials)...")
    logging.info(f"Beginning analysis with {n_trials} trials, CV={cv}")
    
    # Test patterns
    test_patterns = []
    
    # Regular patterns
    for freq in [5, 10, 15, 20, 30]:
        dd = generate_regular_pattern(freq, n_pulses=15)
        md = generate_regular_pattern(freq, n_pulses=15)
        pd = generate_regular_pattern(freq, n_pulses=15)
        test_patterns.append((dd, md, pd, f"Regular_{freq}Hz"))
    
    # Burst patterns
    test_patterns.append((
        generate_burst_pattern(8, 100, 4, 15),
        generate_burst_pattern(8, 100, 4, 15),
        generate_burst_pattern(8, 100, 4, 15),
        "Burst_Theta"
    ))
    
    test_patterns.append((
        generate_burst_pattern(30, 200, 5, 10),
        generate_burst_pattern(30, 200, 5, 10),
        generate_burst_pattern(30, 200, 5, 10),
        "Burst_Gamma"
    ))
    
    # Poisson patterns
    for rate in [10, 20, 30]:
        test_patterns.append((
            generate_poisson_pattern(rate, seed=42),
            generate_poisson_pattern(rate, seed=43),
            generate_poisson_pattern(rate, seed=44),
            f"Poisson_{rate}Hz"
        ))
    
    # Theta-modulated
    test_patterns.append((
        generate_theta_modulated_pattern(12, 8, 0.7, seed=123),
        generate_theta_modulated_pattern(12, 8, 0.7, seed=124),
        generate_theta_modulated_pattern(12, 8, 0.7, seed=125),
        "Theta_Modulated"
    ))
    
    # Mixed patterns
    test_patterns.append((
        generate_regular_pattern(10, n_pulses=15),
        generate_burst_pattern(8, 100, 4, 15),
        generate_poisson_pattern(15, seed=45),
        "Mixed_Pattern"
    ))
    
    results = []
    for i, (dd, md, pd, name) in enumerate(test_patterns):
        progress = (i + 1) / len(test_patterns) * 100
        print(f"Testing {name} ({progress:.0f}%)")
        
        result = simulate_natural_pattern_integration_multiple_trials(dd, md, pd, name, n_trials, cv)
        results.append(result)
    
    logging.info(f"Completed {len(results)} patterns")
    return results

def create_natural_patterns_figure(results):
    print("Creating Figure 8...")
    logging.info("Generating Figure 8")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    n_trials = 30
    cv = 0.20
    
    fig.suptitle(f'Figure 8: Natural Input Patterns Analysis\nMean ± SEM (n={n_trials} trials, CV={cv})', 
                fontsize=16, fontweight='bold')
    
    pattern_names = [r['pattern_name'] for r in results]
    interaction_coeffs = [r['interaction_coefficient'] * 100 for r in results]
    
    # Panel A: Pattern-specific integration
    ax1 = axes[0, 0]
    
    colors, hatches = [], []
    for name in pattern_names:
        if 'Regular' in name:
            colors.append('#636363'); hatches.append('')
        elif 'Burst' in name or 'Theta' in name or 'Gamma' in name:
            colors.append('#969696'); hatches.append('///')
        elif 'Poisson' in name:
            colors.append('#969696'); hatches.append('\\\\\\')
        elif 'Mixed' in name:
            colors.append('#d9d9d9'); hatches.append('xxx')
        else:
            colors.append('#cccccc'); hatches.append('...')
    
    bars = ax1.bar(range(len(pattern_names)), interaction_coeffs, color=colors, 
                   alpha=0.9, edgecolor='black', linewidth=1.5)
    for bar, hatch in zip(bars, hatches): bar.set_hatch(hatch)
    
    ax1.set_xlabel('Pattern Type')
    ax1.set_ylabel('Interaction Coefficient (%)')
    ax1.set_title('A. Pattern-Specific Integration', fontweight='bold')
    ax1.set_xticks(range(len(pattern_names)))
    ax1.set_xticklabels(pattern_names, fontsize=8, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.7)
    
    from matplotlib.patches import Rectangle
    legend_elements = [
        Rectangle((0,0),1,1, facecolor='#636363', edgecolor='black', hatch='', linewidth=1.5, label='Regular'),
        Rectangle((0,0),1,1, facecolor='#969696', edgecolor='black', hatch='///', linewidth=1.5, label='Bursts/Theta'),
        Rectangle((0,0),1,1, facecolor='#969696', edgecolor='black', hatch='\\\\\\', linewidth=1.5, label='Poisson'),
        Rectangle((0,0),1,1, facecolor='#d9d9d9', edgecolor='black', hatch='xxx', linewidth=1.5, label='Mixed')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Panel B: Linear summation
    ax2 = axes[0, 1]
    individual_sums = [r['individual_sum'] for r in results]
    integrated_responses = [r['integrated_response'] for r in results]
    
    ax2.scatter(individual_sums, integrated_responses, s=80, alpha=0.7, 
               facecolors='none', edgecolors='black', linewidths=1.5)
    
    min_val = min(min(individual_sums), min(integrated_responses))
    max_val = max(max(individual_sums), max(integrated_responses))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8, label='Perfect Linear')
    
    slope, intercept, r_value, _, _ = stats.linregress(individual_sums, integrated_responses)
    line_x = np.array([min_val, max_val])
    line_y = slope * line_x + intercept
    ax2.plot(line_x, line_y, 'g-', linewidth=2, alpha=0.8, label=f'Observed (r²={r_value**2:.4f})')
    
    ax2.set_xlabel('Individual Pathway Sum')
    ax2.set_ylabel('Integrated Response')
    ax2.set_title('B. Linear Summation Test', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Stimulus load
    ax3 = axes[0, 2]
    total_stimuli = [r['total_stimuli'] for r in results]
    
    ax3.scatter(total_stimuli, interaction_coeffs, s=80, alpha=0.7,
               facecolors='none', edgecolors='black', linewidths=1.5)
    
    if len(total_stimuli) > 2:
        slope, intercept, r_value, _, _ = stats.linregress(total_stimuli, interaction_coeffs)
        line_x = np.array([min(total_stimuli), max(total_stimuli)])
        line_y = slope * line_x + intercept
        ax3.plot(line_x, line_y, 'r--', linewidth=2, alpha=0.8, label=f'Trend (r²={r_value**2:.3f})')
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
    
    category_data, category_labels = [], []
    for cat_name, cat_results in pattern_categories.items():
        if cat_results:
            category_data.append([r['interaction_coefficient'] * 100 for r in cat_results])
            category_labels.append(cat_name)
    
    if category_data:
        bp = ax4.boxplot(category_data, labels=category_labels, patch_artist=True)
        box_colors = ['#636363', '#969696', '#969696', '#d9d9d9', '#cccccc']
        box_hatches = ['', '///', '\\\\\\', 'xxx', '...']
        
        for i, patch in enumerate(bp['boxes']):
            if i < len(box_colors):
                patch.set_facecolor(box_colors[i])
                patch.set_hatch(box_hatches[i])
                patch.set_alpha(0.9)
                patch.set_edgecolor('black')
                patch.set_linewidth(1.5)
    
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
    
    ax5.bar(x_pos - bar_width, dd_responses, bar_width, label='DD', 
           color='#636363', alpha=0.9, edgecolor='black', linewidth=1.5)
    ax5.bar(x_pos, md_responses, bar_width, label='MD', 
           color='#969696', alpha=0.9, edgecolor='black', linewidth=1.5)
    ax5.bar(x_pos + bar_width, pd_responses, bar_width, label='PD', 
           color='#cccccc', alpha=0.9, edgecolor='black', linewidth=1.5)
    
    ax5.set_xlabel('Pattern')
    ax5.set_ylabel('Individual Response')
    ax5.set_title('E. Pathway Contributions', fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(range(1, len(results)+1))
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Panel F: Overall Summary (RESTORED)
    ax6 = axes[1, 2]
    
    # Create histogram
    ax6.hist(interaction_coeffs, bins=15, alpha=0.7, color='skyblue',
             edgecolor='black', density=True, linewidth=1.5)
    
    # Calculate statistics
    mean_coeff = np.mean(interaction_coeffs)
    median_coeff = np.median(interaction_coeffs)
    sem_coeff = stats.sem(interaction_coeffs)
    
    # Add vertical lines
    ax6.axvline(mean_coeff, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_coeff:.3f}%')
    ax6.axvline(0, color='black', linestyle='-', alpha=0.7,
                label='Linear summation')
    
    ax6.set_xlabel('Interaction Coefficient (%)')
    ax6.set_ylabel('Probability Density')
    ax6.set_title('F. Overall Summary', fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = (f'Mean ± SEM: {mean_coeff:.3f} ± {sem_coeff:.3f}%\n'
                  f'Range: {min(interaction_coeffs):.2f} to {max(interaction_coeffs):.2f}%\n'
                  f'n = {len(interaction_coeffs)} patterns\n'
                  f'({n_trials} trials each, CV={cv})')
    ax6.text(0.02, 0.98, stats_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Add overall figure title
    fig.suptitle('Figure 8. Natural Input Patterns Analysis\n' + 
                 f'Mean ± SEM (n={n_trials} trials, CV={cv})',
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_filename = f'Figure8_patterns_variability_{timestamp}.png'
    tiff_filename = f'Figure8_patterns_variability_{timestamp}.tiff'
    csv_filename = f'Figure8_data_variability_{timestamp}.csv'
    
    fig.savefig(png_filename, dpi=600, bbox_inches='tight')
    Image.open(png_filename).save(tiff_filename, dpi=(600, 600), compression='tiff_lzw')
    plt.close(fig)
    
    pd.DataFrame(results).to_csv(csv_filename, index=False)
    
    logging.info(f"Saved: {png_filename}, {tiff_filename}, {csv_filename}")
    return png_filename, tiff_filename, csv_filename

def main():
    print("="*70)
    print("FIGURE 8: NATURAL PATTERNS (30 trials, CV=0.20)")
    print("="*70)
    
    log_file = setup_logging()
    
    try:
        results = run_natural_patterns_analysis(n_trials=30, cv=0.20)
        png_file, tiff_file, csv_file = create_natural_patterns_figure(results)
        
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
