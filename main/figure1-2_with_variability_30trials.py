"""
Figure 1-2 Generator with Biological Variability (30 Trials)
=============================================================
Incorporates trial-to-trial variability based on experimental literature
Generates Figure 1 and Figure 2 with mean ± SEM across 30 trials
Outputs: PNG + 600 DPI TIFF (LZW compressed) + CSV data

BIOLOGICAL VARIABILITY REFERENCES:
- Nusser et al., 2001: mIPSC variability ~30%
- Smith et al., 2003: mEPSC amplitude CV = 0.69-0.75
- Implemented as CV = 0.20 (20%, conservative estimate)

Filename: figure1-2_with_variability_30trials.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from PIL import Image
import os
import logging
from datetime import datetime

def setup_logging():
    """Setup logging for simulation"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"figure1-2_variability_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*70)
    logging.info("FIGURE 1-2 GENERATION WITH BIOLOGICAL VARIABILITY")
    logging.info("30 trials per condition with parameter CV = 0.20")
    logging.info("="*70)
    
    return log_filename

# Global style settings for publication
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.2,
})

class TsodyksMarkramSynapse:
    """
    Tsodyks-Markram dynamic synapse model with biological variability
    
    References for parameter variability:
    - Nusser et al., 2001: Synaptic transmission shows trial-to-trial 
      variability of ~30% in mIPSC decay
    - Smith et al., 2003: mEPSC amplitude CV = 0.69-0.75 in hippocampal 
      CA1 synapses
    - Costa et al., 2013: Bayesian inference of STP models accounts for
      variability in synaptic responses
    """
    
    def __init__(self, tau_rec, tau_facil, U, name="synapse"):
        self.base_tau_rec = float(tau_rec)
        self.base_tau_facil = float(tau_facil)
        self.base_U = float(U)
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset synapse to initial state"""
        self.x = 1.0
        self.u = self.base_U
        self.last_time = 0.0
        # Store current parameter values for this trial
        self.tau_rec = self.base_tau_rec
        self.tau_facil = self.base_tau_facil
        self.U = self.base_U
    
    def set_trial_parameters(self, cv=0.20):
        """
        Set parameters with biological variability for a single trial
        
        Parameters:
        -----------
        cv : float
            Coefficient of variation (default 0.20 = 20%)
            Based on conservative estimate from literature:
            - Nusser et al., 2001: ~30% variability in synaptic responses
            - We use 20% as a conservative value
        """
        # Add Gaussian noise to parameters
        self.tau_rec = max(
            np.random.normal(self.base_tau_rec, self.base_tau_rec * cv),
            self.base_tau_rec * 0.5  # Lower bound
        )
        self.tau_facil = max(
            np.random.normal(self.base_tau_facil, self.base_tau_facil * cv),
            self.base_tau_facil * 0.5
        )
        self.U = np.clip(
            np.random.normal(self.base_U, self.base_U * cv),
            0.05, 0.95  # Physiological bounds
        )
        
        logging.debug(f"{self.name} trial params: τ_rec={self.tau_rec:.1f}, "
                     f"τ_facil={self.tau_facil:.1f}, U={self.U:.3f}")
    
    def stimulate(self, time):
        """
        Stimulate synapse at given time and return response
        Based on Tsodyks-Markram equations (Tsodyks & Markram, 1997)
        """
        dt = time - self.last_time
        
        if dt > 0:
            # Update available resources (recovery)
            self.x = 1 - (1 - self.x) * np.exp(-dt / self.tau_rec)
            # Update utilization (facilitation decay)
            self.u = self.U + (self.u - self.U) * np.exp(-dt / self.tau_facil)
        
        # Calculate synaptic response
        response = self.u * self.x
        
        # Update state after stimulation
        self.x = self.x * (1 - self.u)
        self.u = self.u + self.U * (1 - self.u)
        
        self.last_time = time
        return response

def simulate_pathway_response_single_trial(synapse, frequency, n_pulses=10):
    """
    Simulate a single trial of pathway response to pulse train
    Returns normalized response amplitudes (1st pulse = 100%)
    """
    if frequency <= 0:
        return np.array([100.0] * n_pulses)
    
    isi = 1000.0 / frequency
    
    synapse.reset()
    synapse.set_trial_parameters(cv=0.20)  # Apply variability
    responses = []
    
    for i in range(n_pulses):
        time = i * isi
        response = synapse.stimulate(time)
        responses.append(response)
    
    # Normalize to first pulse = 100%
    responses = np.array(responses)
    if responses[0] > 0:
        normalized = 100.0 * responses / responses[0]
    else:
        normalized = np.full(n_pulses, 100.0)
    
    return normalized

def simulate_pathway_response_multiple_trials(synapse, frequency, n_pulses=10, n_trials=30):
    """
    Simulate multiple trials and return mean ± SEM
    
    Parameters:
    -----------
    synapse : TsodyksMarkramSynapse
        Synapse object with base parameters
    frequency : float
        Stimulation frequency (Hz)
    n_pulses : int
        Number of pulses per train
    n_trials : int
        Number of trials (default 30)
    
    Returns:
    --------
    mean_response : np.array
        Mean normalized response across trials
    sem_response : np.array
        Standard error of the mean
    """
    all_trials = []
    
    for trial in range(n_trials):
        trial_response = simulate_pathway_response_single_trial(
            synapse, frequency, n_pulses
        )
        all_trials.append(trial_response)
    
    all_trials = np.array(all_trials)
    mean_response = np.mean(all_trials, axis=0)
    sem_response = stats.sem(all_trials, axis=0)
    
    return mean_response, sem_response

def create_pathway_synapses():
    """
    Create synapses with experimental parameters
    Base parameters from Hayakawa et al., 2014
    """
    logging.info("Creating pathway synapses with experimental parameters...")
    logging.info("Base parameters from Hayakawa et al., 2014")
    logging.info("Variability model: CV = 0.20 (Nusser et al., 2001)")
    
    synapses = {
        'DD': TsodyksMarkramSynapse(
            tau_rec=248.0,
            tau_facil=133.0,
            U=0.20,
            name="DD_LPP"
        ),
        'MD': TsodyksMarkramSynapse(
            tau_rec=3977.0,
            tau_facil=27.0,
            U=0.30,
            name="MD_MPP"
        ),
        'PD': TsodyksMarkramSynapse(
            tau_rec=460.0,
            tau_facil=20.0,
            U=0.32,
            name="PD_SuM_MC_MS"
        )
    }
    
    for name, synapse in synapses.items():
        logging.info(f"  {name}: τ_rec={synapse.base_tau_rec}ms, "
                    f"τ_facil={synapse.base_tau_facil}ms, U={synapse.base_U}")
    
    return synapses

def save_publication_figure(fig, basename, dpi=600):
    """Save figure as both PNG and compressed TIFF (600 dpi, LZW)"""
    png_file = f"{basename}.png"
    tiff_file = f"{basename}.tiff"
    
    fig.savefig(png_file, dpi=dpi, format="png", bbox_inches="tight")
    img = Image.open(png_file)
    img.save(tiff_file, dpi=(dpi, dpi), compression="tiff_lzw")
    
    print(f"[saved] {png_file}, {tiff_file}")
    return png_file, tiff_file

def create_figure1(n_trials=30):
    """
    Create Figure 1: Frequency-dependent response characteristics
    With biological variability across 30 trials
    """
    logging.info(f"Creating Figure 1 with {n_trials} trials per condition...")
    
    # Frequency range
    frequencies = np.array([0.1, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0])
    logging.info(f"Testing frequencies: {frequencies} Hz")
    
    # Create synapses
    synapses = create_pathway_synapses()
    
    # Calculate responses with variability
    results_data = []
    pathway_responses_mean = {}
    pathway_responses_sem = {}
    
    for pathway_name, synapse in synapses.items():
        logging.info(f"\nProcessing {pathway_name} pathway ({n_trials} trials)...")
        means = []
        sems = []
        
        for freq in frequencies:
            # Simulate multiple trials
            mean_train, sem_train = simulate_pathway_response_multiple_trials(
                synapse, freq, n_pulses=10, n_trials=n_trials
            )
            
            # Use 2nd pulse for frequency response
            response_mean = mean_train[1]
            response_sem = sem_train[1]
            
            means.append(response_mean)
            sems.append(response_sem)
            
            logging.info(f"  {freq} Hz: {response_mean:.1f} ± {response_sem:.1f}%")
            
            # Store for CSV output
            results_data.append({
                'pathway': pathway_name,
                'frequency_hz': freq,
                'response_mean_percent': response_mean,
                'response_sem_percent': response_sem,
                'pulse_number': 2,
                'n_trials': n_trials
            })
        
        pathway_responses_mean[pathway_name] = np.array(means)
        pathway_responses_sem[pathway_name] = np.array(sems)
    
    # Create publication-quality figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot with error bars
    colors = {'DD': '#377eb8', 'MD': '#4daf4a', 'PD': '#e41a1c'}
    markers = {'DD': 'o', 'MD': 's', 'PD': '^'}
    
    for pathway in ['DD', 'MD', 'PD']:
        ax.errorbar(
            frequencies, 
            pathway_responses_mean[pathway],
            yerr=pathway_responses_sem[pathway],
            marker=markers[pathway], 
            linewidth=2.5, 
            markersize=8,
            color=colors[pathway], 
            label=f'{pathway}',
            markerfacecolor='white',
            markeredgewidth=2, 
            markeredgecolor=colors[pathway],
            capsize=5,
            capthick=2
        )
    
    ax.set_xlabel('Frequency (Hz)', fontweight='bold')
    ax.set_ylabel('Normalized fEPSP amplitude (%)', fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, facecolor='white', edgecolor='black', 
              loc='best', fontsize=11)
    ax.set_ylim(0, 140)
    
    # Add note about trials
    ax.text(0.02, 0.98, f'Mean ± SEM (n={n_trials} trials)', 
            transform=ax.transAxes, fontsize=9, 
            verticalalignment='top')
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_file, tiff_file = save_publication_figure(
        fig, f"Figure1_variability_{timestamp}"
    )
    plt.close(fig)
    
    # Save numerical data
    df = pd.DataFrame(results_data)
    csv_file = f"Figure1_data_variability_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    logging.info(f"Saved numerical data: {csv_file}")
    
    return pathway_responses_mean, pathway_responses_sem, results_data

def create_figure2(pathway_responses_mean, n_trials=30):
    """
    Create Figure 2: Pathway integration heatmap
    Using mean responses across trials
    Custom colormap: blue (min) -> white (100%) -> red (max)
    """
    print(f"Creating Figure 2 (mean across {n_trials} trials)...")
    
    frequencies = np.array([0.1, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0])
    pathways = ['DD', 'MD', 'PD']
    
    # Create heatmap data matrix
    heatmap_data = np.array([pathway_responses_mean[pw] for pw in pathways])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create custom colormap: blue -> white (at 100%) -> red
    from matplotlib.colors import LinearSegmentedColormap
    
    # Find data range
    vmin = heatmap_data.min()
    vmax = heatmap_data.max()
    
    # Set 100% as the center (white)
    center = 100.0
    
    # Create custom colormap
    # If data spans across 100%, use diverging colormap
    if vmin < center < vmax:
        colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', 
                  '#ffffff',  # White at center (100%)
                  '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('custom_diverging', colors, N=n_bins)
        
        # Normalize so that 100% is at center (white)
        from matplotlib.colors import TwoSlopeNorm
        norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        
        im = ax.imshow(heatmap_data, cmap=cmap, norm=norm, aspect='auto',
                       extent=[np.log10(frequencies[0]), np.log10(frequencies[-1]), 
                              0, len(pathways)])
    else:
        # Fallback to standard colormap if data doesn't span 100%
        im = ax.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto',
                       extent=[np.log10(frequencies[0]), np.log10(frequencies[-1]), 
                              0, len(pathways)])
    
    # Set ticks and labels
    ax.set_yticks(np.arange(len(pathways)) + 0.5)
    ax.set_yticklabels(pathways)
    
    freq_positions = np.log10(frequencies)
    ax.set_xticks(freq_positions)
    ax.set_xticklabels([f'{f:g}' for f in frequencies])
    
    ax.set_xlabel('Frequency (Hz)', fontweight='bold')
    ax.set_ylabel('Pathway', fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Response amplitude (%)', fontweight='bold', 
                   rotation=270, labelpad=20)
    
    # Add note
    ax.text(0.02, 0.98, f'Mean values (n={n_trials} trials)', 
            transform=ax.transAxes, fontsize=9, 
            verticalalignment='top', color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_file, tiff_file = save_publication_figure(
        fig, f"Figure2_variability_{timestamp}"
    )
    plt.close(fig)
    
    return heatmap_data

def generate_summary_statistics(results_data):
    """Generate summary statistics"""
    df = pd.DataFrame(results_data)
    
    print("\n" + "="*70)
    print("SIMULATION RESULTS SUMMARY (WITH BIOLOGICAL VARIABILITY)")
    print("="*70)
    print(f"Number of trials per condition: {df['n_trials'].iloc[0]}")
    print(f"Parameter variability: CV = 0.20 (20%)")
    print("\nReferences:")
    print("- Nusser et al., 2001: ~30% trial-to-trial variability")
    print("- Smith et al., 2003: mEPSC CV = 0.69-0.75")
    print("="*70)
    
    for pathway in ['DD', 'MD', 'PD']:
        pathway_data = df[df['pathway'] == pathway]
        
        freq_1hz = pathway_data[pathway_data['frequency_hz'] == 1.0]
        freq_40hz = pathway_data[pathway_data['frequency_hz'] == 40.0]
        
        mean_1hz = freq_1hz['response_mean_percent'].iloc[0]
        sem_1hz = freq_1hz['response_sem_percent'].iloc[0]
        mean_40hz = freq_40hz['response_mean_percent'].iloc[0]
        sem_40hz = freq_40hz['response_sem_percent'].iloc[0]
        
        print(f"\n{pathway} Pathway:")
        print(f"  Response at 1 Hz: {mean_1hz:.1f} ± {sem_1hz:.1f}%")
        print(f"  Response at 40 Hz: {mean_40hz:.1f} ± {sem_40hz:.1f}%")
        print(f"  Frequency dependence: {mean_40hz/mean_1hz:.3f} (40Hz/1Hz ratio)")

def main():
    """Main execution function"""
    print("="*70)
    print("FIGURE 1-2 GENERATION WITH BIOLOGICAL VARIABILITY")
    print("30 trials per condition, CV = 0.20")
    print("="*70)
    
    # Setup logging
    log_file = setup_logging()
    
    n_trials = 30
    
    try:
        # Generate Figure 1 with variability
        logging.info("Starting Figure 1 generation...")
        pathway_responses_mean, pathway_responses_sem, results_data = create_figure1(n_trials)
        
        # Generate Figure 2 with mean values
        logging.info("Starting Figure 2 generation...")
        heatmap_data = create_figure2(pathway_responses_mean, n_trials)
        
        # Generate summary statistics
        logging.info("Generating summary statistics...")
        generate_summary_statistics(results_data)
        
        logging.info("="*70)
        logging.info("FIGURE GENERATION COMPLETE")
        logging.info("="*70)
        logging.info("Files generated:")
        logging.info("- Figure1_variability_*.png/tiff (with error bars)")
        logging.info("- Figure2_variability_*.png/tiff (mean values)")
        logging.info("- Figure1_data_variability_*.csv")
        logging.info(f"- {log_file}")
        logging.info("")
        logging.info(f"All data based on {n_trials} trials with CV=0.20")
        logging.info("Variability based on Nusser et al., 2001")
        
        print(f"\nLog saved as: {log_file}")
        print("All files generated successfully!")
        
    except Exception as e:
        logging.error(f"Error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
