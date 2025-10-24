"""
Figure 3 - Hayakawa 2014 Reproduction with Biological Variability
==================================================================
Reproduces Hayakawa et al., 2014 Fig.3 with 30 trials per condition
Incorporates trial-to-trial variability (CV = 0.20)
Outputs: PNG + 600 DPI TIFF (LZW compressed) + CSV data

BIOLOGICAL VARIABILITY REFERENCES:
- Nusser et al., 2001: mIPSC variability ~30%
- Smith et al., 2003: mEPSC amplitude CV = 0.69-0.75
- Implemented as CV = 0.20 (20%, conservative estimate)

Filename: figure3_with_variability_30trials.py
"""

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from PIL import Image
import pandas as pd
import logging

def setup_logging():
    """Setup logging for simulation"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"figure3_variability_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*70)
    logging.info("FIGURE 3: HAYAKAWA 2014 REPRODUCTION WITH VARIABILITY")
    logging.info("30 trials per condition with parameter CV = 0.20")
    logging.info("="*70)
    
    return log_filename

# Global figure style
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

class TMSynapse:
    """
    Tsodyks-Markram synapse with biological variability
    
    References:
    - Tsodyks & Markram, 1997: Original TM model
    - Nusser et al., 2001: Trial-to-trial variability ~30%
    - Smith et al., 2003: mEPSC CV = 0.69-0.75
    """
    
    def __init__(self, tau_rec, tau_facil, U, weight=1.0):
        self.base_tau_rec = float(tau_rec)
        self.base_tau_facil = float(tau_facil)
        self.base_U = float(U)
        self.weight = float(weight)
        self.reset()
    
    def reset(self):
        """Reset synapse to initial state"""
        self.x = 1.0
        self.u = self.base_U
        # Store current trial parameters
        self.tau_rec = self.base_tau_rec
        self.tau_facil = self.base_tau_facil
        self.U = self.base_U
    
    def set_trial_parameters(self, cv=0.20):
        """
        Set parameters with biological variability
        
        Parameters:
        -----------
        cv : float
            Coefficient of variation (default 0.20)
            Based on Nusser et al., 2001 (~30% variability)
            Using 20% as conservative estimate
        """
        self.tau_rec = max(
            np.random.normal(self.base_tau_rec, self.base_tau_rec * cv),
            self.base_tau_rec * 0.5
        )
        self.tau_facil = max(
            np.random.normal(self.base_tau_facil, self.base_tau_facil * cv),
            self.base_tau_facil * 0.5
        )
        self.U = np.clip(
            np.random.normal(self.base_U, self.base_U * cv),
            0.05, 0.95
        )
    
    def spike(self):
        """Process a spike and return synaptic response"""
        rel = self.u * self.x
        self.x -= rel
        self.u += self.U * (1.0 - self.u)
        return self.weight * rel
    
    def evolve(self, dt):
        """Evolve synapse state over time dt (ms)"""
        self.x += (1.0 - self.x) * (dt / self.tau_rec)
        self.u += (self.U - self.u) * (dt / self.tau_facil)
        self.x = min(max(self.x, 0.0), 1.0)
        self.u = max(self.u, self.U)

def simulate_train_single_trial(tau_rec, tau_facil, U, freq_hz, n_pulses=10, cv=0.20):
    """
    Simulate a single trial of pulse train
    
    Parameters:
    -----------
    tau_rec, tau_facil, U : float
        Base TM parameters
    freq_hz : float
        Stimulation frequency
    n_pulses : int
        Number of pulses
    cv : float
        Coefficient of variation for parameter variability
    
    Returns:
    --------
    normalized_response : list
        Response normalized to first pulse = 100%
    """
    syn = TMSynapse(tau_rec, tau_facil, U)
    syn.reset()
    syn.set_trial_parameters(cv=cv)
    
    isi = 1000.0 / float(freq_hz)  # Inter-stimulus interval (ms)
    dt = 0.1  # Time step (ms)
    
    out = []
    for k in range(n_pulses):
        if k > 0:
            # Evolve between pulses
            for _ in range(int(isi / dt)):
                syn.evolve(dt)
        out.append(syn.spike())
    
    # Normalize to first pulse = 100%
    if out[0] > 0:
        normalized = [100.0 * v / out[0] for v in out]
    else:
        normalized = [100.0] * n_pulses
    
    return normalized

def simulate_train_multiple_trials(tau_rec, tau_facil, U, freq_hz, 
                                   n_pulses=10, n_trials=30, cv=0.20):
    """
    Simulate multiple trials and return mean ± SEM
    
    Returns:
    --------
    mean_response : np.array
        Mean normalized response across trials
    sem_response : np.array
        Standard error of the mean
    """
    all_trials = []
    
    for trial in range(n_trials):
        trial_response = simulate_train_single_trial(
            tau_rec, tau_facil, U, freq_hz, n_pulses, cv
        )
        all_trials.append(trial_response)
    
    all_trials = np.array(all_trials)
    mean_response = np.mean(all_trials, axis=0)
    sem_response = stats.sem(all_trials, axis=0)
    
    return mean_response, sem_response

def save_publication_figure(fig, basename, dpi=600):
    """Save figure as both PNG and compressed TIFF (LZW)"""
    png_file = f"{basename}.png"
    tiff_file = f"{basename}.tiff"
    
    fig.savefig(png_file, dpi=dpi, bbox_inches="tight", format="png")
    img = Image.open(png_file)
    img.save(tiff_file, dpi=(dpi, dpi), compression="tiff_lzw")
    
    print(f"[saved] {png_file}, {tiff_file}")
    return png_file, tiff_file

def create_figure_with_variability(n_trials=30, cv=0.20):
    """
    Create Figure 3 with biological variability
    
    Parameters:
    -----------
    n_trials : int
        Number of trials per condition (default 30)
    cv : float
        Coefficient of variation for parameters (default 0.20)
    """
    logging.info(f"Creating Figure 3 with {n_trials} trials per condition (CV={cv})...")
    
    # Test frequencies (Hz)
    freqs = [0.1, 1, 2, 5, 10, 20]
    
    # Pathway parameters (Hayakawa et al., 2014)
    params = {
        "DD": (248.0, 133.0, 0.20),
        "MD": (3977.0, 27.0, 0.30),
        "PD": (460.2, 20.0, 0.318),
    }
    
    logging.info("Pathway parameters (Hayakawa et al., 2014):")
    for name, (tau_rec, tau_facil, U) in params.items():
        logging.info(f"  {name}: τ_rec={tau_rec}ms, τ_facil={tau_facil}ms, U={U}")
    
    # Store data for CSV export
    csv_data = []
    
    # Create subplots
    fig, axes = plt.subplots(1, len(freqs), figsize=(18, 4))
    if len(freqs) == 1:
        axes = [axes]
    
    # Colors for pathways
    colors = {'DD': '#377eb8', 'MD': '#4daf4a', 'PD': '#e41a1c'}
    
    for i, f in enumerate(freqs):
        logging.info(f"\nProcessing {f} Hz ({n_trials} trials)...")
        ax = axes[i]
        
        for name, (tau_rec, tau_facil, U) in params.items():
            # Simulate with variability
            mean_y, sem_y = simulate_train_multiple_trials(
                tau_rec, tau_facil, U, f, n_pulses=10, n_trials=n_trials, cv=cv
            )
            
            pulse_numbers = np.arange(1, len(mean_y) + 1)
            
            # Plot with error bars
            ax.errorbar(
                pulse_numbers, mean_y, yerr=sem_y,
                fmt='o-', linewidth=2, markersize=5,
                color=colors[name], label=name,
                capsize=3, capthick=1.5, alpha=0.8
            )
            
            # Store data for CSV
            for pulse_num, mean_val, sem_val in zip(pulse_numbers, mean_y, sem_y):
                csv_data.append({
                    'pathway': name,
                    'frequency_hz': f,
                    'pulse_number': pulse_num,
                    'response_mean_percent': mean_val,
                    'response_sem_percent': sem_val,
                    'n_trials': n_trials,
                    'cv': cv
                })
            
            logging.info(f"  {name}: P1={mean_y[0]:.1f}±{sem_y[0]:.1f}%, "
                        f"P10={mean_y[-1]:.1f}±{sem_y[-1]:.1f}%")
        
        ax.set_title(f"{f} Hz", fontweight="bold")
        ax.set_xlabel("Pulse number")
        if i == 0:
            ax.set_ylabel("Normalized fEPSP [%]")
        ax.set_ylim(20, 140)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc="lower left", frameon=True, facecolor='white')
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.suptitle(
        f"Reproduction of Hayakawa 2014 Fig.3 with Biological Variability\n"
        f"Mean ± SEM (n={n_trials} trials, CV={cv})",
        fontsize=14, fontweight="bold", y=0.98
    )
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    basename = f"Figure3_Hayakawa2014_variability_{timestamp}"
    png_file, tiff_file = save_publication_figure(fig, basename)
    plt.close(fig)
    
    # Save CSV data
    df = pd.DataFrame(csv_data)
    csv_file = f"Figure3_data_variability_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    logging.info(f"\nSaved numerical data: {csv_file}")
    
    return csv_data, png_file, tiff_file, csv_file

def generate_summary_statistics(csv_data):
    """Generate summary statistics"""
    df = pd.DataFrame(csv_data)
    
    print("\n" + "="*70)
    print("HAYAKAWA 2014 REPRODUCTION - SUMMARY WITH VARIABILITY")
    print("="*70)
    print(f"Number of trials: {df['n_trials'].iloc[0]}")
    print(f"Parameter CV: {df['cv'].iloc[0]}")
    print("\nReferences:")
    print("- Hayakawa et al., 2014: Base parameters")
    print("- Nusser et al., 2001: ~30% trial-to-trial variability")
    print("- Smith et al., 2003: mEPSC CV = 0.69-0.75")
    print("="*70)
    
    for freq in [0.1, 1, 5, 20]:
        print(f"\n{freq} Hz:")
        freq_data = df[df['frequency_hz'] == freq]
        
        for pathway in ['DD', 'MD', 'PD']:
            pathway_data = freq_data[freq_data['pathway'] == pathway]
            
            p1 = pathway_data[pathway_data['pulse_number'] == 1]
            p10 = pathway_data[pathway_data['pulse_number'] == 10]
            
            p1_mean = p1['response_mean_percent'].iloc[0]
            p1_sem = p1['response_sem_percent'].iloc[0]
            p10_mean = p10['response_mean_percent'].iloc[0]
            p10_sem = p10['response_sem_percent'].iloc[0]
            
            print(f"  {pathway}: P1={p1_mean:.1f}±{p1_sem:.1f}%, "
                  f"P10={p10_mean:.1f}±{p10_sem:.1f}%, "
                  f"Ratio={p10_mean/p1_mean:.3f}")

def main():
    """Main execution"""
    print("="*70)
    print("FIGURE 3: HAYAKAWA 2014 REPRODUCTION WITH VARIABILITY")
    print("30 trials per condition, CV = 0.20")
    print("="*70)
    
    # Setup logging
    log_file = setup_logging()
    
    n_trials = 30
    cv = 0.20
    
    try:
        # Generate figure
        csv_data, png_file, tiff_file, csv_file = create_figure_with_variability(
            n_trials=n_trials, cv=cv
        )
        
        # Generate summary
        generate_summary_statistics(csv_data)
        
        logging.info("\n" + "="*70)
        logging.info("FIGURE GENERATION COMPLETE")
        logging.info("="*70)
        logging.info(f"Files generated:")
        logging.info(f"- {png_file}")
        logging.info(f"- {tiff_file}")
        logging.info(f"- {csv_file}")
        logging.info(f"- {log_file}")
        logging.info(f"\nAll data based on {n_trials} trials with CV={cv}")
        logging.info("Variability based on Nusser et al., 2001")
        
        print(f"\nLog saved as: {log_file}")
        print("All files generated successfully!")
        
    except Exception as e:
        logging.error(f"Error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
