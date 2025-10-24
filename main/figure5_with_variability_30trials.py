"""
Figure 5 - PD Parameter Validation with Biological Variability
===============================================================
30 trials per condition with CV = 0.20
Implements frequency-dependent parameter switching at 20Hz threshold

BIOLOGICAL VARIABILITY REFERENCES:
- Nusser et al., 2001: mIPSC variability ~30%
- Smith et al., 2003: mEPSC amplitude CV = 0.69-0.75
- Implemented as CV = 0.20 (20%, conservative estimate)

Filename: figure5_with_variability_30trials.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from PIL import Image
import logging
from datetime import datetime

def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"figure5_variability_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*70)
    logging.info("FIGURE 5: PD VALIDATION WITH VARIABILITY")
    logging.info("30 trials per condition, CV = 0.20")
    logging.info("="*70)
    
    return log_filename

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 12,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
    "axes.linewidth": 1.2,
})

class TsodyksMarkramSynapse:
    """TM synapse with variability (Nusser et al., 2001)"""
    
    def __init__(self, tau_rec, tau_facil, U, name="synapse"):
        self.base_tau_rec = float(tau_rec)
        self.base_tau_facil = float(tau_facil)
        self.base_U = float(U)
        self.name = name
        self.reset()
    
    def reset(self):
        self.x = 1.0
        self.u = self.base_U
        self.last_time = 0.0
        self.tau_rec = self.base_tau_rec
        self.tau_facil = self.base_tau_facil
        self.U = self.base_U
    
    def set_trial_parameters(self, cv=0.20):
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

def get_pd_parameters_20hz_threshold(frequency):
    """Frequency-dependent PD parameters with 20Hz threshold"""
    if frequency < 20.0:
        return 460.0, 20.0, 0.32  # Low-medium freq
    else:
        return 184.0, 52.5, 0.2135  # High freq (≥20Hz)

def simulate_single_trial(frequency, n_pulses=10, cv=0.20):
    tau_rec, tau_facil, U = get_pd_parameters_20hz_threshold(frequency)
    synapse = TsodyksMarkramSynapse(tau_rec, tau_facil, U, f"PD_{frequency}Hz")
    synapse.reset()
    synapse.set_trial_parameters(cv=cv)
    
    isi = 1000.0 / frequency
    responses = []
    
    for i in range(n_pulses):
        time = i * isi
        response = synapse.stimulate(time)
        responses.append(response)
    
    responses = np.array(responses)
    if responses[0] > 0:
        normalized = 100.0 * responses / responses[0]
    else:
        normalized = np.full(n_pulses, 100.0)
    
    return normalized

def simulate_multiple_trials(frequency, n_pulses=10, n_trials=30, cv=0.20):
    all_trials = []
    for trial in range(n_trials):
        trial_response = simulate_single_trial(frequency, n_pulses, cv)
        all_trials.append(trial_response)
    
    all_trials = np.array(all_trials)
    mean_response = np.mean(all_trials, axis=0)
    sem_response = stats.sem(all_trials, axis=0)
    
    return mean_response, sem_response

def create_figure5(n_trials=30, cv=0.20):
    logging.info(f"Creating Figure 5 ({n_trials} trials, CV={cv})...")
    
    frequencies = [1, 5, 10, 15, 20, 25, 30, 40]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    csv_data = []
    
    for idx, freq in enumerate(frequencies):
        logging.info(f"Processing {freq} Hz...")
        ax = axes[idx]
        
        mean_y, sem_y = simulate_multiple_trials(freq, n_pulses=10, n_trials=n_trials, cv=cv)
        pulse_numbers = np.arange(1, len(mean_y) + 1)
        
        tau_rec, tau_facil, U = get_pd_parameters_20hz_threshold(freq)
        
        ax.errorbar(pulse_numbers, mean_y, yerr=sem_y,
                   fmt='o-', linewidth=2, markersize=6,
                   color='#e41a1c', capsize=4, capthick=1.5)
        
        ax.set_title(f'{freq} Hz', fontweight='bold')
        ax.set_xlabel('Pulse number')
        if idx % 4 == 0:
            ax.set_ylabel('Normalized fEPSP (%)')
        ax.set_ylim(20, 160)
        ax.grid(alpha=0.3)
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
        
        param_text = f'τ_rec={tau_rec:.0f}ms\nτ_fac={tau_facil:.1f}ms\nU={U:.3f}'
        ax.text(0.98, 0.02, param_text, transform=ax.transAxes,
               fontsize=8, verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        for pulse_num, mean_val, sem_val in zip(pulse_numbers, mean_y, sem_y):
            csv_data.append({
                'frequency_hz': freq,
                'pulse_number': pulse_num,
                'response_mean_percent': mean_val,
                'response_sem_percent': sem_val,
                'tau_rec': tau_rec,
                'tau_facil': tau_facil,
                'U': U,
                'n_trials': n_trials,
                'cv': cv
            })
    
    plt.suptitle(f'Figure 5: PD Pathway Validation (20Hz Threshold)\n'
                f'Mean ± SEM (n={n_trials} trials, CV={cv})',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_file = f"Figure5_PD_validation_variability_{timestamp}.png"
    tiff_file = f"Figure5_PD_validation_variability_{timestamp}.tiff"
    
    fig.savefig(png_file, dpi=600, bbox_inches="tight")
    img = Image.open(png_file)
    img.save(tiff_file, dpi=(600, 600), compression="tiff_lzw")
    
    plt.close(fig)
    
    df = pd.DataFrame(csv_data)
    csv_file = f"Figure5_data_variability_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    
    logging.info(f"Saved: {png_file}, {tiff_file}, {csv_file}")
    
    return csv_data

def main():
    print("="*70)
    print("FIGURE 5: PD VALIDATION WITH VARIABILITY (30 trials, CV=0.20)")
    print("="*70)
    
    log_file = setup_logging()
    
    try:
        csv_data = create_figure5(n_trials=30, cv=0.20)
        
        logging.info("="*70)
        logging.info("COMPLETE")
        logging.info(f"References: Nusser et al., 2001; Smith et al., 2003")
        logging.info("="*70)
        
        print(f"Log: {log_file}")
        print("Success!")
        
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
