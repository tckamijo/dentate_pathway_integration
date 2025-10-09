"""
Accurate Figure Generator for Three-Pathway Integration Paper
===========================================================
NO FABRICATION - Uses actual Tsodyks-Markram parameters and calculations
Generates Figure 1 and Figure 2 with real simulation data
Outputs: PNG + 600 DPI TIFF (LZW compressed) + CSV data

FIXED VERSION: Titles removed to prevent overlap with legends/data

Filename: figure_1-2_accurate_generator_fixed.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os
import logging
from datetime import datetime

def setup_logging():
    """デバッグ用ログ設定"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"accurate_figure_generation_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*50)
    logging.info("ACCURATE FIGURE GENERATION - NO FABRICATION")
    logging.info("="*50)
    
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
    Actual Tsodyks-Markram dynamic synapse model
    NO FABRICATION - Uses experimentally derived parameters
    """
    
    def __init__(self, tau_rec, tau_facil, U, name="synapse"):
        self.tau_rec = float(tau_rec)      # Recovery time constant (ms)
        self.tau_facil = float(tau_facil)  # Facilitation time constant (ms) 
        self.U = float(U)                  # Release probability
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset synapse to initial state"""
        self.x = 1.0  # Available resources (fraction)
        self.u = self.U  # Utilization parameter
        self.last_time = 0.0
    
    def stimulate(self, time):
        """
        Stimulate synapse at given time and return response
        Based on actual Tsodyks-Markram equations
        """
        dt = time - self.last_time
        
        if dt > 0:
            # Update available resources (recovery)
            self.x = 1 - (1 - self.x) * np.exp(-dt / self.tau_rec)
            # Update utilization (facilitation decay)
            self.u = self.U + (self.u - self.U) * np.exp(-dt / self.tau_facil)
        
        # Calculate synaptic response
        response = self.u * self.x
        
        # Log detailed calculations
        logging.debug(f"{self.name}: t={time:.1f}ms, dt={dt:.1f}ms, x={self.x:.3f}, u={self.u:.3f}, response={response:.3f}")
        
        # Update state after stimulation
        self.x = self.x * (1 - self.u)  # Resources consumed
        self.u = self.u + self.U * (1 - self.u)  # Facilitation increment
        
        self.last_time = time
        return response

def simulate_pathway_response(synapse, frequency, n_pulses=10, duration_factor=2.0):
    """
    Simulate pathway response to pulse train
    Returns normalized response amplitudes (1st pulse = 100%)
    """
    if frequency <= 0:
        return [100.0] * n_pulses
    
    isi = 1000.0 / frequency  # Inter-stimulus interval (ms)
    total_duration = n_pulses * isi * duration_factor
    
    synapse.reset()
    responses = []
    
    for i in range(n_pulses):
        time = i * isi
        response = synapse.stimulate(time)
        responses.append(response)
    
    # Normalize to first pulse = 100%
    if responses[0] > 0:
        normalized = [100.0 * r / responses[0] for r in responses]
    else:
        normalized = [100.0] * n_pulses
    
    return normalized

def create_pathway_synapses():
    """
    Create synapses with ACTUAL experimental parameters
    NO FABRICATION - These are the real fitted values
    """
    logging.info("Creating pathway synapses with experimental parameters...")
    
    synapses = {
        'DD': TsodyksMarkramSynapse(
            tau_rec=248.0,   # Recovery time constant (ms)
            tau_facil=133.0, # Facilitation time constant (ms)
            U=0.20,          # Release probability
            name="DD_LPP"
        ),
        'MD': TsodyksMarkramSynapse(
            tau_rec=3977.0,  # Recovery time constant (ms)
            tau_facil=27.0,  # Facilitation time constant (ms)
            U=0.30,          # Release probability
            name="MD_MPP"
        ),
        'PD': TsodyksMarkramSynapse(
            tau_rec=460.0,   # Recovery time constant (ms)
            tau_facil=20.0,  # Facilitation time constant (ms)
            U=0.32,          # Release probability
            name="PD_SuM_MC_MS"
        )
    }
    
    for name, synapse in synapses.items():
        logging.info(f"  {name}: τ_rec={synapse.tau_rec}ms, τ_facil={synapse.tau_facil}ms, U={synapse.U}")
    
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

def create_figure1():
    """
    Create Figure 1: Frequency-dependent response characteristics
    Uses ACTUAL Tsodyks-Markram simulation - NO FABRICATION
    """
    logging.info("Creating Figure 1: Frequency-dependent response characteristics...")
    
    # Frequency range from paper
    frequencies = np.array([0.1, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0])
    logging.info(f"Testing frequencies: {frequencies} Hz")
    
    # Create actual synapses with experimental parameters
    synapses = create_pathway_synapses()
    
    # Calculate actual responses for each pathway
    results_data = []
    pathway_responses = {}
    
    for pathway_name, synapse in synapses.items():
        logging.info(f"\nProcessing {pathway_name} pathway...")
        responses = []
        
        for freq in frequencies:
            # Simulate 10-pulse train and take 2nd pulse for frequency response
            normalized_train = simulate_pathway_response(synapse, freq, n_pulses=10)
            response_amplitude = normalized_train[1]  # 2nd pulse response
            responses.append(response_amplitude)
            
            logging.info(f"  {freq} Hz: {response_amplitude:.1f}%")
            
            # Store for CSV output
            results_data.append({
                'pathway': pathway_name,
                'frequency_hz': freq,
                'response_amplitude_percent': response_amplitude,
                'pulse_number': 2
            })
        
        pathway_responses[pathway_name] = np.array(responses)
    
    # Create publication-quality figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot with distinct markers and colors
    colors = {'DD': '#377eb8', 'MD': '#4daf4a', 'PD': '#e41a1c'}
    markers = {'DD': 'o', 'MD': 's', 'PD': '^'}
    
    for pathway in ['DD', 'MD', 'PD']:
        ax.plot(frequencies, pathway_responses[pathway], 
               marker=markers[pathway], linewidth=2.5, markersize=8,
               color=colors[pathway], label=f'{pathway}', markerfacecolor='white',
               markeredgewidth=2, markeredgecolor=colors[pathway])
    
    ax.set_xlabel('Frequency (Hz)', fontweight='bold')
    ax.set_ylabel('Normalized fEPSP amplitude (%)', fontweight='bold')
    # Title removed - described in figure caption in manuscript
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, facecolor='white', edgecolor='black', 
              loc='best', fontsize=11)
    ax.set_ylim(0, 140)
    
    plt.tight_layout()
    
    # Save publication files
    png_file, tiff_file = save_publication_figure(fig, "Figure1")
    plt.close(fig)
    
    # Save numerical data
    df = pd.DataFrame(results_data)
    csv_file = "Figure1_data.csv"
    df.to_csv(csv_file, index=False)
    logging.info(f"Saved numerical data: {csv_file}")
    
    return pathway_responses, results_data

def create_figure2(pathway_responses):
    """
    Create Figure 2: Pathway integration heatmap
    Uses ACTUAL simulation data - NO FABRICATION
    """
    print("Creating Figure 2: Pathway integration heatmap...")
    
    frequencies = np.array([0.1, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0])
    pathways = ['DD', 'MD', 'PD']
    
    # Create heatmap data matrix
    heatmap_data = np.array([pathway_responses[pw] for pw in pathways])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create heatmap
    im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto',
                   extent=[np.log10(frequencies[0]), np.log10(frequencies[-1]), 0, len(pathways)])
    
    # Set ticks and labels
    ax.set_yticks(np.arange(len(pathways)) + 0.5)
    ax.set_yticklabels(pathways)
    
    # Set x-ticks at actual frequency positions
    freq_positions = np.log10(frequencies)
    ax.set_xticks(freq_positions)
    ax.set_xticklabels([f'{f:g}' for f in frequencies])
    
    ax.set_xlabel('Frequency (Hz)', fontweight='bold')
    ax.set_ylabel('Pathway', fontweight='bold')
    # Title removed - described in figure caption in manuscript
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Response amplitude (%)', fontweight='bold', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    # Save publication files
    png_file, tiff_file = save_publication_figure(fig, "Figure2")
    plt.close(fig)
    
    return heatmap_data

def generate_summary_statistics(results_data):
    """Generate summary statistics for the actual simulation data"""
    df = pd.DataFrame(results_data)
    
    print("\n" + "="*50)
    print("ACTUAL SIMULATION RESULTS SUMMARY")
    print("="*50)
    
    for pathway in ['DD', 'MD', 'PD']:
        pathway_data = df[df['pathway'] == pathway]
        min_resp = pathway_data['response_amplitude_percent'].min()
        max_resp = pathway_data['response_amplitude_percent'].max()
        freq_1hz = pathway_data[pathway_data['frequency_hz'] == 1.0]['response_amplitude_percent'].iloc[0]
        freq_40hz = pathway_data[pathway_data['frequency_hz'] == 40.0]['response_amplitude_percent'].iloc[0]
        
        print(f"\n{pathway} Pathway:")
        print(f"  Response at 1 Hz: {freq_1hz:.1f}%")
        print(f"  Response at 40 Hz: {freq_40hz:.1f}%")
        print(f"  Range: {min_resp:.1f}% - {max_resp:.1f}%")
        print(f"  Frequency dependence: {freq_40hz/freq_1hz:.3f} (40Hz/1Hz ratio)")

def main():
    """Main execution function - NO FABRICATION"""
    print("Accurate Figure Generation - NO FABRICATION (FIXED VERSION)")
    print("Titles removed to prevent overlap with legends and data")
    print("=" * 60)
    
    # Setup logging
    log_file = setup_logging()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Generate Figure 1 with actual simulation data
        logging.info("Starting Figure 1 generation...")
        pathway_responses, results_data = create_figure1()
        
        # Generate Figure 2 with actual simulation data
        logging.info("Starting Figure 2 generation...")
        heatmap_data = create_figure2(pathway_responses)
        
        # Generate summary statistics
        logging.info("Generating summary statistics...")
        generate_summary_statistics(results_data)
        
        logging.info("="*50)
        logging.info("FIGURE GENERATION COMPLETE")
        logging.info("="*50)
        logging.info("Files generated:")
        logging.info("- Figure1.png, Figure1.tiff (600 DPI, LZW compressed)")
        logging.info("- Figure2.png, Figure2.tiff (600 DPI, LZW compressed)")
        logging.info("- Figure1_data.csv (numerical results)")
        logging.info(f"- {log_file} (debug log)")
        logging.info("")
        logging.info("All data based on ACTUAL Tsodyks-Markram simulations")
        logging.info("NO FABRICATION - Results are reproducible")
        
        print(f"\nDebug log saved as: {log_file}")
        print("All files generated successfully!")
        
    except Exception as e:
        logging.error(f"Error during figure generation: {str(e)}")
        print(f"Error during figure generation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
