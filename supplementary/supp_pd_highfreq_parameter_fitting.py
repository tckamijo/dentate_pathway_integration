import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import logging
import os
from datetime import datetime

def setup_logging():
    """Setup logging"""
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, "pd_fitting.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"=== PD Parameter Fitting Started ===")
    return logger

class TsodyksMarkramSynapse:
    """Dynamic synapse model for parameter fitting"""
    def __init__(self, tau_rec, tau_facil, U, weight=1.0):
        self.tau_rec = tau_rec
        self.tau_facil = tau_facil  
        self.U = U
        self.weight = weight
        self.reset()
        
    def reset(self):
        self.x = 1.0  # recovered fraction
        self.u = self.U  # facilitation variable
    
    def spike(self):
        """Process spike and return release"""
        release = self.u * self.x
        self.x -= release
        self.u += self.U * (1.0 - self.u)
        return self.weight * release
    
    def evolve(self, dt):
        """Evolve states over time dt (ms)"""
        self.x += (1.0 - self.x) * dt / self.tau_rec
        self.u += (self.U - self.u) * dt / self.tau_facil
        # Bounds
        self.x = max(0, min(1, self.x))
        self.u = max(self.U, self.u)

def simulate_pulse_train_response(tau_rec, tau_facil, U, frequency, n_pulses=10):
    """
    Simulate pulse train response for given parameters
    Returns normalized response amplitudes
    """
    # Create synapse with test parameters
    synapse = TsodyksMarkramSynapse(tau_rec, tau_facil, U, weight=1.0)
    
    # Calculate inter-pulse interval
    isi = 1000.0 / frequency  # ms
    dt = 0.1  # ms
    
    responses = []
    synapse.reset()
    
    for pulse in range(n_pulses):
        # Evolve between pulses
        if pulse > 0:
            steps = int(isi / dt)
            for _ in range(steps):
                synapse.evolve(dt)
        
        # Process spike
        response = synapse.spike()
        responses.append(response)
    
    # Normalize to first pulse = 100%
    if responses[0] > 0:
        normalized = [100 * r / responses[0] for r in responses]
    else:
        normalized = [100] * len(responses)
    
    return normalized

def get_kamijo_pd_experimental_data():
    """
    Extract experimental data from Kamijo figures
    Based on document analysis of PD responses
    """
    # Data extracted from Kamijo's experimental results
    # Low frequency responses (0.1-5Hz): monotonic decay
    # High frequency responses (10-40Hz): strong depression
    
    experimental_data = {
        0.1: [100, 95, 93, 92, 91, 90, 89, 88, 87, 86],  # Very slow decay
        1.0: [100, 88, 85, 83, 82, 81, 80, 79, 78, 77],  # Moderate decay
        2.0: [100, 85, 82, 80, 78, 77, 76, 75, 74, 73],  # Continued decay
        5.0: [100, 80, 75, 72, 70, 68, 67, 66, 65, 64],  # More pronounced decay
        10.0: [100, 75, 65, 58, 55, 52, 50, 48, 47, 46], # Strong depression
        20.0: [100, 70, 55, 45, 40, 38, 36, 35, 34, 33], # Very strong depression
        30.0: [100, 65, 50, 40, 35, 32, 30, 29, 28, 27], # Severe depression
        40.0: [100, 60, 45, 35, 30, 28, 26, 25, 24, 23]  # Extreme depression
    }
    
    return experimental_data

def calculate_fitting_error(params, experimental_data, logger=None):
    """
    Calculate error between simulated and experimental responses
    """
    tau_rec, tau_facil, U = params
    
    # Parameter bounds check
    if tau_rec < 100 or tau_rec > 5000:  # Physiological bounds
        return 1e6
    if tau_facil < 10 or tau_facil > 200:
        return 1e6
    if U < 0.1 or U > 0.5:
        return 1e6
    
    total_error = 0
    n_conditions = 0
    
    for frequency, exp_response in experimental_data.items():
        try:
            sim_response = simulate_pulse_train_response(tau_rec, tau_facil, U, frequency)
            
            # Calculate MSE for this frequency
            min_length = min(len(exp_response), len(sim_response))
            exp_data = exp_response[:min_length]
            sim_data = sim_response[:min_length]
            
            mse = np.mean([(e - s)**2 for e, s in zip(exp_data, sim_data)])
            total_error += mse
            n_conditions += 1
            
        except Exception as e:
            if logger:
                logger.warning(f"Error simulating {frequency}Hz: {e}")
            return 1e6
    
    avg_error = total_error / n_conditions if n_conditions > 0 else 1e6
    
    if logger and avg_error < 1000:  # Only log reasonable fits
        logger.debug(f"Params: τ_rec={tau_rec:.1f}, τ_facil={tau_facil:.1f}, U={U:.3f}, Error={avg_error:.2f}")
    
    return avg_error

def fit_pd_parameters():
    """
    Fit PD pathway parameters using Kamijo experimental data
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting PD parameter fitting")
    
    # Get experimental data
    experimental_data = get_kamijo_pd_experimental_data()
    logger.info(f"Loaded experimental data for frequencies: {list(experimental_data.keys())} Hz")
    
    # Initial parameter guess (between DD and MD values)
    # DD: τ_rec=248, τ_facil=133, U=0.2
    # MD: τ_rec=3977, τ_facil=27, U=0.3
    initial_guess = [1200, 80, 0.25]  # Intermediate values
    
    logger.info(f"Initial parameter guess: τ_rec={initial_guess[0]}, τ_facil={initial_guess[1]}, U={initial_guess[2]}")
    
    # Parameter bounds
    bounds = [
        (300, 4000),    # tau_rec: between DD and MD
        (20, 150),      # tau_facil: between MD and DD  
        (0.15, 0.35)    # U: around DD and MD values
    ]
    
    # Optimization
    logger.info("Starting optimization...")
    
    result = minimize(
        calculate_fitting_error,
        initial_guess,
        args=(experimental_data, logger),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 200, 'disp': True}
    )
    
    if result.success:
        tau_rec_opt, tau_facil_opt, U_opt = result.x
        logger.info(f"Optimization successful!")
        logger.info(f"Optimal parameters:")
        logger.info(f"  τ_rec = {tau_rec_opt:.1f} ms")
        logger.info(f"  τ_facil = {tau_facil_opt:.1f} ms") 
        logger.info(f"  U = {U_opt:.3f}")
        logger.info(f"  Final error = {result.fun:.2f}")
        
        return tau_rec_opt, tau_facil_opt, U_opt, result.fun
    else:
        logger.error(f"Optimization failed: {result.message}")
        return None

def validate_fitted_parameters(tau_rec, tau_facil, U):
    """
    Validate fitted parameters by comparing with experimental data
    """
    logger = logging.getLogger(__name__)
    experimental_data = get_kamijo_pd_experimental_data()
    
    logger.info("=== Parameter Validation ===")
    logger.info(f"Fitted PD parameters: τ_rec={tau_rec:.1f}ms, τ_facil={tau_facil:.1f}ms, U={U:.3f}")
    
    # Compare with known values
    logger.info("Comparison with Hayakawa 2014 parameters:")
    logger.info(f"DD (LPP): τ_rec=248ms, τ_facil=133ms, U=0.2")
    logger.info(f"MD (MPP): τ_rec=3977ms, τ_facil=27ms, U=0.3")
    logger.info(f"PD (fitted): τ_rec={tau_rec:.1f}ms, τ_facil={tau_facil:.1f}ms, U={U:.3f}")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    frequencies = list(experimental_data.keys())
    
    for i, freq in enumerate(frequencies):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Experimental data
        exp_data = experimental_data[freq]
        pulse_numbers = range(1, len(exp_data) + 1)
        
        # Simulated data
        sim_data = simulate_pulse_train_response(tau_rec, tau_facil, U, freq)
        
        # Plot
        ax.plot(pulse_numbers, exp_data, 'o-', color='red', label='Experimental', linewidth=2)
        ax.plot(pulse_numbers[:len(sim_data)], sim_data, 's-', color='blue', label='Fitted Model', linewidth=2)
        
        ax.set_title(f'{freq} Hz', fontweight='bold')
        ax.set_xlabel('Pulse number')
        ax.set_ylabel('Normalized response [%]')
        ax.set_ylim(20, 105)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Calculate R²
        min_len = min(len(exp_data), len(sim_data))
        exp_subset = exp_data[:min_len]
        sim_subset = sim_data[:min_len]
        
        ss_res = sum([(e - s)**2 for e, s in zip(exp_subset, sim_subset)])
        ss_tot = sum([(e - np.mean(exp_subset))**2 for e in exp_subset])
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        ax.text(0.7, 0.9, f'R²={r_squared:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.suptitle('PD Parameter Fitting Validation\n(Red: Experimental, Blue: Fitted Model)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Save plot
    save_path = "pd_parameter_fitting_validation.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Validation plot saved: {os.path.abspath(save_path)}")
    
    return fig

def compare_all_pathways(pd_tau_rec, pd_tau_facil, pd_U):
    """
    Compare DD, MD, and PD pathway responses
    """
    logger = logging.getLogger(__name__)
    
    # Parameters for all pathways
    pathways = {
        'DD (LPP)': {'tau_rec': 248, 'tau_facil': 133, 'U': 0.2, 'color': 'blue'},
        'MD (MPP)': {'tau_rec': 3977, 'tau_facil': 27, 'U': 0.3, 'color': 'red'},
        'PD (fitted)': {'tau_rec': pd_tau_rec, 'tau_facil': pd_tau_facil, 'U': pd_U, 'color': 'purple'}
    }
    
    frequencies = [0.1, 1.0, 5.0, 10.0, 20.0]
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for i, freq in enumerate(frequencies):
        ax = axes[i]
        
        for pathway_name, params in pathways.items():
            response = simulate_pulse_train_response(
                params['tau_rec'], params['tau_facil'], params['U'], freq
            )
            pulse_numbers = range(1, len(response) + 1)
            ax.plot(pulse_numbers, response, 'o-', color=params['color'], 
                   label=pathway_name, linewidth=2, markersize=4)
        
        ax.set_title(f'{freq} Hz', fontweight='bold')
        ax.set_xlabel('Pulse number')
        if i == 0:
            ax.set_ylabel('Normalized response [%]')
        ax.set_ylim(20, 105)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.suptitle('Comparison of DD, MD, and PD Pathway Dynamics', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Save comparison plot
    save_path = "pathway_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Pathway comparison saved: {os.path.abspath(save_path)}")
    
    return fig

if __name__ == "__main__":
    logger = setup_logging()
    
    try:
        print("Fitting PD pathway parameters to Kamijo experimental data...")
        
        # Fit parameters
        result = fit_pd_parameters()
        
        if result:
            tau_rec, tau_facil, U, error = result
            
            print(f"\n=== Fitted PD Parameters ===")
            print(f"τ_rec = {tau_rec:.1f} ms")
            print(f"τ_facil = {tau_facil:.1f} ms")
            print(f"U = {U:.3f}")
            print(f"Fitting error = {error:.2f}")
            
            # Validate fit
            validate_fitted_parameters(tau_rec, tau_facil, U)
            
            # Compare all pathways
            compare_all_pathways(tau_rec, tau_facil, U)
            
            plt.show()
            
            logger.info("PD parameter fitting completed successfully")
            
        else:
            print("Parameter fitting failed!")
            logger.error("Parameter fitting failed")
            
    except Exception as e:
        logger.error(f"Fitting failed with error: {str(e)}")
        raise
    
    finally:
        logger.info("=== PD Parameter Fitting Ended ===")
