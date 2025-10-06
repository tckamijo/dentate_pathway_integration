from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Global style settings
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2

class DentateGranuleCellModel:
    def __init__(self, threshold=0.3):
        self.threshold = threshold

    def simulate_response(self, freq, pathway="DD"):
        if pathway == "DD":
            return np.exp(-freq / 50.0) * 0.8
        elif pathway == "MD":
            return np.exp(-freq / 40.0) * 0.6
        elif pathway == "PD":
            return np.exp(-freq / 30.0) * 0.9
        return 0.0

def save_tiff_compressed(fig, filename, dpi=600):
    """Save matplotlib figure as compressed TIFF (LZW)."""
    tmp_png = filename.replace(".tiff", ".png")
    fig.savefig(tmp_png, dpi=dpi, format="png", bbox_inches="tight")
    plt.close(fig)

    # Reopen and save as LZW-compressed TIFF
    img = Image.open(tmp_png)
    img.save(filename, dpi=(dpi, dpi), compression="tiff_lzw")

def create_figure1(frequencies, model):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(frequencies, [model.simulate_response(f, "DD") for f in frequencies],
            'o-', lw=2, color='#377eb8', label="DD")
    ax.plot(frequencies, [model.simulate_response(f, "MD") for f in frequencies],
            's-', lw=2, color='#4daf4a', label="MD")
    ax.plot(frequencies, [model.simulate_response(f, "PD") for f in frequencies],
            '^-', lw=2, color='#e41a1c', label="PD")

    ax.set_xlabel("Frequency (Hz)", fontsize=14)
    ax.set_ylabel("Normalized fEPSP amplitude", fontsize=14)
    ax.set_title("Figure 1. Frequency-dependent responses", fontsize=14, pad=12)
    ax.legend(fontsize=12, frameon=False)
    fig.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_tiff_compressed(fig, f"Figure1_Threshold_Summary_{timestamp}.tiff")
    print(f"✓ Saved: Figure1_Threshold_Summary_{timestamp}.tiff")

def create_figure2(frequencies, model):
    fig, ax = plt.subplots(figsize=(6, 4))
    heatmap_data = np.array([[model.simulate_response(f, pw)
                               for f in frequencies]
                               for pw in ["DD", "MD", "PD"]])

    im = ax.imshow(heatmap_data, aspect="auto", cmap="viridis",
                   extent=[min(frequencies), max(frequencies), 0, 3])
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(["DD", "MD", "PD"])
    ax.set_xlabel("Frequency (Hz)", fontsize=14)
    ax.set_title("Figure 2. Pathway integration (heatmap)", fontsize=14, pad=12)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Response strength", fontsize=12)
    fig.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_tiff_compressed(fig, f"Figure2_Threshold_Detail_{timestamp}.tiff")
    print(f"✓ Saved: Figure2_Threshold_Detail_{timestamp}.tiff")

def main():
    print("=== RUNNING THRESHOLD-ONLY FIGURE GENERATION (COMPRESSED TIFF OUTPUT) ===")
    frequencies = np.array([1, 2, 5, 10, 20, 40])
    model = DentateGranuleCellModel(threshold=0.3)

    create_figure1(frequencies, model)
    create_figure2(frequencies, model)
    print("Figures saved as LZW-compressed TIFF at 600 dpi")

if __name__ == "__main__":
    main()

