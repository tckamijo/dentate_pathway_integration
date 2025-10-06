"""
Reproduction of Hayakawa 2014 Fig.3 – publication ready
Generates corrected Fig.3 with PNG + compressed TIFF output (600 dpi).
"""
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# -------- Global figure style --------
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

# -------- Synapse Model (Tsodyks–Markram minimal) --------
class TMSynapse:
    def __init__(self, tau_rec, tau_facil, U, weight=1.0):
        self.tau_rec = float(tau_rec)
        self.tau_facil = float(tau_facil)
        self.U = float(U)
        self.weight = float(weight)
        self.reset()

    def reset(self):
        self.x = 1.0
        self.u = self.U

    def spike(self):
        rel = self.u * self.x
        self.x -= rel
        self.u += self.U * (1.0 - self.u)
        return self.weight * rel

    def evolve(self, dt):
        self.x += (1.0 - self.x) * (dt / self.tau_rec)
        self.u += (self.U - self.u) * (dt / self.tau_facil)
        self.x = min(max(self.x, 0.0), 1.0)
        self.u = max(self.u, self.U)

def simulate_train(tau_rec, tau_facil, U, freq_hz, n_pulses=10):
    """Simulate normalized response for pulse train (P1=100)."""
    syn = TMSynapse(tau_rec, tau_facil, U)
    isi = 1000.0 / float(freq_hz)  # ms
    dt = 0.1
    out = []
    syn.reset()
    for k in range(n_pulses):
        if k > 0:
            for _ in range(int(isi / dt)):
                syn.evolve(dt)
        out.append(syn.spike())
    if out[0] == 0:
        return [100.0] * n_pulses
    base = out[0]
    return [100.0 * v / base for v in out]

# -------- File save helper --------
def save_publication_figure(fig, basename="Fig3_Reproduction_Hayakawa2014", dpi=600):
    """Save figure as both PNG and compressed TIFF (LZW)."""
    png_file = f"{basename}.png"
    tiff_file = f"{basename}.tiff"

    # Save PNG
    fig.savefig(png_file, dpi=dpi, bbox_inches="tight", format="png")

    # Convert PNG → LZW-compressed TIFF
    img = Image.open(png_file)
    img.save(tiff_file, dpi=(dpi, dpi), compression="tiff_lzw")

    return png_file, tiff_file

# -------- Main Figure Generation --------
def create_corrected_figure():
    freqs = [0.1, 1, 2, 5, 10, 20]
    params = {
        "DD": (248.0, 133.0, 0.20),
        "MD": (3977.0, 27.0, 0.30),
        "PD": (460.2, 20.0, 0.318),
    }

    fig, axes = plt.subplots(1, len(freqs), figsize=(18, 4))
    if len(freqs) == 1:
        axes = [axes]

    for i, f in enumerate(freqs):
        ax = axes[i]
        for name, (tau_rec, tau_facil, U) in params.items():
            y = simulate_train(tau_rec, tau_facil, U, f, n_pulses=10)
            ax.plot(range(1, len(y) + 1), y, 'o-', lw=2, ms=5, label=name)
        ax.set_title(f"{f} Hz", fontweight="bold")
        ax.set_xlabel("Pulse number")
        if i == 0:
            ax.set_ylabel("Normalized fEPSP [%]")
        ax.set_ylim(20, 140)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc="lower left", frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.suptitle("Reproduction of Hayakawa 2014 Fig.3\nCorrected pathways (DD, MD, PD)",
                 fontsize=16, fontweight="bold", y=0.99)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    basename = f"Figure3_Hayakawa2014_Reproduction_{timestamp}"
    png_file, tiff_file = save_publication_figure(fig, basename)
    print(f"[saved] {png_file}, {tiff_file}")

# -------- Run --------
if __name__ == "__main__":
    print("=== RUNNING HAYAKAWA FIG.3 REPRODUCTION (PUBREADY) ===")
    create_corrected_figure()

