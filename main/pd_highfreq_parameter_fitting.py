"""
PD high-frequency parameter fitting – publication-ready rendering
Outputs both PNG and compressed TIFF (600 dpi, LZW).
"""

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

# -------- Tsodyks–Markram short-term plasticity --------
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

# -------- Toy experimental PD trend --------
def pd_experimental_trend():
    return {
        0.1: [100, 95, 93, 92, 91, 90, 89, 88, 87, 86],
        1.0: [100, 88, 85, 83, 82, 81, 80, 79, 78, 77],
        2.0: [100, 85, 82, 80, 78, 77, 76, 75, 74, 73],
        5.0: [100, 80, 75, 72, 70, 68, 67, 66, 65, 64],
        10.0: [100, 75, 65, 58, 55, 52, 50, 48, 47, 46],
        20.0: [100, 70, 55, 45, 40, 38, 36, 35, 34, 33],
        30.0: [100, 65, 50, 40, 35, 32, 30, 29, 28, 27],
        40.0: [100, 60, 45, 35, 30, 28, 26, 25, 24, 23],
    }

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred[:len(y_true)], float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
    return 1.0 - ss_res / ss_tot

# -------- Save helper --------
def save_publication_figure(fig, basename, dpi=600):
    png_file = f"{basename}.png"
    tiff_file = f"{basename}.tiff"

    fig.savefig(png_file, dpi=dpi, format="png", bbox_inches="tight")
    img = Image.open(png_file)
    img.save(tiff_file, dpi=(dpi, dpi), compression="tiff_lzw")

    plt.close(fig)
    print(f"[saved] {png_file}, {tiff_file}")

# -------- Figure A --------
def figure_pd_validation(pd_tau_rec=460.2, pd_tau_facil=20.0, pd_U=0.318):
    data = pd_experimental_trend()
    freqs = [0.1, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, f in enumerate(freqs):
        ax = axes[i]
        exp = data[f]
        sim = simulate_train(pd_tau_rec, pd_tau_facil, pd_U, f)
        pulses = np.arange(1, len(exp) + 1)

        ax.plot(pulses, exp, 'o-', lw=2, color='red', label='Experimental')
        ax.plot(pulses, sim[:len(exp)], 's-', lw=2, color='blue', label='Fitted Model')
        ax.set_title(f"{f} Hz", fontweight="bold")
        ax.set_xlabel("Pulse number")
        ax.set_ylabel("Normalized response [%]")
        ax.set_ylim(20, 140)
        ax.grid(alpha=0.3)
        if i in (0, 4):
            ax.legend(loc="lower left", frameon=False)
        ax.text(0.98, 0.08, f"R²={r2_score(exp, sim):.3f}",
                transform=ax.transAxes, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.9), fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle("PD Parameter Fitting Validation\n(Red: Experimental, Blue: Fitted Model)",
                 fontsize=16, fontweight="bold", y=0.99)
    save_publication_figure(fig, "pd_parameter_fitting_validation")

# -------- Figure B --------
def figure_pathway_comparison(pd_tau_rec=460.2, pd_tau_facil=20.0, pd_U=0.318):
    pathways = {
        "DD (LPP)": dict(tau_rec=248.0,  tau_facil=133.0, U=0.20, color="#377eb8"),
        "MD (MPP)": dict(tau_rec=3977.0, tau_facil=27.0,  U=0.30, color="#ff7f0e"),
        "PD (fitted)": dict(tau_rec=pd_tau_rec, tau_facil=pd_tau_facil, U=pd_U, color="purple"),
    }
    freqs = [0.1, 1.0, 5.0, 10.0, 20.0]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i, f in enumerate(freqs):
        ax = axes[i]
        for name, p in pathways.items():
            y = simulate_train(p["tau_rec"], p["tau_facil"], p["U"], f)
            ax.plot(range(1, len(y) + 1), y, 'o-', lw=2, ms=4, color=p["color"], label=name)
        ax.set_title(f"{f} Hz", fontweight="bold")
        ax.set_xlabel("Pulse number")
        if i == 0:
            ax.set_ylabel("Normalized response [%]")
        ax.set_ylim(20, 140)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc="lower left", frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.suptitle("Comparison of DD, MD, and PD Pathway Dynamics",
                 fontsize=16, fontweight="bold", y=0.98)
    save_publication_figure(fig, "pathway_comparison")

# -------- Main --------
if __name__ == "__main__":
    PD_TAU_REC = 460.2
    PD_TAU_FAC = 20.0
    PD_U       = 0.318

    figure_pd_validation(PD_TAU_REC, PD_TAU_FAC, PD_U)
    figure_pathway_comparison(PD_TAU_REC, PD_TAU_FAC, PD_U)

