import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

class TsodyksMarkramSynapse:
    """Tsodyks-Markram動的シナプスモデル"""
    
    def __init__(self, tau_rec, tau_facil, U, name="synapse"):
        self.tau_rec = tau_rec
        self.tau_facil = tau_facil
        self.U = U
        self.name = name
        self.reset()
    
    def reset(self):
        self.x = 1.0
        self.u = self.U
    
    def stimulate(self, dt_since_last=None):
        if dt_since_last is not None and dt_since_last > 0:
            self.x = 1 - (1 - self.x) * np.exp(-dt_since_last / self.tau_rec)
            self.u = self.U + (self.u - self.U) * np.exp(-dt_since_last / self.tau_facil)
        
        response = self.u * self.x
        self.x = self.x * (1 - self.u)
        self.u = self.U + self.u * (1 - self.U)
        
        return response

def test_threshold_calibration():
    """閾値キャリブレーションテスト"""
    print("=== Threshold Calibration Test ===")
    
    # Sample integrated responses from different conditions
    test_responses = {
        'First_pulse': 0.92,     # Always fires
        'Second_pulse_short': 0.24,  # ISI=5ms
        'Second_pulse_long': 0.37,   # ISI=20ms
        'Third_pulse': 0.18          # Weakest
    }
    
    thresholds = [0.20, 0.25, 0.30, 0.35]
    
    for threshold in thresholds:
        spikes = []
        for pulse_type, response in test_responses.items():
            spike = response >= threshold
            spikes.append(spike)
            print(f"  Threshold {threshold:.2f}: {pulse_type} ({response:.2f}) → {'SPIKE' if spike else 'no spike'}")
        
        total_spikes = sum(spikes)
        print(f"    Total spikes: {total_spikes}/4 → I/O ratio: {total_spikes/4:.3f}")
        print()

def simulate_burst_calibrated(synapse_params, weights, n_pulses, isi_ms, spike_threshold=0.30):
    """キャリブレーション版バースト刺激シミュレーション"""
    
    # シナプス初期化
    synapses = {}
    for pathway, params in synapse_params.items():
        synapses[pathway] = TsodyksMarkramSynapse(**params, name=pathway)
    
    # シナプスリセット
    for synapse in synapses.values():
        synapse.reset()
    
    spike_count = 0
    responses = []
    
    for pulse_idx in range(n_pulses):
        dt = isi_ms if pulse_idx > 0 else None
        
        # 各経路の応答
        pathway_responses = {}
        for pathway, synapse in synapses.items():
            pathway_responses[pathway] = synapse.stimulate(dt)
        
        # 重み付き統合
        integrated_response = sum(weights[pathway] * pathway_responses[pathway] 
                                for pathway in weights.keys() if pathway in pathway_responses)
        
        responses.append(integrated_response)
        
        # 発火判定
        if integrated_response >= spike_threshold:
            spike_count += 1
    
    io_ratio = spike_count / n_pulses if n_pulses > 0 else 0
    max_response = max(responses) if responses else 0
    
    return io_ratio, max_response, responses

def run_calibrated_simulation():
    """キャリブレーション版シミュレーション"""
    print("Starting calibrated Hayakawa Fig5 reproduction")
    
    # 閾値キャリブレーション
    test_threshold_calibration()
    
    # 最適閾値選択（診断結果に基づく）
    optimal_threshold = 0.30  # 2-3番目のパルスの中間値
    print(f"Selected optimal threshold: {optimal_threshold}")
    
    # シナプスパラメータ
    synapse_params = {
        'DD': {'tau_rec': 50, 'tau_facil': 300, 'U': 0.90},
        'MD': {'tau_rec': 57, 'tau_facil': 192, 'U': 0.90},
        'PD': {'tau_rec': 50, 'tau_facil': 77, 'U': 0.92}
    }
    
    # 実験条件
    conditions = {
        'DD_only': {'DD': 1.0, 'MD': 0.0, 'PD': 0.0},
        'MD_only': {'DD': 0.0, 'MD': 1.0, 'PD': 0.0},
        'PD_only': {'DD': 0.0, 'MD': 0.0, 'PD': 1.0},
        'MD+PD': {'DD': 0.0, 'MD': 0.5, 'PD': 0.5},
        'SuM_dominant': {'DD': 0.17, 'MD': 0.17, 'PD': 0.66},
        'Traditional': {'DD': 0.5, 'MD': 0.5, 'PD': 0.0}
    }
    
    # パラメータ範囲
    n_range = list(range(1, 6))  # 1-5 pulses
    isi_range = list(range(5, 27, 2))  # 5-25 ms
    
    results = {}
    
    # 詳細テスト（代表的条件）
    print("\nDetailed testing for key conditions:")
    test_conditions = ['PD_only', 'SuM_dominant', 'Traditional']
    for cond in test_conditions:
        io, max_resp, resp = simulate_burst_calibrated(
            synapse_params, conditions[cond], 3, 10, optimal_threshold
        )
        print(f"  {cond}: I/O={io:.3f}, responses={[f'{r:.3f}' for r in resp]}")
    
    # 全条件シミュレーション
    print("\nRunning full simulation...")
    for condition_name, weights in conditions.items():
        print(f"Processing {condition_name}...")
        
        io_matrix = np.zeros((len(n_range), len(isi_range)))
        max_matrix = np.zeros((len(n_range), len(isi_range)))
        
        for i, n in enumerate(n_range):
            for j, isi in enumerate(isi_range):
                io_ratio, max_resp, _ = simulate_burst_calibrated(
                    synapse_params, weights, n, isi, optimal_threshold
                )
                io_matrix[i, j] = io_ratio
                max_matrix[i, j] = max_resp
        
        # 結果要約
        avg_io = np.mean(io_matrix)
        isi_effect = np.max(io_matrix[2, :]) - np.min(io_matrix[2, :])  # n=3でのISI効果
        
        print(f"  Average I/O: {avg_io:.3f}")
        print(f"  ISI effect (n=3): {isi_effect:.3f}")
        
        results[condition_name] = {
            'io_ratio': io_matrix,
            'max_response': max_matrix
        }
    
    return results, n_range, isi_range, optimal_threshold

def create_calibrated_plots(results, n_range, isi_range, threshold):
    """キャリブレーション版プロット"""
    
    save_dir = "figures"
    os.makedirs(save_dir, exist_ok=True)
    
    conditions = list(results.keys())
    
    # 1. ヒートマップ
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Hayakawa Fig5 Reproduction (Threshold: {threshold})', fontsize=16)
    axes = axes.flatten()
    
    for i, condition in enumerate(conditions):
        if i < len(axes):
            data = results[condition]['io_ratio']
            
            im = axes[i].imshow(data, aspect='auto', cmap='viridis', 
                              vmin=0, vmax=1,
                              extent=[min(isi_range), max(isi_range), 
                                     max(n_range), min(n_range)])
            
            axes[i].set_title(f'{condition}')
            axes[i].set_xlabel('ISI (ms)')
            axes[i].set_ylabel('Burst size (N)')
            
            # データ範囲とカラーバー
            data_min, data_max = data.min(), data.max()
            cbar = plt.colorbar(im, ax=axes[i])
            cbar.set_label('I/O ratio')
            
            axes[i].text(0.02, 0.98, f'{data_min:.2f}-{data_max:.2f}', 
                        transform=axes[i].transAxes, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        verticalalignment='top')
    
    # 余分な軸を非表示
    for i in range(len(conditions), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename1 = f"{save_dir}/hayakawa_calibrated_{timestamp}.png"
    plt.savefig(filename1, dpi=300, bbox_inches='tight')
    print(f"Calibrated heatmaps saved: {filename1}")
    plt.close()
    
    # 2. ISI依存性比較（複数のnで比較）
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    n_values_to_plot = [1, 2, 3, 4]
    
    for idx, n_val in enumerate(n_values_to_plot):
        n_idx = n_val - 1
        
        for condition, data in results.items():
            n_data = data['io_ratio'][n_idx, :]
            axes[idx].plot(isi_range, n_data, 'o-', label=condition, linewidth=2, markersize=4)
        
        axes[idx].set_xlabel('ISI (ms)')
        axes[idx].set_ylabel('I/O ratio')
        axes[idx].set_title(f'N = {n_val} pulses')
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim(0, 1)
    
    plt.tight_layout()
    
    filename2 = f"{save_dir}/hayakawa_isi_detailed_{timestamp}.png"
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    print(f"Detailed ISI plots saved: {filename2}")
    plt.close()
    
    return filename1, filename2

def analyze_calibrated_results(results, n_range, isi_range):
    """キャリブレーション結果解析"""
    print("\n=== Calibrated Results Analysis ===")
    
    # 単一経路比較
    print("Single pathway comparison (n=3, ISI=10ms):")
    single_pathways = ['DD_only', 'MD_only', 'PD_only']
    for pathway in single_pathways:
        if pathway in results:
            io_val = results[pathway]['io_ratio'][2, 2]  # n=3, ISI=10ms
            print(f"  {pathway}: {io_val:.3f}")
    
    # SuM vs Traditional - 詳細比較
    print("\nSuM vs Traditional detailed comparison:")
    if 'SuM_dominant' in results and 'Traditional' in results:
        sum_data = results['SuM_dominant']['io_ratio']
        trad_data = results['Traditional']['io_ratio']
        
        sum_avg = np.mean(sum_data)
        trad_avg = np.mean(trad_data)
        advantage = ((sum_avg - trad_avg) / trad_avg) * 100
        
        print(f"  SuM_dominant average: {sum_avg:.3f}")
        print(f"  Traditional average: {trad_avg:.3f}")
        print(f"  SuM advantage: {advantage:+.1f}%")
        
        # 条件別優位性
        print("  Condition-specific advantages:")
        for i, n in enumerate(n_range):
            sum_n = np.mean(sum_data[i, :])
            trad_n = np.mean(trad_data[i, :])
            adv_n = ((sum_n - trad_n) / trad_n) * 100 if trad_n > 0 else 0
            print(f"    N={n}: SuM={sum_n:.3f}, Trad={trad_n:.3f}, Advantage={adv_n:+.1f}%")

if __name__ == "__main__":
    # キャリブレーション版実行
    results, n_range, isi_range, threshold = run_calibrated_simulation()
    
    # プロット作成
    create_calibrated_plots(results, n_range, isi_range, threshold)
    
    # 結果解析
    analyze_calibrated_results(results, n_range, isi_range)
    
    print("\n" + "="*60)
    print("CALIBRATED HAYAKAWA FIG5 REPRODUCTION COMPLETED!")
    print("="*60)
    print("Check 'figures' directory for calibrated results.")
