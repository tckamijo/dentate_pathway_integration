import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
from datetime import datetime

# デバッグ用ロギング設定
def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"three_pathway_simulation_debug_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return log_filename

class TsodyksMarkramSynapse:
    """Tsodyks-Markram動的シナプスモデル"""
    
    def __init__(self, tau_rec, tau_facil, U, name="synapse"):
        self.tau_rec = tau_rec
        self.tau_facil = tau_facil
        self.U = U
        self.name = name
        self.reset()
        logging.info(f"Created {name}: tau_rec={tau_rec}, tau_facil={tau_facil}, U={U}")
    
    def reset(self):
        self.x = 1.0
        self.u = self.U
        self.last_spike_time = None
    
    def stimulate(self, spike_time):
        """スパイク時刻での刺激"""
        if self.last_spike_time is not None:
            dt = spike_time - self.last_spike_time
            if dt > 0:
                # 回復計算
                self.x = 1 - (1 - self.x) * np.exp(-dt / self.tau_rec)
                self.u = self.U + (self.u - self.U) * np.exp(-dt / self.tau_facil)
        
        # 応答計算
        response = self.u * self.x
        
        # シナプス状態更新
        self.x = self.x * (1 - self.u)
        self.u = self.U + self.u * (1 - self.U)
        self.last_spike_time = spike_time
        
        return response

class PhysiologicalSpikeGenerator:
    """生理学的スパイクパターン生成器"""
    
    @staticmethod
    def dd_random_input(duration_ms=1000, base_rate=10, seed=None):
        """DD経路: ランダム入力（ポアソン過程）"""
        if seed is not None:
            np.random.seed(seed)
        
        spike_times = []
        t = 0
        while t < duration_ms:
            isi = np.random.exponential(1000 / base_rate)
            t += isi
            if t < duration_ms:
                spike_times.append(t)
        
        logging.info(f"DD random input: {len(spike_times)} spikes, rate={base_rate}Hz")
        return sorted(spike_times)
    
    @staticmethod
    def md_theta_bursts(duration_ms=1000, theta_freq=6, seed=None):
        """MD経路: θリズムバースト"""
        if seed is not None:
            np.random.seed(seed)
        
        theta_period = 1000 / theta_freq
        spike_times = []
        burst_count = 0
        
        for cycle in range(int(duration_ms / theta_period)):
            burst_start = cycle * theta_period
            burst_size = np.random.randint(3, 6)  # 3-5スパイク
            burst_count += 1
            
            for spike in range(burst_size):
                spike_time = burst_start + spike * np.random.uniform(3, 5)
                if spike_time < duration_ms:
                    spike_times.append(spike_time)
        
        logging.info(f"MD theta bursts: {len(spike_times)} spikes, {burst_count} bursts, {theta_freq}Hz")
        return sorted(spike_times)
    
    @staticmethod
    def pd_behavioral_input(duration_ms=1000, exploration_state=True, seed=None):
        """PD経路: 行動状態依存的入力"""
        if seed is not None:
            np.random.seed(seed)
        
        spike_times = []
        
        if exploration_state:
            # 探索時: γバースト
            burst_interval = np.random.uniform(80, 120)
            t = 0
            burst_count = 0
            
            while t < duration_ms:
                burst_size = np.random.randint(2, 4)
                gamma_isi = np.random.uniform(25, 40)
                burst_count += 1
                
                for spike in range(burst_size):
                    spike_time = t + spike * gamma_isi
                    if spike_time < duration_ms:
                        spike_times.append(spike_time)
                
                t += burst_interval
            
            logging.info(f"PD exploration input: {len(spike_times)} spikes, {burst_count} gamma bursts")
        else:
            # 非探索時: 低頻度ランダム
            spike_times = PhysiologicalSpikeGenerator.dd_random_input(duration_ms, base_rate=3, seed=seed)
            logging.info(f"PD rest input: {len(spike_times)} spikes, low frequency")
        
        return sorted(spike_times)

class ThreePathwayIntegrationModel:
    """3経路統合モデル（実験パラメータ厳守版）"""
    
    def __init__(self, weights, domain="gamma"):
        self.weights = weights
        self.domain = domain
        self.create_synapses()
        
        logging.info(f"Created 3-pathway model: domain={domain}")
        logging.info(f"Weights: {weights}")
    
    def create_synapses(self):
        """実験パラメータでシナプス作成"""
        if self.domain == "gamma":
            # γ周波数域（実験パラメータ）
            self.synapses = {
                'DD': TsodyksMarkramSynapse(tau_rec=50, tau_facil=300, U=0.90, name="DD_gamma"),
                'MD': TsodyksMarkramSynapse(tau_rec=57, tau_facil=192, U=0.90, name="MD_gamma"),
                'PD': TsodyksMarkramSynapse(tau_rec=50, tau_facil=77, U=0.92, name="PD_gamma")
            }
            self.spike_threshold = 0.30
        else:  # theta
            # θ周波数域（実験パラメータ）
            self.synapses = {
                'DD': TsodyksMarkramSynapse(tau_rec=248, tau_facil=133, U=0.20, name="DD_theta"),
                'MD': TsodyksMarkramSynapse(tau_rec=3977, tau_facil=27, U=0.30, name="MD_theta"),
                'PD': TsodyksMarkramSynapse(tau_rec=460, tau_facil=20, U=0.32, name="PD_theta")
            }
            self.spike_threshold = 0.18
        
        logging.info(f"Spike threshold: {self.spike_threshold}")
    
    def simulate_integration(self, spike_trains, debug_interval=100):
        """3経路統合シミュレーション"""
        # 入力検証
        total_spikes = sum(len(spikes) for spikes in spike_trains.values())
        if total_spikes == 0:
            logging.warning("No input spikes detected!")
            return {
                'integrated_responses': [],
                'output_spikes': [],
                'io_ratio': 0.0,
                'total_inputs': 0,
                'total_outputs': 0
            }
        
        # 全スパイクを時系列でソート
        all_events = []
        for pathway, spikes in spike_trains.items():
            for spike_time in spikes:
                all_events.append((spike_time, pathway))
        
        all_events.sort()
        logging.info(f"Total events: {len(all_events)}")
        
        # 統合応答計算
        integrated_responses = []
        output_spikes = []
        event_count = 0
        
        for spike_time, pathway in all_events:
            event_count += 1
            
            try:
                # 該当経路のシナプス応答
                response = self.synapses[pathway].stimulate(spike_time)
                
                # 3経路の現在の重み付け統合
                pathway_responses = {}
                for pw in self.weights.keys():
                    if pw == pathway:
                        pathway_responses[pw] = response
                    else:
                        # 他の経路の現在状態を取得（最後の応答値を使用）
                        pathway_responses[pw] = 0  # シンプル化：刺激されていない経路は0
                
                total_response = sum(
                    self.weights[pw] * pathway_responses[pw]
                    for pw in self.weights.keys()
                )
                
                integrated_responses.append((spike_time, total_response, pathway))
                
                # 発火判定
                if total_response >= self.spike_threshold:
                    output_spikes.append(spike_time)
                
                # デバッグ出力
                if event_count % debug_interval == 0 or event_count <= 10:
                    logging.info(f"Event {event_count}: t={spike_time:.1f}ms, {pathway}, "
                               f"response={response:.3f}, total={total_response:.3f}, "
                               f"spike={'YES' if total_response >= self.spike_threshold else 'NO'}")
            
            except Exception as e:
                logging.error(f"Error processing event {event_count}: {e}")
                continue
        
        # 結果サマリー
        input_count = len(all_events)
        output_count = len(output_spikes)
        io_ratio = output_count / input_count if input_count > 0 else 0.0
        
        logging.info(f"Integration summary: {input_count} inputs → {output_count} outputs (I/O={io_ratio:.3f})")
        
        return {
            'integrated_responses': integrated_responses,
            'output_spikes': output_spikes,
            'io_ratio': io_ratio,
            'total_inputs': input_count,
            'total_outputs': output_count
        }

def run_physiological_baseline_experiment():
    """生理学的ベースライン実験"""
    
    logging.info("="*60)
    logging.info("PHYSIOLOGICAL BASELINE EXPERIMENT")
    logging.info("="*60)
    
    # 実験設定
    duration_ms = 2000  # 2秒間
    seed = 42  # 再現性のため
    
    # 3つの主要条件
    conditions = {
        'SuM_dominant': {'DD': 0.17, 'MD': 0.17, 'PD': 0.66},
        'Traditional': {'DD': 0.5, 'MD': 0.5, 'PD': 0.0},
        'Balanced': {'DD': 0.33, 'MD': 0.33, 'PD': 0.33}
    }
    
    # 結果保存用
    results = {}
    
    for condition_name, weights in conditions.items():
        logging.info(f"\n--- Testing {condition_name} condition ---")
        
        # 生理学的スパイクパターン生成
        spike_trains = {
            'DD': PhysiologicalSpikeGenerator.dd_random_input(duration_ms, base_rate=12, seed=seed),
            'MD': PhysiologicalSpikeGenerator.md_theta_bursts(duration_ms, theta_freq=6, seed=seed),
            'PD': PhysiologicalSpikeGenerator.pd_behavioral_input(duration_ms, exploration_state=True, seed=seed)
        }
        
        # γ域での統合（主要解析）
        model_gamma = ThreePathwayIntegrationModel(weights, domain="gamma")
        result_gamma = model_gamma.simulate_integration(spike_trains)
        
        # θ域での統合（比較用）
        model_theta = ThreePathwayIntegrationModel(weights, domain="theta")
        result_theta = model_theta.simulate_integration(spike_trains)
        
        results[condition_name] = {
            'gamma': result_gamma,
            'theta': result_theta,
            'spike_trains': spike_trains
        }
        
        logging.info(f"{condition_name} results:")
        logging.info(f"  Gamma domain: I/O = {result_gamma['io_ratio']:.3f}")
        logging.info(f"  Theta domain: I/O = {result_theta['io_ratio']:.3f}")
    
    return results

def create_physiological_figures(results):
    """生理学的統合結果の可視化"""
    
    save_dir = "manuscript_figures"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Figure 3: 生理学的3経路統合
    fig = plt.figure(figsize=(18, 12))
    
    # Panel A: 各条件のI/O比比較
    ax1 = plt.subplot(2, 4, 1)
    conditions = list(results.keys())
    gamma_ios = [results[cond]['gamma']['io_ratio'] for cond in conditions]
    theta_ios = [results[cond]['theta']['io_ratio'] for cond in conditions]
    
    x = np.arange(len(conditions))
    width = 0.35
    
    ax1.bar(x - width/2, gamma_ios, width, label='Gamma domain', alpha=0.8, color='red')
    ax1.bar(x + width/2, theta_ios, width, label='Theta domain', alpha=0.8, color='blue')
    
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('I/O Ratio')
    ax1.set_title('A. Physiological Integration Efficiency')
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel B-D: 各条件のスパイクトレイン可視化
    for i, (condition, data) in enumerate(results.items()):
        ax = plt.subplot(2, 4, i + 2)
        
        spike_trains = data['spike_trains']
        colors = ['blue', 'orange', 'green']
        pathways = ['DD', 'MD', 'PD']
        
        for j, pathway in enumerate(pathways):
            spikes = spike_trains[pathway]
            y_pos = [j] * len(spikes)
            ax.scatter(spikes, y_pos, c=colors[j], s=20, alpha=0.7, label=pathway)
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Pathway')
        ax.set_title(f'{condition}')
        ax.set_yticks(range(3))
        ax.set_yticklabels(pathways)
        ax.legend()
        ax.set_xlim(0, 500)  # 最初の500msのみ表示
    
    # Panel E: SuM優位性定量化
    ax5 = plt.subplot(2, 4, 5)
    
    sum_gamma = results['SuM_dominant']['gamma']['io_ratio']
    trad_gamma = results['Traditional']['gamma']['io_ratio']
    balanced_gamma = results['Balanced']['gamma']['io_ratio']
    
    sum_advantage_vs_trad = ((sum_gamma - trad_gamma) / trad_gamma * 100) if trad_gamma > 0 else 0
    sum_advantage_vs_bal = ((sum_gamma - balanced_gamma) / balanced_gamma * 100) if balanced_gamma > 0 else 0
    
    comparisons = ['vs Traditional', 'vs Balanced']
    advantages = [sum_advantage_vs_trad, sum_advantage_vs_bal]
    
    bars = ax5.bar(comparisons, advantages, color=['red', 'blue'], alpha=0.7)
    ax5.set_ylabel('SuM Advantage (%)')
    ax5.set_title('E. SuM Advantage (Physiological)')
    ax5.grid(True, alpha=0.3)
    
    # 数値表示
    for bar, adv in zip(bars, advantages):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{adv:.1f}%', ha='center', va='bottom')
    
    # Panel F-H: 統合応答の時系列
    for i, (condition, data) in enumerate(results.items()):
        ax = plt.subplot(2, 4, i + 6)
        
        responses = data['gamma']['integrated_responses']
        times = [r[0] for r in responses[:100]]  # 最初の100イベント
        amplitudes = [r[1] for r in responses[:100]]
        
        ax.plot(times, amplitudes, 'o-', alpha=0.7, markersize=3)
        ax.axhline(y=0.30, color='red', linestyle='--', alpha=0.7, label='Threshold')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Integrated Response')
        ax.set_title(f'{condition} - Integration')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    filename = f"{save_dir}/Figure3_physiological_integration_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logging.info(f"Figure 3 saved: {filename}")
    
    plt.close()
    return filename

def generate_physiological_summary(results):
    """生理学的実験結果サマリー"""
    
    save_dir = "manuscript_figures"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(f"{save_dir}/physiological_summary_{timestamp}.txt", 'w') as f:
        f.write("PHYSIOLOGICAL THREE-PATHWAY INTEGRATION ANALYSIS\n")
        f.write("="*60 + "\n\n")
        
        f.write("EXPERIMENTAL DESIGN\n")
        f.write("-"*30 + "\n")
        f.write("DD pathway: Random input (Poisson, ~12Hz)\n")
        f.write("MD pathway: Theta bursts (6Hz, 3-5 spikes/burst)\n")
        f.write("PD pathway: Exploration gamma bursts\n\n")
        
        f.write("INTEGRATION RESULTS\n")
        f.write("-"*30 + "\n")
        
        for condition, data in results.items():
            gamma_io = data['gamma']['io_ratio']
            theta_io = data['theta']['io_ratio']
            
            f.write(f"{condition}:\n")
            f.write(f"  Gamma domain I/O: {gamma_io:.3f}\n")
            f.write(f"  Theta domain I/O: {theta_io:.3f}\n")
            
            # ゼロ除算エラー回避
            if theta_io > 0:
                ratio = gamma_io / theta_io
                f.write(f"  Gamma/Theta ratio: {ratio:.2f}\n")
            else:
                f.write(f"  Gamma/Theta ratio: infinite (theta I/O = 0)\n")
            f.write("\n")
        
        # SuM優位性計算
        sum_io = results['SuM_dominant']['gamma']['io_ratio']
        trad_io = results['Traditional']['gamma']['io_ratio']
        
        if trad_io > 0:
            advantage = (sum_io - trad_io) / trad_io * 100
            f.write(f"SuM ADVANTAGE\n")
            f.write("-"*30 + "\n")
            f.write(f"SuM dominant: {sum_io:.3f}\n")
            f.write(f"Traditional: {trad_io:.3f}\n")
            f.write(f"Advantage: {advantage:+.1f}%\n")

def main():
    """メイン実行関数"""
    
    # ロギング設定
    log_file = setup_logging()
    
    try:
        logging.info("Starting physiological three-pathway integration simulation")
        
        # 実験実行
        results = run_physiological_baseline_experiment()
        
        # 可視化
        figure_file = create_physiological_figures(results)
        
        # サマリー生成
        generate_physiological_summary(results)
        
        logging.info("="*60)
        logging.info("PHYSIOLOGICAL SIMULATION COMPLETED!")
        logging.info("="*60)
        logging.info(f"Debug log: {log_file}")
        logging.info(f"Figure: {figure_file}")
        
        # 主要結果の表示
        for condition, data in results.items():
            gamma_io = data['gamma']['io_ratio']
            logging.info(f"{condition}: Gamma I/O = {gamma_io:.3f}")
        
    except Exception as e:
        logging.error(f"Simulation failed: {e}")
        raise

if __name__ == "__main__":
    main()
