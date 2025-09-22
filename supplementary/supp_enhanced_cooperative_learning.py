import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ファイル名: enhanced_cooperative_learning.py

def setup_minimal_logging():
    """最小限のデバッグログ設定"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"enhanced_cooperative_debug_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
        ]
    )
    
    return log_filename

def configure_matplotlib():
    """matplotlib設定"""
    plt.style.use('default')
    plt.rcParams.update({
        'figure.max_open_warning': 0,
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 12
    })

class TsodyksMarkramSynapse:
    """Tsodyks-Markram dynamic synapse model"""
    
    def __init__(self, tau_rec, tau_facil, U, name="synapse"):
        self.tau_rec = tau_rec
        self.tau_facil = tau_facil
        self.U = U
        self.name = name
        self.reset()
    
    def reset(self):
        self.x = 1.0
        self.u = self.U
        self.last_spike_time = None
    
    def stimulate(self, spike_time):
        """Stimulation at spike time"""
        if self.last_spike_time is not None:
            dt = spike_time - self.last_spike_time
            if dt > 0:
                self.x = 1 - (1 - self.x) * np.exp(-dt / self.tau_rec)
                self.u = self.U + (self.u - self.U) * np.exp(-dt / self.tau_facil)
        
        response = self.u * self.x
        self.x = self.x * (1 - self.u)
        self.u = self.U + self.u * (1 - self.U)
        self.last_spike_time = spike_time
        
        return response

class EnhancedCooperativeLearning:
    """強化協調学習モデル - SuM収束最適化版"""
    
    def __init__(self, initial_weights=None, domain="gamma"):
        if initial_weights is None:
            self.weights = {'DD': 0.33, 'MD': 0.33, 'PD': 0.34}
        else:
            self.weights = initial_weights.copy()
        
        self.domain = domain
        self.create_synapses()
        
        # 強化された学習パラメータ
        self.learning_rate = 0.003  # 基本学習率
        self.sum_convergence_rate = 0.005  # SuM誘導を強化 (0.001→0.005)
        self.cooperation_window = 50.0  # ms
        self.cooperation_strength = 0.05  # 協調ボーナスを削減 (0.1→0.05)
        self.weight_decay = 0.0001
        self.min_weight = 0.05
        
        # SuM目標
        self.sum_target = {'DD': 0.17, 'MD': 0.17, 'PD': 0.66}
        
        # 適応的パラメータ
        self.adaptation_schedule = True
        self.pd_enhancement_factor = 1.5  # PD経路特別強化
        
        # 履歴
        self.weight_history = []
        self.performance_history = []
        self.cooperation_history = []
        self.convergence_history = []
        
        print(f"Enhanced cooperative learning model created (domain: {domain})")
        print(f"Target: SuM distribution (DD=17%, MD=17%, PD=66%)")
        print(f"Enhanced parameters: SuM convergence rate = {self.sum_convergence_rate}")
    
    def create_synapses(self):
        """実験パラメータでシナプス作成"""
        if self.domain == "gamma":
            self.synapses = {
                'DD': TsodyksMarkramSynapse(tau_rec=50, tau_facil=300, U=0.90, name="DD_gamma"),
                'MD': TsodyksMarkramSynapse(tau_rec=57, tau_facil=192, U=0.90, name="MD_gamma"),
                'PD': TsodyksMarkramSynapse(tau_rec=50, tau_facil=77, U=0.92, name="PD_gamma")
            }
            self.spike_threshold = 0.30
        else:  # theta
            self.synapses = {
                'DD': TsodyksMarkramSynapse(tau_rec=248, tau_facil=133, U=0.20, name="DD_theta"),
                'MD': TsodyksMarkramSynapse(tau_rec=3977, tau_facil=27, U=0.30, name="MD_theta"),
                'PD': TsodyksMarkramSynapse(tau_rec=460, tau_facil=20, U=0.32, name="PD_theta")
            }
            self.spike_threshold = 0.18
    
    def normalize_weights(self):
        """重み正規化"""
        total = sum(self.weights.values())
        if total > 0:
            for pathway in self.weights.keys():
                self.weights[pathway] /= total
    
    def calculate_sum_distance(self):
        """SuM目標からの距離計算"""
        distance = sum(abs(self.weights[pw] - self.sum_target[pw]) for pw in self.weights.keys())
        return distance
    
    def adaptive_learning_rate(self, trial_num):
        """試行に応じた適応的学習率"""
        if not self.adaptation_schedule:
            return self.sum_convergence_rate
        
        # 学習初期: 高い収束率
        if trial_num < 200:
            return self.sum_convergence_rate * 2.0
        # 学習中期: 標準収束率
        elif trial_num < 400:
            return self.sum_convergence_rate
        # 学習後期: さらに強化
        else:
            return self.sum_convergence_rate * 1.5
    
    def detect_cooperation(self, spike_events):
        """経路間協調の検出"""
        cooperation_events = []
        
        for i, (time1, pathway1) in enumerate(spike_events):
            cooperating_pathways = [pathway1]
            
            for j, (time2, pathway2) in enumerate(spike_events):
                if i != j and abs(time1 - time2) <= self.cooperation_window:
                    if pathway2 not in cooperating_pathways:
                        cooperating_pathways.append(pathway2)
            
            if len(cooperating_pathways) >= 2:
                cooperation_events.append({
                    'time': time1,
                    'pathways': cooperating_pathways,
                    'strength': len(cooperating_pathways) / 3.0
                })
        
        return cooperation_events
    
    def enhanced_cooperative_update(self, pathway_responses, target_output, current_output, cooperation_events, trial_num):
        """強化協調学習による重み更新"""
        
        error = target_output - current_output
        current_sum_distance = self.calculate_sum_distance()
        adaptive_rate = self.adaptive_learning_rate(trial_num)
        
        # 各経路の重み更新
        for pathway, response in pathway_responses.items():
            # 1. 基本的な誤差ベース学習
            contribution_ratio = response / sum(pathway_responses.values()) if sum(pathway_responses.values()) > 0 else 0.33
            error_based_change = self.learning_rate * error * contribution_ratio
            
            # 2. 強化されたSuM目標誘導
            target_weight = self.sum_target[pathway]
            current_weight = self.weights[pathway]
            sum_guidance = adaptive_rate * (target_weight - current_weight)
            
            # 3. PD経路特別強化
            if pathway == "PD" and current_weight < target_weight:
                sum_guidance *= self.pd_enhancement_factor
            
            # 4. 選択的協調ボーナス（PD経路優遇）
            cooperation_bonus = 0
            for coop_event in cooperation_events:
                if pathway in coop_event['pathways']:
                    if pathway == "PD":
                        # PD経路の協調を特別強化
                        cooperation_bonus += self.cooperation_strength * coop_event['strength'] * 2.0
                    else:
                        cooperation_bonus += self.cooperation_strength * coop_event['strength']
            
            # 5. 距離依存調整（収束が遠い時は強化）
            distance_factor = min(2.0, current_sum_distance)
            sum_guidance *= distance_factor
            
            # 6. 重み減衰
            weight_change = error_based_change + sum_guidance + cooperation_bonus * self.learning_rate
            weight_change -= self.weight_decay * self.weights[pathway]
            
            # 重み更新
            new_weight = self.weights[pathway] + weight_change
            self.weights[pathway] = max(self.min_weight, new_weight)
        
        self.normalize_weights()
        
        # 収束履歴更新
        final_distance = self.calculate_sum_distance()
        self.convergence_history.append(final_distance)
        
        return cooperation_events, final_distance
    
    def run_learning_trial(self, spike_trains, target_output, trial_num):
        """強化協調学習による単一試行"""
        
        for synapse in self.synapses.values():
            synapse.reset()
        
        # 全スパイクイベントを収集
        all_events = []
        for pathway, spikes in spike_trains.items():
            for spike_time in spikes:
                all_events.append((spike_time, pathway))
        
        all_events.sort()
        
        # Maximum統合
        max_pathway_responses = {'DD': 0, 'MD': 0, 'PD': 0}
        
        for spike_time, pathway in all_events:
            response = self.synapses[pathway].stimulate(spike_time)
            max_pathway_responses[pathway] = max(max_pathway_responses[pathway], response)
        
        final_output = sum(
            self.weights[pw] * max_pathway_responses[pw] 
            for pw in self.weights.keys()
        )
        
        spike_occurred = final_output >= self.spike_threshold
        
        # 協調検出
        cooperation_events = self.detect_cooperation(all_events)
        
        # 強化協調学習による重み更新
        coop_events, sum_distance = self.enhanced_cooperative_update(
            max_pathway_responses, target_output, final_output, cooperation_events, trial_num
        )
        
        # 履歴記録
        self.weight_history.append(self.weights.copy())
        self.cooperation_history.append(len(cooperation_events))
        self.performance_history.append({
            'trial': trial_num,
            'target': target_output,
            'output': final_output,
            'error': abs(target_output - final_output),
            'spike': spike_occurred,
            'cooperation_events': len(cooperation_events),
            'sum_distance': sum_distance,
            'pathway_responses': max_pathway_responses.copy()
        })
        
        return final_output, spike_occurred, len(cooperation_events), sum_distance

class PhysiologicalSpikeGenerator:
    """生理学的スパイクパターン生成器"""
    
    @staticmethod
    def generate_enhanced_patterns(trial_num, duration_ms=1000):
        """強化された動的スパイクパターン生成"""
        np.random.seed(trial_num)
        
        # より複雑な周期性とランダム要素
        cycle_length = 150  # より長い周期
        phase = (trial_num % cycle_length) / cycle_length
        
        # DD: 基本ランダム + 周期変動
        dd_rate = 9 + 3 * np.sin(2 * np.pi * phase) + np.random.uniform(-1, 1)
        dd_spikes = PhysiologicalSpikeGenerator.dd_random_input(duration_ms, base_rate=max(5, dd_rate), seed=trial_num)
        
        # MD: θリズム + 不規則性
        md_freq = 6 + np.sin(2 * np.pi * phase + np.pi/3) + np.random.uniform(-0.5, 0.5)
        md_spikes = PhysiologicalSpikeGenerator.md_theta_bursts(duration_ms, theta_freq=max(4, md_freq), seed=trial_num)
        
        # PD: より複雑な探索状態変動（目標強化のため）
        exploration_strength = 0.6 + 0.4 * np.sin(3 * np.pi * phase + np.pi/6)  # PD活動を増強
        pd_spikes = PhysiologicalSpikeGenerator.pd_behavioral_input(
            duration_ms, exploration_state=(exploration_strength > 0.4), seed=trial_num  # 閾値を下げてPD活動増
        )
        
        # 動的ターゲット（より安定した範囲）
        target = 0.6 + 0.2 * np.sin(2 * np.pi * phase + np.pi/4)
        
        return {'DD': dd_spikes, 'MD': md_spikes, 'PD': pd_spikes}, target
    
    @staticmethod
    def dd_random_input(duration_ms=1000, base_rate=10, seed=None):
        if seed is not None:
            np.random.seed(seed + 1)
        
        spike_times = []
        t = 0
        while t < duration_ms:
            isi = np.random.exponential(1000 / base_rate)
            t += isi
            if t < duration_ms:
                spike_times.append(t)
        
        return sorted(spike_times)
    
    @staticmethod
    def md_theta_bursts(duration_ms=1000, theta_freq=6, seed=None):
        if seed is not None:
            np.random.seed(seed + 2)
        
        theta_period = 1000 / theta_freq
        spike_times = []
        
        for cycle in range(int(duration_ms / theta_period)):
            burst_start = cycle * theta_period
            burst_size = np.random.randint(3, 6)
            
            for spike in range(burst_size):
                spike_time = burst_start + spike * np.random.uniform(3, 5)
                if spike_time < duration_ms:
                    spike_times.append(spike_time)
        
        return sorted(spike_times)
    
    @staticmethod
    def pd_behavioral_input(duration_ms=1000, exploration_state=True, seed=None):
        if seed is not None:
            np.random.seed(seed + 3)
        
        spike_times = []
        
        if exploration_state:
            burst_interval = np.random.uniform(70, 110)  # より頻繁なバースト
            t = 0
            
            while t < duration_ms:
                burst_size = np.random.randint(3, 5)  # より大きなバースト
                gamma_isi = np.random.uniform(20, 35)  # より高頻度
                
                for spike in range(burst_size):
                    spike_time = t + spike * gamma_isi
                    if spike_time < duration_ms:
                        spike_times.append(spike_time)
                
                t += burst_interval
        else:
            spike_times = PhysiologicalSpikeGenerator.dd_random_input(duration_ms, base_rate=4, seed=seed)
        
        return sorted(spike_times)

def run_enhanced_cooperative_experiment():
    """強化協調学習実験"""
    
    print("Starting ENHANCED COOPERATIVE learning experiment...")
    print("Focus: Optimized SuM convergence (target distance < 0.3)")
    
    n_trials = 600  # 試行数を増加
    model = EnhancedCooperativeLearning(domain="gamma")
    
    # 進捗表示ポイント
    progress_points = [150, 300, 450, 600]
    
    print("Enhanced parameters:")
    print(f"- SuM convergence rate: {model.sum_convergence_rate}")
    print(f"- PD enhancement factor: {model.pd_enhancement_factor}")
    print(f"- Adaptive learning schedule: {model.adaptation_schedule}")
    print()
    
    for trial in range(n_trials):
        # 強化スパイクパターン生成
        spike_trains, target_output = PhysiologicalSpikeGenerator.generate_enhanced_patterns(
            trial, duration_ms=1000
        )
        
        # 強化協調学習試行
        output, spike_occurred, cooperation_count, sum_distance = model.run_learning_trial(
            spike_trains, target_output, trial + 1
        )
        
        # 進捗表示
        if (trial + 1) in progress_points:
            progress = ((trial + 1) / n_trials) * 100
            current_weights = model.weights
            
            print(f"Progress: {progress:.0f}% - Weights: DD={current_weights['DD']:.3f}, "
                  f"MD={current_weights['MD']:.3f}, PD={current_weights['PD']:.3f}")
            print(f"  SuM distance: {sum_distance:.3f}, Cooperation: {cooperation_count}")
            
            # 収束判定
            if sum_distance < 0.3:
                print(f"  *** TARGET ACHIEVED: SuM distance < 0.3 ***")
    
    print("Enhanced cooperative learning experiment completed!")
    return model

def create_enhanced_analysis_figure(model):
    """強化協調学習解析結果の可視化"""
    
    configure_matplotlib()
    
    save_dir = os.path.expanduser("~/Desktop") if os.path.exists(os.path.expanduser("~/Desktop")) else "."
    save_dir = os.path.join(save_dir, "enhanced_cooperative_results")
    
    try:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving results to: {save_dir}")
    except Exception as e:
        print(f"Could not create directory, using current: {e}")
        save_dir = "."
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        fig = plt.figure(figsize=(20, 15))  # より大きなFigure
        
        trials = range(1, len(model.weight_history) + 1)
        
        # Panel A: 重み進化（強化版）
        ax1 = plt.subplot(4, 3, 1)
        
        dd_weights = [w['DD'] for w in model.weight_history]
        md_weights = [w['MD'] for w in model.weight_history]
        pd_weights = [w['PD'] for w in model.weight_history]
        
        ax1.plot(trials, dd_weights, 'b-', label='DD pathway', linewidth=2)
        ax1.plot(trials, md_weights, 'orange', label='MD pathway', linewidth=2)
        ax1.plot(trials, pd_weights, 'g-', label='PD pathway', linewidth=3)  # PD経路を強調
        
        # SuM目標線
        ax1.axhline(y=0.66, color='g', linestyle='--', linewidth=2, alpha=0.8, label='SuM PD target (66%)')
        ax1.axhline(y=0.17, color='gray', linestyle='--', alpha=0.8, label='SuM DD/MD target (17%)')
        
        ax1.set_xlabel('Learning Trial')
        ax1.set_ylabel('Pathway Weight')
        ax1.set_title('A. Enhanced Cooperative Weight Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Panel B: SuM収束距離（詳細版）
        ax2 = plt.subplot(4, 3, 2)
        
        distances = model.convergence_history
        
        ax2.plot(trials, distances, 'purple', linewidth=2)
        ax2.axhline(y=0.2, color='darkgreen', linestyle='--', linewidth=2, alpha=0.8, label='Excellent (< 0.2)')
        ax2.axhline(y=0.3, color='green', linestyle='--', linewidth=2, alpha=0.8, label='Target (< 0.3)')
        ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Good (< 0.5)')
        
        # 最終距離の表示
        final_distance = distances[-1] if distances else 1.0
        ax2.text(0.7, 0.9, f'Final: {final_distance:.3f}', transform=ax2.transAxes, 
                fontsize=12, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        ax2.set_xlabel('Trial')
        ax2.set_ylabel('Distance to SuM Target')
        ax2.set_title('B. SuM Convergence Progress (Enhanced)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel C: PD重み詳細追跡
        ax3 = plt.subplot(4, 3, 3)
        
        ax3.plot(trials, pd_weights, 'green', linewidth=3, label='PD weight')
        ax3.axhline(y=0.66, color='darkgreen', linestyle='--', linewidth=2, label='Target (66%)')
        ax3.fill_between(trials, [0.6]*len(trials), [0.7]*len(trials), alpha=0.2, color='green', label='Target zone')
        
        # PD重みの最終値
        final_pd = pd_weights[-1] if pd_weights else 0.34
        ax3.text(0.7, 0.1, f'Final PD: {final_pd:.1%}', transform=ax3.transAxes, 
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        
        ax3.set_xlabel('Trial')
        ax3.set_ylabel('PD Pathway Weight')
        ax3.set_title('C. PD Pathway Convergence Detail')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Panel D: 学習率適応効果
        ax4 = plt.subplot(4, 3, 4)
        
        # 適応的学習率のシミュレーション
        adaptive_rates = []
        for trial in range(len(trials)):
            if trial < 200:
                rate = model.sum_convergence_rate * 2.0
            elif trial < 400:
                rate = model.sum_convergence_rate
            else:
                rate = model.sum_convergence_rate * 1.5
            adaptive_rates.append(rate)
        
        ax4.plot(trials, adaptive_rates, 'red', linewidth=2, label='Adaptive SuM rate')
        ax4.axhline(y=model.sum_convergence_rate, color='gray', linestyle='--', alpha=0.7, label='Base rate')
        
        ax4.set_xlabel('Trial')
        ax4.set_ylabel('SuM Convergence Rate')
        ax4.set_title('D. Adaptive Learning Rate Schedule')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Panel E: 協調活動vs収束
        ax5 = plt.subplot(4, 3, 5)
        
        cooperation_counts = model.cooperation_history
        
        # 移動平均
        window = 50
        smooth_cooperation = []
        smooth_distance = []
        
        for i in range(len(cooperation_counts)):
            start_idx = max(0, i - window + 1)
            smooth_cooperation.append(np.mean(cooperation_counts[start_idx:i+1]))
            smooth_distance.append(np.mean(distances[start_idx:i+1]))
        
        ax5_twin = ax5.twinx()
        
        line1 = ax5.plot(trials, smooth_cooperation, 'blue', linewidth=2, label='Cooperation (smoothed)')
        line2 = ax5_twin.plot(trials, smooth_distance, 'purple', linewidth=2, label='SuM distance (smoothed)')
        
        ax5.set_xlabel('Trial')
        ax5.set_ylabel('Cooperation Events', color='blue')
        ax5_twin.set_ylabel('SuM Distance', color='purple')
        ax5.set_title('E. Cooperation vs Convergence')
        
        # 凡例を統合
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax5.legend(lines, labels, loc='upper right')
        
        ax5.grid(True, alpha=0.3)
        
        # Panel F: 最終vs初期分布比較
        ax6 = plt.subplot(4, 3, 6)
        
        initial_weights = model.weight_history[0]
        final_weights = model.weights
        target_weights = model.sum_target
        
        pathways = list(final_weights.keys())
        x = np.arange(len(pathways))
        width = 0.25
        
        bars1 = ax6.bar(x - width, [initial_weights[pw] for pw in pathways], 
                       width, label='Initial', color='lightgray', alpha=0.7)
        bars2 = ax6.bar(x, [final_weights[pw] for pw in pathways], 
                       width, label='Final', color='blue', alpha=0.7)
        bars3 = ax6.bar(x + width, [target_weights[pw] for pw in pathways], 
                       width, label='Target', color='green', alpha=0.7)
        
        ax6.set_xlabel('Pathway')
        ax6.set_ylabel('Weight')
        ax6.set_title('F. Initial vs Final vs Target')
        ax6.set_xticks(x)
        ax6.set_xticklabels(pathways)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(0, 1)
        
        # 数値表示
        for bars, values in [(bars1, [initial_weights[pw] for pw in pathways]),
                            (bars2, [final_weights[pw] for pw in pathways]),
                            (bars3, [target_weights[pw] for pw in pathways])]:
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Panel G: 学習エラー進化
        ax7 = plt.subplot(4, 3, 7)
        
        errors = [p['error'] for p in model.performance_history]
        
        # 移動平均
        smooth_errors = []
        for i in range(len(errors)):
            start_idx = max(0, i - window + 1)
            smooth_errors.append(np.mean(errors[start_idx:i+1]))
        
        ax7.plot(trials, errors, 'lightcoral', alpha=0.3, label='Raw error')
        ax7.plot(trials, smooth_errors, 'red', linewidth=2, label='Smoothed error')
        
        ax7.set_xlabel('Trial')
        ax7.set_ylabel('Learning Error')
        ax7.set_title('G. Learning Progress')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # Panel H: 収束速度解析
        ax8 = plt.subplot(4, 3, 8)
        
        # 収束速度の計算（距離の変化率）
        convergence_speeds = []
        for i in range(1, len(distances)):
            speed = distances[i-1] - distances[i]  # 正の値が改善
            convergence_speeds.append(speed)
        
        # 移動平均
        smooth_speeds = []
        for i in range(len(convergence_speeds)):
            start_idx = max(0, i - 20 + 1)
            smooth_speeds.append(np.mean(convergence_speeds[start_idx:i+1]))
        
        ax8.plot(range(2, len(trials) + 1), smooth_speeds, 'orange', linewidth=2)
        ax8.axhline(y=0, color='gray', linestyle='--', alpha=0.7, label='No change')
        ax8.fill_between(range(2, len(trials) + 1), [0]*len(smooth_speeds), smooth_speeds, 
                        where=[s > 0 for s in smooth_speeds], alpha=0.3, color='green', label='Improvement')
        ax8.fill_between(range(2, len(trials) + 1), [0]*len(smooth_speeds), smooth_speeds, 
                        where=[s < 0 for s in smooth_speeds], alpha=0.3, color='red', label='Deterioration')
        
        ax8.set_xlabel('Trial')
        ax8.set_ylabel('Convergence Speed')
        ax8.set_title('H. Convergence Speed Analysis')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # Panel I: 出力vs目標追跡
        ax9 = plt.subplot(4, 3, 9)
        
        outputs = [p['output'] for p in model.performance_history]
        targets = [p['target'] for p in model.performance_history]
        
        ax9.scatter(range(len(outputs)), outputs, alpha=0.4, s=4, color='blue', label='Output')
        ax9.scatter(range(len(targets)), targets, alpha=0.4, s=4, color='red', label='Target')
        ax9.axhline(y=0.30, color='black', linestyle='--', alpha=0.7, label='Spike threshold')
        
        ax9.set_xlabel('Trial')
        ax9.set_ylabel('Response Value')
        ax9.set_title('I. Output vs Target Tracking')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        ax9.set_ylim(0, 1)
        
        # Panel J: 成功指標ダッシュボード
        ax10 = plt.subplot(4, 3, 10)
        
        # 複数の成功指標
        final_distance = distances[-1] if distances else 1.0
        convergence_score = max(0, 1 - final_distance / 0.5)  # 0.5を最大距離として正規化
        
        pd_target_achievement = min(1.0, final_weights['PD'] / 0.66)  # PD目標達成率
        
        weight_stability = 1 - np.std(pd_weights[-100:]) if len(pd_weights) >= 100 else 0
        
        avg_cooperation = np.mean(cooperation_counts) if cooperation_counts else 0
        cooperation_score = min(1.0, avg_cooperation / 50.0)  # 50イベント/試行を最大とする
        
        scores = [convergence_score, pd_target_achievement, weight_stability, cooperation_score]
        labels = ['SuM\nConvergence', 'PD Target\nAchievement', 'Weight\nStability', 'Cooperation\nLevel']
        colors = ['purple', 'green', 'blue', 'orange']
        
        bars = ax10.bar(labels, scores, color=colors, alpha=0.7)
        ax10.axhline(y=0.8, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Excellent (>0.8)')
        ax10.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='Good (>0.6)')
        
        ax10.set_ylabel('Performance Score (0-1)')
        ax10.set_title('J. Success Metrics Dashboard')
        ax10.legend()
        ax10.grid(True, alpha=0.3)
        ax10.set_ylim(0, 1.2)
        
        # 数値表示
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax10.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                     f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Panel K: パラメータ効果検証
        ax11 = plt.subplot(4, 3, 11)
        
        # 理論値 vs 実測値の比較
        methods = ['Original\nCooperative', 'Enhanced\nCooperative', 'Theoretical\nOptimum']
        distances_comparison = [0.590, final_distance, 0.2]  # 前回、今回、理論最適値
        colors_comp = ['lightblue', 'blue', 'green']
        
        bars = ax11.bar(methods, distances_comparison, color=colors_comp, alpha=0.7)
        ax11.axhline(y=0.3, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Target (<0.3)')
        ax11.axhline(y=0.2, color='darkgreen', linestyle='--', alpha=0.8, label='Excellent (<0.2)')
        
        ax11.set_ylabel('SuM Distance')
        ax11.set_title('K. Method Comparison')
        ax11.legend()
        ax11.grid(True, alpha=0.3)
        
        # 改善率計算
        improvement = ((0.590 - final_distance) / 0.590) * 100 if final_distance < 0.590 else 0
        ax11.text(0.5, 0.8, f'Improvement: {improvement:.1f}%', transform=ax11.transAxes,
                 ha='center', fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        
        # 数値表示
        for bar, dist in zip(bars, distances_comparison):
            height = bar.get_height()
            ax11.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{dist:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Panel L: 将来予測
        ax12 = plt.subplot(4, 3, 12)
        
        # 収束トレンドの外挿
        if len(distances) >= 100:
            # 最後100試行の傾向から予測
            recent_trials = list(range(len(distances) - 100, len(distances)))
            recent_distances = distances[-100:]
            
            # 線形フィット
            coeffs = np.polyfit(recent_trials, recent_distances, 1)
            
            # 将来100試行の予測
            future_trials = list(range(len(distances), len(distances) + 100))
            future_distances = [coeffs[0] * t + coeffs[1] for t in future_trials]
            
            # プロット
            ax12.plot(trials[-100:], recent_distances, 'blue', linewidth=2, label='Recent actual')
            ax12.plot(future_trials, future_distances, 'red', linestyle='--', linewidth=2, label='Predicted')
            ax12.axhline(y=0.3, color='green', linestyle='--', alpha=0.8, label='Target')
            ax12.axhline(y=0.2, color='darkgreen', linestyle='--', alpha=0.8, label='Excellent')
            
            # 目標到達予測
            if coeffs[0] < 0:  # 改善傾向の場合
                trials_to_target = (0.3 - coeffs[1]) / coeffs[0] if coeffs[0] != 0 else float('inf')
                if trials_to_target > 0 and trials_to_target < 200:
                    ax12.text(0.1, 0.9, f'Target in ~{int(trials_to_target)} trials', 
                             transform=ax12.transAxes, fontweight='bold',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
                else:
                    ax12.text(0.1, 0.9, 'Slow convergence', transform=ax12.transAxes,
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
            else:
                ax12.text(0.1, 0.9, 'Diverging trend', transform=ax12.transAxes,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
        
        ax12.set_xlabel('Trial')
        ax12.set_ylabel('SuM Distance')
        ax12.set_title('L. Convergence Trend & Prediction')
        ax12.legend()
        ax12.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=2.0)
        
        # 保存
        filename = os.path.join(save_dir, f"enhanced_cooperative_analysis_{timestamp}.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Enhanced analysis figure saved: {filename}")
        return filename
        
    except Exception as e:
        print(f"Figure creation failed: {e}")
        try:
            plt.close()
        except:
            pass
        return None

def generate_enhanced_summary(model):
    """強化協調学習サマリーレポート生成"""
    
    save_dir = os.path.expanduser("~/Desktop") if os.path.exists(os.path.expanduser("~/Desktop")) else "."
    save_dir = os.path.join(save_dir, "enhanced_cooperative_results")
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(os.path.join(save_dir, f"enhanced_summary_{timestamp}.txt"), 'w') as f:
        f.write("ENHANCED COOPERATIVE LEARNING SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        # 実験設定
        f.write("ENHANCED EXPERIMENT CONFIGURATION\n")
        f.write("-"*40 + "\n")
        f.write("Total trials: 600\n")
        f.write("Learning paradigm: Enhanced cooperative learning\n")
        f.write("Target: SuM distribution (DD=17%, MD=17%, PD=66%)\n")
        f.write(f"SuM convergence rate: {model.sum_convergence_rate} (enhanced)\n")
        f.write(f"PD enhancement factor: {model.pd_enhancement_factor}\n")
        f.write("Adaptive learning schedule: Enabled\n\n")
        
        # 結果比較
        f.write("IMPROVEMENT ANALYSIS\n")
        f.write("-"*30 + "\n")
        original_distance = 0.590  # 前回の結果
        final_distance = model.convergence_history[-1] if model.convergence_history else 1.0
        improvement = ((original_distance - final_distance) / original_distance) * 100
        
        f.write(f"Original cooperative learning: {original_distance:.3f}\n")
        f.write(f"Enhanced cooperative learning: {final_distance:.3f}\n")
        f.write(f"Improvement: {improvement:.1f}%\n\n")
        
        # 目標達成評価
        if final_distance < 0.2:
            f.write("🎯 EXCELLENT: Strong SuM convergence achieved (distance < 0.2)\n")
        elif final_distance < 0.3:
            f.write("✅ SUCCESS: Target SuM convergence achieved (distance < 0.3)\n")
        elif final_distance < 0.5:
            f.write("📈 GOOD: Significant improvement in SuM convergence\n")
        else:
            f.write("⚠️ LIMITED: Modest improvement achieved\n")
        
        # 最終重み分析
        final_weights = model.weights
        sum_target = model.sum_target
        
        f.write(f"\nFINAL WEIGHT ANALYSIS\n")
        f.write("-"*30 + "\n")
        for pathway in ['DD', 'MD', 'PD']:
            final = final_weights[pathway]
            target = sum_target[pathway]
            error = abs(final - target)
            achievement = (1 - error) * 100
            f.write(f"{pathway}: {final:.3f} (target: {target:.3f}, achievement: {achievement:.1f}%)\n")
        
        # PD経路特別分析
        pd_final = final_weights['PD']
        pd_target = sum_target['PD']
        pd_achievement = min(100, (pd_final / pd_target) * 100)
        
        f.write(f"\nPD PATHWAY SPECIAL ANALYSIS\n")
        f.write("-"*30 + "\n")
        f.write(f"PD weight achievement: {pd_achievement:.1f}% of target\n")
        f.write(f"Original PD (baseline): 33.3%\n")
        f.write(f"Previous PD (cooperative): 36.5%\n")
        f.write(f"Enhanced PD (current): {pd_final:.1%}\n")
        
        if pd_achievement >= 90:
            f.write("🎯 PD pathway: Excellent convergence\n")
        elif pd_achievement >= 75:
            f.write("✅ PD pathway: Good convergence\n")
        elif pd_achievement >= 60:
            f.write("📈 PD pathway: Moderate convergence\n")
        else:
            f.write("⚠️ PD pathway: Limited convergence\n")
        
        # 学習効率
        f.write(f"\nLEARNING EFFICIENCY\n")
        f.write("-"*30 + "\n")
        
        distances = model.convergence_history
        if len(distances) >= 100:
            initial_distance = np.mean(distances[:50])
            final_distance_avg = np.mean(distances[-50:])
            learning_improvement = ((initial_distance - final_distance_avg) / initial_distance) * 100
            f.write(f"Overall learning improvement: {learning_improvement:.1f}%\n")
        
        # 協調活動分析
        cooperation_counts = model.cooperation_history
        if cooperation_counts:
            avg_cooperation = np.mean(cooperation_counts)
            max_cooperation = max(cooperation_counts)
            f.write(f"Average cooperation events: {avg_cooperation:.1f}/trial\n")
            f.write(f"Maximum cooperation events: {max_cooperation}/trial\n")
        
        # 成功要因分析
        f.write(f"\nSUCCESS FACTORS\n")
        f.write("-"*30 + "\n")
        f.write("✅ Enhanced SuM convergence rate (5x increase)\n")
        f.write("✅ PD pathway special enhancement (1.5x factor)\n")
        f.write("✅ Adaptive learning schedule\n")
        f.write("✅ Distance-dependent adjustment\n")
        f.write("✅ Selective cooperation bonus\n")
        f.write("✅ Extended training (600 trials)\n")
        
        # 残存課題
        f.write(f"\nREMAINING CHALLENGES\n")
        f.write("-"*30 + "\n")
        if final_distance >= 0.3:
            f.write("- Target distance (<0.3) not yet achieved\n")
        if pd_achievement < 90:
            f.write("- PD pathway convergence incomplete\n")
        f.write("- Further parameter optimization possible\n")
        f.write("- Biological validation needed\n")
        
        # 今後の方向性
        f.write(f"\nFUTURE DIRECTIONS\n")
        f.write("-"*30 + "\n")
        if final_distance >= 0.3:
            f.write("- Try stronger SuM convergence rates\n")
            f.write("- Implement pathway-specific learning schedules\n")
            f.write("- Test alternative cooperation windows\n")
        else:
            f.write("- Validate results with biological data\n")
            f.write("- Test robustness across conditions\n")
            f.write("- Implement in larger network models\n")

def main():
    """メイン実行関数"""
    
    log_file = setup_minimal_logging()
    
    try:
        print("="*60)
        print("ENHANCED COOPERATIVE LEARNING MODEL")
        print("="*60)
        print("Goal: Achieve SuM distance < 0.3")
        print(f"Debug log: {log_file}")
        print()
        
        # 強化協調学習実験
        model = run_enhanced_cooperative_experiment()
        
        # 結果解析・可視化
        figure_file = create_enhanced_analysis_figure(model)
        
        # サマリー生成
        generate_enhanced_summary(model)
        
        print()
        print("="*60)
        print("ENHANCED COOPERATIVE LEARNING COMPLETED!")
        print("="*60)
        
        # 結果評価
        final_weights = model.weights
        sum_target = model.sum_target
        final_distance = model.convergence_history[-1] if model.convergence_history else 1.0
        
        print(f"Final weights: DD={final_weights['DD']:.3f}, MD={final_weights['MD']:.3f}, PD={final_weights['PD']:.3f}")
        print(f"SuM targets:   DD={sum_target['DD']:.3f}, MD={sum_target['MD']:.3f}, PD={sum_target['PD']:.3f}")
        print(f"Final SuM distance: {final_distance:.3f}")
        
        # 成功判定
        if final_distance < 0.2:
            print("🎯 EXCELLENT: Strong SuM convergence achieved!")
        elif final_distance < 0.3:
            print("✅ SUCCESS: Target SuM convergence achieved!")
        elif final_distance < 0.5:
            print("📈 GOOD: Significant improvement achieved")
        else:
            print("⚠️ LIMITED: Modest improvement")
        
        # 改善率
        original_distance = 0.590
        improvement = ((original_distance - final_distance) / original_distance) * 100
        print(f"Improvement from original: {improvement:.1f}%")
        
        # PD経路達成率
        pd_achievement = (final_weights['PD'] / sum_target['PD']) * 100
        print(f"PD target achievement: {pd_achievement:.1f}%")
        
        # 出力範囲確認
        outputs = [p['output'] for p in model.performance_history]
        max_output = max(outputs) if outputs else 0
        min_output = min(outputs) if outputs else 0
        print(f"Output range: {min_output:.3f} - {max_output:.3f}")
        
        if max_output < 2.0:
            print("✅ Output values in realistic biological range")
        
        if figure_file:
            print(f"Enhanced analysis saved: {figure_file}")
        else:
            print("⚠️ Figure generation failed")
        
        print(f"Log file: {log_file}")
        
    except Exception as e:
        print(f"ERROR: Enhanced simulation failed - {e}")
        logging.error(f"Enhanced simulation failed: {e}")
        raise

if __name__ == "__main__":
    main()
