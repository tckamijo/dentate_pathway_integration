import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
from datetime import datetime
import warnings
import json
warnings.filterwarnings('ignore')

# ファイル名: robustness_validation_study.py

def setup_logging():
    """検証実験用ログ設定"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"robustness_validation_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return log_filename

def configure_matplotlib():
    """matplotlib設定"""
    plt.style.use('default')
    plt.rcParams.update({
        'figure.max_open_warning': 0,
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 11
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

class ValidationCooperativeLearning:
    """検証用協調学習モデル"""
    
    def __init__(self, sum_convergence_rate=0.005, pd_enhancement_factor=1.5, 
                 initial_weights=None, domain="gamma", seed=None):
        
        if initial_weights is None:
            self.weights = {'DD': 0.33, 'MD': 0.33, 'PD': 0.34}
        else:
            self.weights = initial_weights.copy()
        
        self.domain = domain
        self.seed = seed
        self.create_synapses()
        
        # パラメータ（外部から指定可能）
        self.learning_rate = 0.003
        self.sum_convergence_rate = sum_convergence_rate
        self.cooperation_window = 50.0
        self.cooperation_strength = 0.05
        self.pd_enhancement_factor = pd_enhancement_factor
        self.weight_decay = 0.0001
        self.min_weight = 0.05
        
        # SuM目標
        self.sum_target = {'DD': 0.17, 'MD': 0.17, 'PD': 0.66}
        
        # 履歴
        self.weight_history = []
        self.performance_history = []
        self.cooperation_history = []
        self.convergence_history = []
        
    def create_synapses(self):
        if self.domain == "gamma":
            self.synapses = {
                'DD': TsodyksMarkramSynapse(tau_rec=50, tau_facil=300, U=0.90, name="DD_gamma"),
                'MD': TsodyksMarkramSynapse(tau_rec=57, tau_facil=192, U=0.90, name="MD_gamma"),
                'PD': TsodyksMarkramSynapse(tau_rec=50, tau_facil=77, U=0.92, name="PD_gamma")
            }
            self.spike_threshold = 0.30
        else:
            self.synapses = {
                'DD': TsodyksMarkramSynapse(tau_rec=248, tau_facil=133, U=0.20, name="DD_theta"),
                'MD': TsodyksMarkramSynapse(tau_rec=3977, tau_facil=27, U=0.30, name="MD_theta"),
                'PD': TsodyksMarkramSynapse(tau_rec=460, tau_facil=20, U=0.32, name="PD_theta")
            }
            self.spike_threshold = 0.18
    
    def normalize_weights(self):
        total = sum(self.weights.values())
        if total > 0:
            for pathway in self.weights.keys():
                self.weights[pathway] /= total
    
    def calculate_sum_distance(self):
        return sum(abs(self.weights[pw] - self.sum_target[pw]) for pw in self.weights.keys())
    
    def adaptive_learning_rate(self, trial_num):
        if trial_num < 200:
            return self.sum_convergence_rate * 2.0
        elif trial_num < 400:
            return self.sum_convergence_rate
        else:
            return self.sum_convergence_rate * 1.5
    
    def detect_cooperation(self, spike_events):
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
        error = target_output - current_output
        current_sum_distance = self.calculate_sum_distance()
        adaptive_rate = self.adaptive_learning_rate(trial_num)
        
        for pathway, response in pathway_responses.items():
            # 基本学習
            contribution_ratio = response / sum(pathway_responses.values()) if sum(pathway_responses.values()) > 0 else 0.33
            error_based_change = self.learning_rate * error * contribution_ratio
            
            # SuM誘導
            target_weight = self.sum_target[pathway]
            current_weight = self.weights[pathway]
            sum_guidance = adaptive_rate * (target_weight - current_weight)
            
            # PD強化
            if pathway == "PD" and current_weight < target_weight:
                sum_guidance *= self.pd_enhancement_factor
            
            # 協調ボーナス
            cooperation_bonus = 0
            for coop_event in cooperation_events:
                if pathway in coop_event['pathways']:
                    if pathway == "PD":
                        cooperation_bonus += self.cooperation_strength * coop_event['strength'] * 2.0
                    else:
                        cooperation_bonus += self.cooperation_strength * coop_event['strength']
            
            # 距離依存調整
            distance_factor = min(2.0, current_sum_distance)
            sum_guidance *= distance_factor
            
            # 重み更新
            weight_change = error_based_change + sum_guidance + cooperation_bonus * self.learning_rate
            weight_change -= self.weight_decay * self.weights[pathway]
            
            new_weight = self.weights[pathway] + weight_change
            self.weights[pathway] = max(self.min_weight, new_weight)
        
        self.normalize_weights()
        final_distance = self.calculate_sum_distance()
        self.convergence_history.append(final_distance)
        
        return cooperation_events, final_distance
    
    def run_learning_trial(self, spike_trains, target_output, trial_num):
        for synapse in self.synapses.values():
            synapse.reset()
        
        all_events = []
        for pathway, spikes in spike_trains.items():
            for spike_time in spikes:
                all_events.append((spike_time, pathway))
        
        all_events.sort()
        
        max_pathway_responses = {'DD': 0, 'MD': 0, 'PD': 0}
        
        for spike_time, pathway in all_events:
            response = self.synapses[pathway].stimulate(spike_time)
            max_pathway_responses[pathway] = max(max_pathway_responses[pathway], response)
        
        final_output = sum(
            self.weights[pw] * max_pathway_responses[pw] 
            for pw in self.weights.keys()
        )
        
        spike_occurred = final_output >= self.spike_threshold
        cooperation_events = self.detect_cooperation(all_events)
        
        coop_events, sum_distance = self.enhanced_cooperative_update(
            max_pathway_responses, target_output, final_output, cooperation_events, trial_num
        )
        
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
    def generate_enhanced_patterns(trial_num, duration_ms=1000, base_seed=None):
        if base_seed is not None:
            np.random.seed(base_seed + trial_num)
        else:
            np.random.seed(trial_num)
        
        cycle_length = 150
        phase = (trial_num % cycle_length) / cycle_length
        
        dd_rate = 9 + 3 * np.sin(2 * np.pi * phase) + np.random.uniform(-1, 1)
        dd_spikes = PhysiologicalSpikeGenerator.dd_random_input(duration_ms, base_rate=max(5, dd_rate), seed=trial_num)
        
        md_freq = 6 + np.sin(2 * np.pi * phase + np.pi/3) + np.random.uniform(-0.5, 0.5)
        md_spikes = PhysiologicalSpikeGenerator.md_theta_bursts(duration_ms, theta_freq=max(4, md_freq), seed=trial_num)
        
        exploration_strength = 0.6 + 0.4 * np.sin(3 * np.pi * phase + np.pi/6)
        pd_spikes = PhysiologicalSpikeGenerator.pd_behavioral_input(
            duration_ms, exploration_state=(exploration_strength > 0.4), seed=trial_num
        )
        
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
            burst_interval = np.random.uniform(70, 110)
            t = 0
            
            while t < duration_ms:
                burst_size = np.random.randint(3, 5)
                gamma_isi = np.random.uniform(20, 35)
                
                for spike in range(burst_size):
                    spike_time = t + spike * gamma_isi
                    if spike_time < duration_ms:
                        spike_times.append(spike_time)
                
                t += burst_interval
        else:
            spike_times = PhysiologicalSpikeGenerator.dd_random_input(duration_ms, base_rate=4, seed=seed)
        
        return sorted(spike_times)

def run_single_experiment(sum_rate, pd_factor, seed, n_trials=400):
    """単一実験の実行"""
    model = ValidationCooperativeLearning(
        sum_convergence_rate=sum_rate,
        pd_enhancement_factor=pd_factor,
        seed=seed
    )
    
    for trial in range(n_trials):
        spike_trains, target_output = PhysiologicalSpikeGenerator.generate_enhanced_patterns(
            trial, duration_ms=1000, base_seed=seed
        )
        
        output, spike_occurred, cooperation_count, sum_distance = model.run_learning_trial(
            spike_trains, target_output, trial + 1
        )
    
    return {
        'final_distance': model.convergence_history[-1] if model.convergence_history else 1.0,
        'final_weights': model.weights.copy(),
        'convergence_history': model.convergence_history.copy(),
        'pd_achievement': (model.weights['PD'] / 0.66) * 100,
        'avg_cooperation': np.mean(model.cooperation_history) if model.cooperation_history else 0
    }

def reproducibility_test():
    """再現性テスト：異なるシードで複数回実行"""
    logging.info("=== REPRODUCIBILITY TEST ===")
    logging.info("Testing with different random seeds (baseline parameters)")
    
    seeds = [42, 123, 456, 789, 999, 1337, 2021, 3141, 5678, 9999]
    results = []
    
    for i, seed in enumerate(seeds):
        logging.info(f"Running experiment {i+1}/10 with seed {seed}")
        result = run_single_experiment(
            sum_rate=0.005,
            pd_factor=1.5,
            seed=seed,
            n_trials=400
        )
        result['seed'] = seed
        results.append(result)
        
        logging.info(f"  Final distance: {result['final_distance']:.3f}")
        logging.info(f"  PD achievement: {result['pd_achievement']:.1f}%")
    
    return results

def parameter_sensitivity_analysis():
    """パラメータ感度解析"""
    logging.info("=== PARAMETER SENSITIVITY ANALYSIS ===")
    
    # SuM収束率テスト
    logging.info("Testing SuM convergence rates: 0.003, 0.005, 0.007")
    sum_rates = [0.003, 0.005, 0.007]
    sum_rate_results = []
    
    for rate in sum_rates:
        logging.info(f"Testing SuM rate: {rate}")
        results_for_rate = []
        
        # 各条件で3回実行
        for seed in [42, 123, 456]:
            result = run_single_experiment(
                sum_rate=rate,
                pd_factor=1.5,
                seed=seed,
                n_trials=300  # 少し短縮
            )
            results_for_rate.append(result)
        
        avg_distance = np.mean([r['final_distance'] for r in results_for_rate])
        avg_pd = np.mean([r['pd_achievement'] for r in results_for_rate])
        
        logging.info(f"  Average distance: {avg_distance:.3f}")
        logging.info(f"  Average PD achievement: {avg_pd:.1f}%")
        
        sum_rate_results.append({
            'sum_rate': rate,
            'results': results_for_rate,
            'avg_distance': avg_distance,
            'avg_pd_achievement': avg_pd
        })
    
    # PD強化係数テスト
    logging.info("Testing PD enhancement factors: 1.0, 1.5, 2.0")
    pd_factors = [1.0, 1.5, 2.0]
    pd_factor_results = []
    
    for factor in pd_factors:
        logging.info(f"Testing PD factor: {factor}")
        results_for_factor = []
        
        for seed in [42, 123, 456]:
            result = run_single_experiment(
                sum_rate=0.005,
                pd_factor=factor,
                seed=seed,
                n_trials=300
            )
            results_for_factor.append(result)
        
        avg_distance = np.mean([r['final_distance'] for r in results_for_factor])
        avg_pd = np.mean([r['pd_achievement'] for r in results_for_factor])
        
        logging.info(f"  Average distance: {avg_distance:.3f}")
        logging.info(f"  Average PD achievement: {avg_pd:.1f}%")
        
        pd_factor_results.append({
            'pd_factor': factor,
            'results': results_for_factor,
            'avg_distance': avg_distance,
            'avg_pd_achievement': avg_pd
        })
    
    return sum_rate_results, pd_factor_results

def create_validation_report(reproducibility_results, sum_rate_results, pd_factor_results):
    """検証レポートの作成"""
    configure_matplotlib()
    
    save_dir = os.path.expanduser("~/Desktop")
    save_dir = os.path.join(save_dir, "validation_results")
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 統計サマリーの計算
    repro_distances = [r['final_distance'] for r in reproducibility_results]
    repro_pd_achievements = [r['pd_achievement'] for r in reproducibility_results]
    
    # Figure作成
    fig = plt.figure(figsize=(16, 12))
    
    # Panel A: 再現性テスト結果
    ax1 = plt.subplot(2, 4, 1)
    
    seeds = [r['seed'] for r in reproducibility_results]
    ax1.bar(range(len(seeds)), repro_distances, alpha=0.7, color='blue')
    ax1.axhline(y=0.3, color='red', linestyle='--', alpha=0.8, label='Target (<0.3)')
    ax1.axhline(y=np.mean(repro_distances), color='orange', linestyle='-', label=f'Mean: {np.mean(repro_distances):.3f}')
    
    ax1.set_xlabel('Experiment Run')
    ax1.set_ylabel('Final SuM Distance')
    ax1.set_title('A. Reproducibility Test (10 runs)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel B: 再現性統計
    ax2 = plt.subplot(2, 4, 2)
    
    ax2.boxplot([repro_distances], labels=['SuM Distance'])
    ax2.axhline(y=0.3, color='red', linestyle='--', alpha=0.8, label='Target')
    ax2.axhline(y=0.2, color='green', linestyle='--', alpha=0.8, label='Excellent')
    
    # 統計情報を表示
    mean_dist = np.mean(repro_distances)
    std_dist = np.std(repro_distances)
    min_dist = np.min(repro_distances)
    max_dist = np.max(repro_distances)
    
    ax2.text(0.5, 0.95, f'Mean: {mean_dist:.3f}', transform=ax2.transAxes, ha='center')
    ax2.text(0.5, 0.90, f'Std: {std_dist:.3f}', transform=ax2.transAxes, ha='center')
    ax2.text(0.5, 0.85, f'Range: {min_dist:.3f}-{max_dist:.3f}', transform=ax2.transAxes, ha='center')
    
    ax2.set_ylabel('SuM Distance')
    ax2.set_title('B. Reproducibility Statistics')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel C: SuM収束率の影響
    ax3 = plt.subplot(2, 4, 3)
    
    sum_rates = [r['sum_rate'] for r in sum_rate_results]
    sum_avg_distances = [r['avg_distance'] for r in sum_rate_results]
    sum_std_distances = [np.std([res['final_distance'] for res in r['results']]) for r in sum_rate_results]
    
    ax3.errorbar(sum_rates, sum_avg_distances, yerr=sum_std_distances, 
                marker='o', capsize=5, capthick=2, linewidth=2)
    ax3.axhline(y=0.3, color='red', linestyle='--', alpha=0.8, label='Target')
    ax3.axhline(y=0.2, color='green', linestyle='--', alpha=0.8, label='Excellent')
    
    ax3.set_xlabel('SuM Convergence Rate')
    ax3.set_ylabel('Average SuM Distance')
    ax3.set_title('C. SuM Rate Sensitivity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel D: PD強化係数の影響
    ax4 = plt.subplot(2, 4, 4)
    
    pd_factors = [r['pd_factor'] for r in pd_factor_results]
    pd_avg_distances = [r['avg_distance'] for r in pd_factor_results]
    pd_std_distances = [np.std([res['final_distance'] for res in r['results']]) for r in pd_factor_results]
    
    ax4.errorbar(pd_factors, pd_avg_distances, yerr=pd_std_distances, 
                marker='s', capsize=5, capthick=2, linewidth=2, color='green')
    ax4.axhline(y=0.3, color='red', linestyle='--', alpha=0.8, label='Target')
    ax4.axhline(y=0.2, color='darkgreen', linestyle='--', alpha=0.8, label='Excellent')
    
    ax4.set_xlabel('PD Enhancement Factor')
    ax4.set_ylabel('Average SuM Distance')
    ax4.set_title('D. PD Factor Sensitivity')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Panel E: PD達成率の比較
    ax5 = plt.subplot(2, 4, 5)
    
    sum_avg_pd = [r['avg_pd_achievement'] for r in sum_rate_results]
    pd_avg_pd = [r['avg_pd_achievement'] for r in pd_factor_results]
    
    x1 = np.arange(len(sum_rates))
    x2 = np.arange(len(pd_factors))
    
    bars1 = ax5.bar(x1 - 0.2, sum_avg_pd, 0.4, label='SuM Rate Variation', alpha=0.7)
    bars2 = ax5.bar(x2 + 0.2, pd_avg_pd, 0.4, label='PD Factor Variation', alpha=0.7, color='green')
    
    ax5.axhline(y=100, color='green', linestyle='--', alpha=0.8, label='Perfect (100%)')
    ax5.axhline(y=80, color='orange', linestyle='--', alpha=0.8, label='Good (80%)')
    
    ax5.set_xlabel('Parameter Index')
    ax5.set_ylabel('PD Achievement (%)')
    ax5.set_title('E. PD Pathway Achievement')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Panel F: 成功率分析
    ax6 = plt.subplot(2, 4, 6)
    
    # 成功率の計算（距離<0.3を成功とする）
    repro_success_rate = (np.array(repro_distances) < 0.3).mean() * 100
    
    conditions = ['Reproducibility\n(10 runs)', 'SuM Rate\n0.003', 'SuM Rate\n0.005', 'SuM Rate\n0.007',
                 'PD Factor\n1.0', 'PD Factor\n1.5', 'PD Factor\n2.0']
    
    success_rates = [repro_success_rate]
    
    for result_set in sum_rate_results:
        distances = [r['final_distance'] for r in result_set['results']]
        success_rate = (np.array(distances) < 0.3).mean() * 100
        success_rates.append(success_rate)
    
    for result_set in pd_factor_results:
        distances = [r['final_distance'] for r in result_set['results']]
        success_rate = (np.array(distances) < 0.3).mean() * 100
        success_rates.append(success_rate)
    
    colors = ['blue'] + ['red']*3 + ['green']*3
    bars = ax6.bar(range(len(conditions)), success_rates, color=colors, alpha=0.7)
    
    ax6.set_xlabel('Condition')
    ax6.set_ylabel('Success Rate (%)')
    ax6.set_title('F. Success Rate (Distance < 0.3)')
    ax6.set_xticks(range(len(conditions)))
    ax6.set_xticklabels(conditions, rotation=45, ha='right')
    ax6.grid(True, alpha=0.3)
    
    # 数値表示
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.0f}%', ha='center', va='bottom', fontsize=8)
    
    # Panel G: 収束トレンドの比較
    ax7 = plt.subplot(2, 4, 7)
    
    # 代表的な条件での収束履歴をプロット
    if reproducibility_results:
        best_run = min(reproducibility_results, key=lambda x: x['final_distance'])
        worst_run = max(reproducibility_results, key=lambda x: x['final_distance'])
        
        trials = range(1, len(best_run['convergence_history']) + 1)
        ax7.plot(trials, best_run['convergence_history'], 'g-', label=f'Best run ({best_run["final_distance"]:.3f})', linewidth=2)
        ax7.plot(trials, worst_run['convergence_history'], 'r-', label=f'Worst run ({worst_run["final_distance"]:.3f})', linewidth=2)
        
        # 平均的な収束
        avg_convergence = np.mean([r['convergence_history'] for r in reproducibility_results], axis=0)
        ax7.plot(trials, avg_convergence, 'b--', label=f'Average', linewidth=2)
    
    ax7.axhline(y=0.3, color='orange', linestyle='--', alpha=0.8, label='Target')
    ax7.axhline(y=0.2, color='green', linestyle='--', alpha=0.8, label='Excellent')
    
    ax7.set_xlabel('Trial')
    ax7.set_ylabel('SuM Distance')
    ax7.set_title('G. Convergence Trajectories')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Panel H: 信頼性評価
    ax8 = plt.subplot(2, 4, 8)
    
    metrics = ['Mean Distance', 'Std Distance', 'Success Rate', 'PD Achievement']
    scores = [
        1 - mean_dist/0.5,  # 距離スコア（0.5を最大として正規化）
        1 - std_dist/0.1,   # 安定性スコア（0.1を最大標準偏差として）
        repro_success_rate/100,  # 成功率スコア
        np.mean(repro_pd_achievements)/100  # PD達成率スコア
    ]
    
    colors = ['blue', 'orange', 'green', 'purple']
    bars = ax8.bar(metrics, scores, color=colors, alpha=0.7)
    ax8.axhline(y=0.8, color='red', linestyle='--', alpha=0.8, label='Good (>0.8)')
    ax8.axhline(y=0.6, color='orange', linestyle='--', alpha=0.8, label='Fair (>0.6)')
    
    ax8.set_ylabel('Reliability Score (0-1)')
    ax8.set_title('H. Overall Reliability Assessment')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.set_ylim(0, 1.2)
    
    # 数値表示
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # 保存
    filename = os.path.join(save_dir, f"validation_report_{timestamp}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 数値レポートの作成
    report_filename = os.path.join(save_dir, f"validation_summary_{timestamp}.txt")
    with open(report_filename, 'w') as f:
        f.write("ROBUSTNESS VALIDATION STUDY REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write("REPRODUCIBILITY TEST RESULTS (10 runs)\n")
        f.write("-"*40 + "\n")
        f.write(f"Mean SuM distance: {mean_dist:.3f} ± {std_dist:.3f}\n")
        f.write(f"Range: {min_dist:.3f} - {max_dist:.3f}\n")
        f.write(f"Success rate (distance < 0.3): {repro_success_rate:.0f}%\n")
        f.write(f"Mean PD achievement: {np.mean(repro_pd_achievements):.1f}% ± {np.std(repro_pd_achievements):.1f}%\n\n")
        
        if repro_success_rate >= 80:
            f.write("✓ EXCELLENT: Highly reproducible results\n")
        elif repro_success_rate >= 60:
            f.write("~ GOOD: Generally reproducible with some variation\n")
        else:
            f.write("⚠ POOR: High variability in results\n")
        
        f.write(f"\nPARAMETER SENSITIVITY ANALYSIS\n")
        f.write("-"*40 + "\n")
        
        f.write("SuM Convergence Rate Effects:\n")
        for i, result in enumerate(sum_rate_results):
            rate = result['sum_rate']
            dist = result['avg_distance']
            pd = result['avg_pd_achievement']
            f.write(f"  Rate {rate}: distance = {dist:.3f}, PD = {pd:.1f}%\n")
        
        f.write("\nPD Enhancement Factor Effects:\n")
        for i, result in enumerate(pd_factor_results):
            factor = result['pd_factor']
            dist = result['avg_distance']
            pd = result['avg_pd_achievement']
            f.write(f"  Factor {factor}: distance = {dist:.3f}, PD = {pd:.1f}%\n")
        
        # 最適パラメータの推定
        best_sum_rate = min(sum_rate_results, key=lambda x: x['avg_distance'])
        best_pd_factor = min(pd_factor_results, key=lambda x: x['avg_distance'])
        
        f.write(f"\nOPTIMAL PARAMETERS\n")
        f.write("-"*20 + "\n")
        f.write(f"Best SuM rate: {best_sum_rate['sum_rate']} (distance: {best_sum_rate['avg_distance']:.3f})\n")
        f.write(f"Best PD factor: {best_pd_factor['pd_factor']} (distance: {best_pd_factor['avg_distance']:.3f})\n")
        
        f.write(f"\nVALIDATION CONCLUSIONS\n")
        f.write("-"*25 + "\n")
        
        if mean_dist < 0.3 and std_dist < 0.1 and repro_success_rate >= 70:
            f.write("✓ VALIDATED: Results are robust and reproducible\n")
            f.write("✓ Method is reliable for scientific publication\n")
        elif mean_dist < 0.5 and repro_success_rate >= 50:
            f.write("~ PARTIALLY VALIDATED: Results show promise but need refinement\n")
            f.write("~ Additional optimization recommended\n")
        else:
            f.write("⚠ NOT VALIDATED: High variability and inconsistent results\n")
            f.write("⚠ Significant method improvement required\n")
        
        f.write(f"\nRECOMMENDATIONS\n")
        f.write("-"*15 + "\n")
        
        if std_dist > 0.05:
            f.write("• Reduce parameter sensitivity through better calibration\n")
        if repro_success_rate < 80:
            f.write("• Investigate sources of variability\n")
        if np.mean(repro_pd_achievements) < 80:
            f.write("• Consider stronger PD enhancement mechanisms\n")
        
        f.write("• Document limitations in paper\n")
        f.write("• Consider ensemble approaches for robustness\n")
    
    # JSON形式での詳細データ保存
    validation_data = {
        'reproducibility': {
            'results': reproducibility_results,
            'statistics': {
                'mean_distance': float(mean_dist),
                'std_distance': float(std_dist),
                'min_distance': float(min_dist),
                'max_distance': float(max_dist),
                'success_rate': float(repro_success_rate),
                'mean_pd_achievement': float(np.mean(repro_pd_achievements)),
                'std_pd_achievement': float(np.std(repro_pd_achievements))
            }
        },
        'parameter_sensitivity': {
            'sum_rate_results': sum_rate_results,
            'pd_factor_results': pd_factor_results
        },
        'validation_timestamp': timestamp
    }
    
    json_filename = os.path.join(save_dir, f"validation_data_{timestamp}.json")
    with open(json_filename, 'w') as f:
        json.dump(validation_data, f, indent=2, default=str)
    
    logging.info(f"Validation report saved: {filename}")
    logging.info(f"Summary report saved: {report_filename}")
    logging.info(f"Data saved: {json_filename}")
    
    return filename, validation_data

def main():
    """メイン検証実験"""
    
    log_file = setup_logging()
    
    try:
        logging.info("="*60)
        logging.info("ROBUSTNESS VALIDATION STUDY")
        logging.info("="*60)
        logging.info("Testing reproducibility and parameter sensitivity")
        logging.info(f"Log file: {log_file}")
        logging.info("")
        
        # 1. 再現性テスト
        logging.info("Phase 1: Reproducibility testing...")
        reproducibility_results = reproducibility_test()
        
        # 2. パラメータ感度解析
        logging.info("\nPhase 2: Parameter sensitivity analysis...")
        sum_rate_results, pd_factor_results = parameter_sensitivity_analysis()
        
        # 3. 検証レポート作成
        logging.info("\nPhase 3: Creating validation report...")
        figure_file, validation_data = create_validation_report(
            reproducibility_results, sum_rate_results, pd_factor_results
        )
        
        logging.info("="*60)
        logging.info("VALIDATION STUDY COMPLETED!")
        logging.info("="*60)
        
        # 結果サマリー
        repro_stats = validation_data['reproducibility']['statistics']
        
        logging.info("VALIDATION SUMMARY:")
        logging.info(f"Mean SuM distance: {repro_stats['mean_distance']:.3f} ± {repro_stats['std_distance']:.3f}")
        logging.info(f"Success rate (< 0.3): {repro_stats['success_rate']:.0f}%")
        logging.info(f"PD achievement: {repro_stats['mean_pd_achievement']:.1f}%")
        
        # 総合評価
        if (repro_stats['mean_distance'] < 0.3 and 
            repro_stats['std_distance'] < 0.1 and 
            repro_stats['success_rate'] >= 70):
            logging.info("✓ VALIDATION SUCCESSFUL: Method is robust and reproducible")
            validation_status = "PASSED"
        elif repro_stats['mean_distance'] < 0.5 and repro_stats['success_rate'] >= 50:
            logging.info("~ PARTIAL VALIDATION: Method shows promise but needs refinement")
            validation_status = "PARTIAL"
        else:
            logging.info("⚠ VALIDATION FAILED: High variability and inconsistent results")
            validation_status = "FAILED"
        
        logging.info(f"Validation figure: {figure_file}")
        logging.info(f"Validation status: {validation_status}")
        
        return validation_status, validation_data
        
    except Exception as e:
        logging.error(f"Validation study failed: {e}")
        raise

if __name__ == "__main__":
    status, data = main()
