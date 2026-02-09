#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于BrainPy的MT区gamma振荡模拟模型
"""

import numpy as np
import matplotlib.pyplot as plt
import brainpy as bp
import brainpy.math as bm

class MTNeuronModel:
    """
    MT区神经元模型，模拟gamma振荡特性
    """
    def __init__(self, gamma_freq=60., neuron_params=None):
        """
        初始化MT区神经元模型
        
        参数:
            gamma_freq: 目标gamma振荡频率 (Hz)
            neuron_params: 神经元参数
        """
        # 设置默认神经元参数，针对MT区gamma振荡
        if neuron_params is None:
            neuron_params = {
                'a': 0.02,    # 恢复时间常数
                'b': 0.2,     # 恢复敏感度
                'c': -65.,    # 重置膜电位
                'd': 6.0,     # 恢复增量
            }
            
            # 根据目标gamma频率调整参数
            if gamma_freq < 50:
                neuron_params['a'] = 0.03
                neuron_params['b'] = 0.25
                neuron_params['d'] = 4.0
            elif gamma_freq > 80:
                neuron_params['a'] = 0.01
                neuron_params['b'] = 0.15
                neuron_params['d'] = 8.0
                
        self.neuron = bp.neurons.Izhikevich(size=1, **neuron_params)
        self.gamma_freq = gamma_freq
        
    def run_simulation(self, input_current=50., duration=1000.):
        """
        运行模拟
        
        参数:
            input_current: 输入电流 (mA)
            duration: 持续时间 (ms)
            
        返回:
            runner: 模拟运行器
        """
        # 生成输入电流
        time_points = np.arange(0, duration, 1.)
        if isinstance(input_current, (int, float)):
            current = np.ones_like(time_points) * input_current
        else:
            current = input_current
        
        # 运行模拟
        runner = bp.DSRunner(self.neuron, monitors=['V', 'spike'], jit=True)
        runner.run(inputs=current)
        
        return runner
    
    def analyze_oscillations(self, runner):
        """
        分析振荡特性
        
        参数:
            runner: 模拟运行器
            
        返回:
            analysis_results: 分析结果字典
        """
        # 检测动作电位
        spikes = runner.mon['spike'][:, 0] > 0.5
        spike_times = runner.mon['ts'][spikes]
        
        # 计算振荡频率
        if len(spike_times) >= 2:
            inter_spike_intervals = np.diff(spike_times)
            avg_frequency = 1000. / np.mean(inter_spike_intervals) if np.mean(inter_spike_intervals) > 0 else 0
        else:
            avg_frequency = 0.
            inter_spike_intervals = np.array([])
        
        # 计算功率谱
        from scipy.signal import welch
        f, Pxx = welch(runner.mon['V'][:, 0], fs=1000.)
        
        # 找到gamma频段的峰值
        gamma_mask = (f >= 30) & (f <= 110)
        gamma_frequencies = f[gamma_mask]
        gamma_power = Pxx[gamma_mask]
        
        if len(gamma_power) > 0:
            gamma_peak_freq = gamma_frequencies[np.argmax(gamma_power)] if len(gamma_power) > 0 else 0
            gamma_peak_power = np.max(gamma_power) if len(gamma_power) > 0 else 0
        else:
            gamma_peak_freq = 0
            gamma_peak_power = 0
        
        # 计算PLV值（自相位锁定）
        plv_value = calculate_plv(runner.mon['V'][:, 0], runner.mon['V'][:, 0])
        
        analysis_results = {
            'spike_count': len(spike_times),
            'avg_frequency': avg_frequency,
            'spike_intervals': inter_spike_intervals,
            'gamma_peak_freq': gamma_peak_freq,
            'gamma_peak_power': gamma_peak_power,
            'gamma_power_spectrum': (gamma_frequencies, gamma_power),
            'plv_value': plv_value
        }
        
        return analysis_results
    
    def plot_results(self, runner, analysis_results):
        """
        绘制结果
        
        参数:
            runner: 模拟运行器
            analysis_results: 分析结果
        """
        plt.figure(figsize=(14, 10))
        
        # 绘制膜电位
        plt.subplot(311)
        plt.plot(runner.mon['ts'], runner.mon['V'][:, 0], label='Membrane Potential', color='red', alpha=0.7)
        spikes = runner.mon['ts'][runner.mon['spike'][:, 0] > 0.5]
        plt.scatter(spikes, np.ones_like(spikes)*(-60), marker='|', s=200, c='black', label='Spikes')
        plt.title(f'MT区神经元gamma振荡模拟 (目标频率: {self.gamma_freq}Hz)')
        plt.ylabel('Membrane Potential (mV)')
        plt.legend()
        
        # 绘制功率谱
        plt.subplot(312)
        f, Pxx = analysis_results['gamma_power_spectrum']
        plt.plot(f, Pxx, label='Gamma Band Power Spectrum', color='blue', alpha=0.7)
        plt.axvline(analysis_results['gamma_peak_freq'], color='red', linestyle='--', 
                   label=f'Peak Frequency: {analysis_results["gamma_peak_freq"]:.1f}Hz')
        plt.title('Gamma Band Power Spectrum')
        plt.ylabel('Power')
        plt.legend()
        
        # 绘制ISI直方图
        plt.subplot(313)
        if len(analysis_results['spike_intervals']) > 0:
            plt.hist(analysis_results['spike_intervals'], bins=20, alpha=0.7, color='green', edgecolor='black')
            plt.axvline(1000./self.gamma_freq, color='red', linestyle='--', 
                       label=f'Target ISI: {1000./self.gamma_freq:.1f}ms')
            plt.title('Inter-Spike Interval (ISI) Histogram')
            plt.xlabel('ISI (ms)')
            plt.ylabel('Count')
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No spikes detected', ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
            plt.title('Inter-Spike Interval (ISI) Histogram')
            plt.xlabel('ISI (ms)')
            plt.ylabel('Count')
        
        plt.tight_layout()
        filename = f'mt_gamma_{self.gamma_freq}hz_simulation.png'
        plt.savefig(filename, dpi=300)
        print(f"模拟结果已保存为: {filename}")
        
        return filename

def calculate_plv(signal1, signal2):
    """
    计算相位锁定值（PLV）
    """
    from scipy.signal import hilbert
    
    # 计算解析信号
    analytic1 = hilbert(signal1)
    analytic2 = hilbert(signal2)
    
    # 计算瞬时相位
    phase1 = np.angle(analytic1)
    phase2 = np.angle(analytic2)
    
    # 计算相位差
    phase_diff = phase1 - phase2
    
    # 计算PLV
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
    
    return plv

def optimize_neuron_parameters(target_freq, input_current=50., duration=1000.):
    """
    优化神经元参数以达到目标频率
    
    参数:
        target_freq: 目标频率 (Hz)
        input_current: 输入电流 (mA)
        duration: 持续时间 (ms)
        
    返回:
        best_model: 最优模型
        best_results: 最优结果
    """
    print(f"=== 优化神经元参数以达到{target_freq}Hz gamma振荡 ===")
    
    # 定义参数搜索空间
    param_space = [
        {'a': 0.01, 'b': 0.15, 'c': -65., 'd': 8.0},
        {'a': 0.02, 'b': 0.2, 'c': -65., 'd': 6.0},
        {'a': 0.03, 'b': 0.25, 'c': -65., 'd': 4.0},
        {'a': 0.01, 'b': 0.2, 'c': -65., 'd': 6.0},
        {'a': 0.02, 'b': 0.15, 'c': -65., 'd': 8.0},
        {'a': 0.01, 'b': 0.25, 'c': -65., 'd': 4.0},
    ]
    
    best_score = float('inf')
    best_model = None
    best_results = None
    
    for i, params in enumerate(param_space):
        print(f"\n尝试参数组合 {i+1}/{len(param_space)}: {params}")
        
        model = MTNeuronModel(gamma_freq=target_freq, neuron_params=params)
        runner = model.run_simulation(input_current=input_current, duration=duration)
        results = model.analyze_oscillations(runner)
        
        print(f"实际频率: {results['avg_frequency']:.1f}Hz, 峰值频率: {results['gamma_peak_freq']:.1f}Hz")
        print(f"PLV值: {results['plv_value']:.3f}")
        
        # 计算频率误差
        freq_error = abs(results['avg_frequency'] - target_freq)
        
        # 选择最优模型
        if freq_error < best_score:
            best_score = freq_error
            best_model = model
            best_results = results
            print(f"新的最优模型，频率误差: {best_score:.1f}Hz")
    
    print(f"\n=== 优化完成 ===")
    print(f"目标频率: {target_freq}Hz")
    print(f"最优频率: {best_results['avg_frequency']:.1f}Hz")
    print(f"频率误差: {best_score:.1f}Hz")
    print(f"最优参数: {best_model.neuron.a}, {best_model.neuron.b}, {best_model.neuron.c}, {best_model.neuron.d}")
    
    return best_model, best_results

def main():
    """
    主函数
    """
    print("=== MT区gamma振荡模拟 ===")
    
    # 模拟不同频率的gamma振荡
    target_frequencies = [40, 60, 80, 100]  # MT区gamma振荡范围：35-110Hz
    
    for target_freq in target_frequencies:
        print(f"\n--- 模拟{target_freq}Hz gamma振荡 ---")
        
        # 初始化模型
        model = MTNeuronModel(gamma_freq=target_freq)
        
        # 运行模拟
        runner = model.run_simulation(input_current=50., duration=1000.)
        
        # 分析结果
        results = model.analyze_oscillations(runner)
        
        # 打印分析结果
        print(f"动作电位数量: {results['spike_count']}")
        print(f"平均频率: {results['avg_frequency']:.1f}Hz")
        print(f"峰值频率: {results['gamma_peak_freq']:.1f}Hz")
        print(f"PLV值: {results['plv_value']:.3f}")
        
        # 绘制结果
        filename = model.plot_results(runner, results)
    
    # 针对60Hz进行参数优化
    print("\n--- 针对60Hz进行参数优化 ---")
    best_model, best_results = optimize_neuron_parameters(target_freq=60., input_current=50., duration=1000.)
    
    # 绘制最优模型结果
    runner = best_model.run_simulation(input_current=50., duration=1000.)
    filename = best_model.plot_results(runner, best_results)
    
    print("\n=== 所有模拟完成 ===")

if __name__ == "__main__":
    main()