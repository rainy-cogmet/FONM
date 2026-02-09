#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于BrainPy的相位-振幅耦合（PAC）模型 - V4版本
测试10Hz和3Hz振幅的直接叠加刺激
"""

import numpy as np
import matplotlib.pyplot as plt
import brainpy as bp
import brainpy.math as bm

def generate_stimulus(frequencies, amplitudes, duration=1000.):
    """
    生成多频率叠加的刺激信号
    
    参数:
        frequencies: 频率列表（Hz）
        amplitudes: 振幅列表
        duration: 持续时间（ms）
        
    返回:
        stimulus: 刺激信号
        time_points: 时间点数组
        component_signals: 各个频率分量信号
    """
    time_points = np.arange(0, duration, 1.)
    
    component_signals = []
    stimulus = np.zeros_like(time_points)
    
    for freq, amp in zip(frequencies, amplitudes):
        signal = amp * np.sin(2 * np.pi * freq * time_points / 1000.)
        component_signals.append(signal)
        stimulus += signal
    
    return stimulus, time_points, component_signals

def run_experiment(experiment_name, frequencies, amplitudes, neuron_params=None):
    """
    运行实验
    """
    print(f"\n--- {experiment_name} ---")
    
    # 创建神经元
    if neuron_params is None:
        neuron_params = {'a': 0.02, 'b': 0.2, 'c': -65., 'd': 6.}
    
    izh = bp.neurons.Izhikevich(size=1, **neuron_params)
    
    # 生成刺激
    stimulus, time_points, component_signals = generate_stimulus(
        frequencies=frequencies,
        amplitudes=amplitudes,
        duration=1000.
    )
    
    # 运行模拟
    runner = bp.DSRunner(izh, monitors=['V', 'spike'], jit=True)
    runner.run(inputs=stimulus)
    
    # 分析结果
    spikes = time_points[runner.mon['spike'][:, 0] > 0.5]
    plv_value = calculate_plv(runner.mon['V'][:, 0], stimulus)
    
    print(f"检测到的动作电位数量: {len(spikes)}")
    print(f"平均脉冲频率: {len(spikes)/1.0:.2f} Hz")
    print(f"相位锁定值 (PLV): {plv_value:.3f}")
    
    # 绘制结果
    num_components = len(frequencies)
    plt.figure(figsize=(14, 6 + 2 * num_components))
    
    # 绘制各个频率分量
    for i in range(num_components):
        plt.subplot(num_components + 2, 1, i + 1)
        plt.plot(time_points, component_signals[i], label=f'{frequencies[i]}Hz Component', alpha=0.7)
        plt.title(f'{frequencies[i]}Hz Component Signal')
        plt.ylabel('Amplitude')
        plt.legend()
    
    # 绘制总刺激信号
    plt.subplot(num_components + 2, 1, num_components + 1)
    plt.plot(time_points, stimulus, label='Total Stimulus', color='purple', alpha=0.7)
    plt.title('Total Stimulus Signal')
    plt.ylabel('Input Current (mA)')
    plt.legend()
    
    # 绘制神经元响应
    plt.subplot(num_components + 2, 1, num_components + 2)
    plt.plot(time_points, runner.mon['V'][:, 0], label='Membrane Potential', color='red', alpha=0.7)
    plt.scatter(spikes, np.ones_like(spikes)*(-60), marker='|', s=200, c='black', label='Spikes')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.legend()
    
    plt.tight_layout()
    filename = experiment_name.lower().replace(' ', '_') + '.png'
    plt.savefig(filename, dpi=300)
    print(f"实验结果已保存为: {filename}")
    
    return izh, runner, spikes, plv_value, stimulus, runner.mon['V'][:, 0]

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

def decode_specific_frequency(membrane_potentials, target_freq=10.0, tolerance=0.2):
    """
    解码特定频率的节律信号
    """
    from scipy.signal import find_peaks
    
    # 找到膜电位的峰值
    peaks, _ = find_peaks(membrane_potentials, height=-60.0)
    
    if len(peaks) >= 2:
        # 计算峰间间隔
        intervals = np.diff(peaks)
        
        # 预期的间隔
        expected_interval = 1000.0 / target_freq
        
        # 计算符合间隔的比例
        valid_ratio = np.mean(np.abs(intervals - expected_interval) < expected_interval * tolerance)
        
        return len(peaks), valid_ratio, intervals
    else:
        return len(peaks), 0.0, np.array([])

def main():
    """
    主函数
    """
    print("=== BrainPy多频率叠加刺激实验 ===")
    
    # 条件a：纯10Hz节律刺激
    print("\n--- 条件a：纯10Hz节律刺激 ---")
    izh_a, runner_a, spikes_a, plv_a, stimulus_a, v_a = run_experiment(
        experiment_name="Condition a - 10Hz only",
        frequencies=[10],
        amplitudes=[50.0]  # 刺激强度
    )
    
    # 条件b：10Hz和3Hz直接叠加
    print("\n--- 条件b：10Hz和3Hz直接叠加 ---")
    izh_b, runner_b, spikes_b, plv_b, stimulus_b, v_b = run_experiment(
        experiment_name="Condition b - 10Hz + 3Hz combined",
        frequencies=[10, 3],
        amplitudes=[40.0, 30.0]  # 10Hz振幅略大于3Hz
    )
    
    # 条件c：10Hz和3Hz等振幅叠加
    print("\n--- 条件c：10Hz和3Hz等振幅叠加 ---")
    izh_c, runner_c, spikes_c, plv_c, stimulus_c, v_c = run_experiment(
        experiment_name="Condition c - 10Hz + 3Hz equal amplitude",
        frequencies=[10, 3],
        amplitudes=[40.0, 40.0]  # 等振幅
    )
    
    # 条件d：只3Hz节律刺激（对照）
    print("\n--- 条件d：只3Hz节律刺激（对照） ---")
    izh_d, runner_d, spikes_d, plv_d, stimulus_d, v_d = run_experiment(
        experiment_name="Condition d - 3Hz only",
        frequencies=[3],
        amplitudes=[50.0]
    )
    
    # --------------------------
    # 绘制对比柱状图
    # --------------------------
    print("\n--- 绘制对比柱状图 ---")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PLV对比
    plv_values = [plv_a, plv_b, plv_c, plv_d]
    labels = ['10Hz only', '10+3Hz (a>3)', '10+3Hz (equal)', '3Hz only']
    colors = ['blue', 'orange', 'green', 'red']
    
    axes[0].bar(labels, plv_values, color=colors)
    axes[0].set_title('Phase Locking Value (PLV) Comparison')
    axes[0].set_ylabel('PLV (0-1)')
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(plv_values):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    # 脉冲数对比
    spike_counts = [len(spikes_a), len(spikes_b), len(spikes_c), len(spikes_d)]
    axes[1].bar(labels, spike_counts, color=colors)
    axes[1].set_title('Number of Spikes Comparison')
    axes[1].set_ylabel('Number of Spikes')
    axes[1].set_ylim(0, max(spike_counts)*1.2 if spike_counts else 10)
    for i, v in enumerate(spike_counts):
        axes[1].text(i, v + 0.5, f'{v:d}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('brainpy_multi_frequency_comparison.png', dpi=300)
    print(f"对比柱状图已保存为: brainpy_multi_frequency_comparison.png")
    
    # --------------------------
    # 解码分析
    # --------------------------
    print("\n--- 解码分析 ---")
    
    print("\n条件a（纯10Hz刺激）解码:")
    peak_count_a, valid_ratio_a, intervals_a = decode_specific_frequency(v_a, target_freq=10.0)
    print(f"峰值数量: {peak_count_a}, 符合10Hz间隔的比例: {valid_ratio_a:.3f}")
    print(f"解码结果: {'成功' if valid_ratio_a > 0.7 else '失败'}")
    
    print("\n条件b（10+3Hz）解码10Hz:")
    peak_count_b, valid_ratio_b, intervals_b = decode_specific_frequency(v_b, target_freq=10.0)
    print(f"峰值数量: {peak_count_b}, 符合10Hz间隔的比例: {valid_ratio_b:.3f}")
    print(f"解码结果: {'成功' if valid_ratio_b > 0.7 else '失败'}")
    
    print("\n条件b（10+3Hz）解码3Hz:")
    peak_count_b3, valid_ratio_b3, intervals_b3 = decode_specific_frequency(v_b, target_freq=3.0)
    print(f"峰值数量: {peak_count_b3}, 符合3Hz间隔的比例: {valid_ratio_b3:.3f}")
    print(f"解码结果: {'成功' if valid_ratio_b3 > 0.7 else '失败'}")
    
    print("\n=== 所有实验完成 ===")

if __name__ == "__main__":
    main()