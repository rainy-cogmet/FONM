#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于BrainPy的相位-振幅耦合（PAC）模型 - V3版本
在工作的基础神经元模型上添加PAC机制
"""

import numpy as np
import matplotlib.pyplot as plt
import brainpy as bp
import brainpy.math as bm

def generate_pac_stimulus(phase_freq, amp_freq, phase_amp=1.0, amp_amp=0.5, duration=1000.):
    """
    生成相位-振幅耦合的刺激信号
    
    参数:
        phase_freq: 相位调制频率（Hz）
        amp_freq: 振幅调制频率（Hz）
        phase_amp: 相位信号幅度
        amp_amp: 振幅调制强度
        duration: 持续时间（ms）
        
    返回:
        stimulus: PAC刺激信号
        phase_signal: 相位调制信号
        amp_signal: 振幅调制信号
        time_points: 时间点数组
    """
    time_points = np.arange(0, duration, 1.)
    
    # 生成相位调制信号（低频）
    phase_signal = phase_amp * np.sin(2 * np.pi * phase_freq * time_points / 1000.)
    
    # 生成振幅调制信号（高频）
    amp_signal = 1.0 + amp_amp * np.cos(2 * np.pi * phase_freq * time_points / 1000.)
    amp_signal = np.clip(amp_signal, 0.1, 2.0)
    
    # 生成PAC刺激信号
    stimulus = amp_signal * np.sin(2 * np.pi * amp_freq * time_points / 1000.)
    
    # 放大刺激强度以确保神经元响应
    stimulus *= 50.0
    
    return stimulus, phase_signal, amp_signal, time_points

def run_pac_experiment(phase_freq, amp_freq, phase_amp=1.0, amp_amp=0.5):
    """
    运行PAC实验
    """
    print(f"\n=== 运行PAC实验 ({phase_freq}Hz相位调制{amp_freq}Hz振幅) ===")
    
    # 创建Izhikevich神经元
    izh = bp.neurons.Izhikevich(size=1, a=0.02, b=0.2, c=-65., d=6.)
    
    # 生成PAC刺激
    stimulus, phase_signal, amp_signal, time_points = generate_pac_stimulus(
        phase_freq=phase_freq,
        amp_freq=amp_freq,
        phase_amp=phase_amp,
        amp_amp=amp_amp,
        duration=1000.
    )
    
    # 运行模拟
    runner = bp.DSRunner(izh, monitors=['V', 'spike'], jit=True)
    runner.run(inputs=stimulus)
    
    # 检测动作电位
    spikes = time_points[runner.mon['spike'][:, 0] > 0.5]
    
    print(f"检测到的动作电位数量: {len(spikes)}")
    print(f"平均脉冲频率: {len(spikes)/1.0:.2f} Hz")
    
    # 计算PLV
    plv_value = calculate_plv(runner.mon['V'][:, 0], stimulus)
    print(f"相位锁定值 (PLV): {plv_value:.3f}")
    
    # 绘制结果
    plt.figure(figsize=(14, 8))
    
    plt.subplot(311)
    plt.plot(time_points, phase_signal, label=f'{phase_freq}Hz Phase Signal', color='blue', alpha=0.7)
    plt.plot(time_points, amp_signal - 1.0, label=f'{phase_freq}Hz Amplitude Modulation', color='green', alpha=0.7)
    plt.title(f'Phase-Amplitude Coupling Stimulus ({phase_freq}Hz phase modulates {amp_freq}Hz amplitude)')
    plt.ylabel('Amplitude')
    plt.legend()
    
    plt.subplot(312)
    plt.plot(time_points, stimulus, label='PAC Input Current', color='purple', alpha=0.7)
    plt.ylabel('Input Current (mA)')
    plt.legend()
    
    plt.subplot(313)
    plt.plot(time_points, runner.mon['V'][:, 0], label='Membrane Potential', color='red', alpha=0.7)
    plt.scatter(spikes, np.ones_like(spikes)*(-60), marker='|', s=200, c='black', label='Spikes')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'brainpy_pac_{phase_freq}hz_{amp_freq}hz.png', dpi=300)
    print(f"PAC实验结果已保存为: brainpy_pac_{phase_freq}hz_{amp_freq}hz.png")
    
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

def decode_10hz_signal(membrane_potentials, threshold=-60.0):
    """
    解码10Hz节律信号
    """
    from scipy.signal import find_peaks
    
    # 找到膜电位的峰值
    peaks, _ = find_peaks(membrane_potentials, height=threshold)
    
    if len(peaks) >= 2:
        # 计算峰间间隔
        intervals = np.diff(peaks)
        
        # 预期的10Hz间隔（100ms）
        expected_interval = 100.0
        
        # 计算符合10Hz间隔的比例
        valid_ratio = np.mean(np.abs(intervals - expected_interval) < 20.0)
        
        return len(peaks), valid_ratio
    else:
        return len(peaks), 0.0

def main():
    """
    主函数
    """
    print("=== BrainPy相位-振幅耦合模型实验 ===")
    
    # 实验1：10Hz相位调制3Hz振幅
    print("\n--- 实验1：10Hz相位调制3Hz振幅（条件b） ---")
    izh1, runner1, spikes1, plv1, stimulus1, v1 = run_pac_experiment(
        phase_freq=10,
        amp_freq=3,
        phase_amp=1.0,
        amp_amp=0.7
    )
    
    # 实验2：3Hz相位调制10Hz振幅（反向PAC）
    print("\n--- 实验2：3Hz相位调制10Hz振幅（类似条件a） ---")
    izh2, runner2, spikes2, plv2, stimulus2, v2 = run_pac_experiment(
        phase_freq=3,
        amp_freq=10,
        phase_amp=1.0,
        amp_amp=0.7
    )
    
    # 实验3：纯10Hz节律刺激（条件a）
    print("\n--- 实验3：纯10Hz节律刺激（条件a） ---")
    izh3 = bp.neurons.Izhikevich(size=1, a=0.02, b=0.2, c=-65., d=6.)
    time_points = np.arange(0, 1000, 1.)
    stimulus3 = 50.0 * np.sin(2 * np.pi * 10 * time_points / 1000.)
    
    runner3 = bp.DSRunner(izh3, monitors=['V', 'spike'], jit=True)
    runner3.run(inputs=stimulus3)
    
    spikes3 = time_points[runner3.mon['spike'][:, 0] > 0.5]
    plv3 = calculate_plv(runner3.mon['V'][:, 0], stimulus3)
    
    print(f"检测到的动作电位数量: {len(spikes3)}")
    print(f"平均脉冲频率: {len(spikes3)/1.0:.2f} Hz")
    print(f"相位锁定值 (PLV): {plv3:.3f}")
    
    # 绘制实验3结果
    plt.figure(figsize=(12, 6))
    
    plt.subplot(211)
    plt.plot(time_points, stimulus3, label='10Hz Rhythmic Stimulus', color='blue', alpha=0.7)
    plt.title('10Hz Rhythmic Stimulus (Condition a)')
    plt.ylabel('Input Current (mA)')
    plt.legend()
    
    plt.subplot(212)
    plt.plot(time_points, runner3.mon['V'][:, 0], label='Membrane Potential', color='red', alpha=0.7)
    plt.scatter(spikes3, np.ones_like(spikes3)*(-60), marker='|', s=200, c='black', label='Spikes')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('brainpy_pac_10hz_pure.png', dpi=300)
    print(f"实验3结果已保存为: brainpy_pac_10hz_pure.png")
    
    # --------------------------
    # 绘制对比柱状图
    # --------------------------
    print("\n--- 绘制对比柱状图 ---")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PLV对比
    plv_values = [plv3, plv1, plv2]
    axes[0].bar(['10Hz pure', '10Hz phase', '3Hz phase'], plv_values, color=['blue', 'orange', 'green'])
    axes[0].set_title('Phase Locking Value (PLV) Comparison')
    axes[0].set_ylabel('PLV (0-1)')
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(plv_values):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    # 脉冲数对比
    spike_counts = [len(spikes3), len(spikes1), len(spikes2)]
    axes[1].bar(['10Hz pure', '10Hz phase', '3Hz phase'], spike_counts, color=['blue', 'orange', 'green'])
    axes[1].set_title('Number of Spikes Comparison')
    axes[1].set_ylabel('Number of Spikes')
    axes[1].set_ylim(0, max(spike_counts)*1.2 if spike_counts else 10)
    for i, v in enumerate(spike_counts):
        axes[1].text(i, v + 0.5, f'{v:d}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('brainpy_pac_comparison_v3.png', dpi=300)
    print(f"对比柱状图已保存为: brainpy_pac_comparison_v3.png")
    
    # --------------------------
    # 解码分析
    # --------------------------
    print("\n--- 解码分析 ---")
    
    print("\n条件a（纯10Hz刺激）解码:")
    peak_count, valid_ratio = decode_10hz_signal(runner3.mon['V'][:, 0])
    print(f"峰值数量: {peak_count}, 符合10Hz间隔的比例: {valid_ratio:.3f}")
    print(f"解码结果: {'成功' if valid_ratio > 0.7 else '失败'}")
    
    print("\n条件b（10Hz相位调制3Hz振幅）解码:")
    peak_count_b, valid_ratio_b = decode_10hz_signal(runner1.mon['V'][:, 0])
    print(f"峰值数量: {peak_count_b}, 符合10Hz间隔的比例: {valid_ratio_b:.3f}")
    print(f"解码结果: {'成功' if valid_ratio_b > 0.7 else '失败'}")
    
    print("\n实验2（3Hz相位调制10Hz振幅）解码:")
    peak_count_2, valid_ratio_2 = decode_10hz_signal(runner2.mon['V'][:, 0])
    print(f"峰值数量: {peak_count_2}, 符合10Hz间隔的比例: {valid_ratio_2:.3f}")
    print(f"解码结果: {'成功' if valid_ratio_2 > 0.7 else '失败'}")
    
    print("\n=== 所有实验完成 ===")

if __name__ == "__main__":
    main()