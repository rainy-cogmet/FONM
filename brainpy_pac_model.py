#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于BrainPy的相位-振幅耦合（PAC）模型
替换现有简化模型为生物真实的神经元模型
"""

import numpy as np
import matplotlib.pyplot as plt
import brainpy as bp
import brainpy.math as bm

class PACModel(bp.DynamicalSystem):
    """
    相位-振幅耦合模型
    """
    def __init__(self, neuron_type='izhikevich', pac_strength=0.7):
        super().__init__()
        
        # 初始化神经元模型
        if neuron_type == 'izhikevich':
            self.neuron = bp.neurons.Izhikevich(size=1)
            self.neuron.V_rest = -60.
            self.neuron.V_th = -40.
            self.neuron.a = 0.02
            self.neuron.b = 0.2
            self.neuron.c = -65.
            self.neuron.d = 6.
        elif neuron_type == 'hh':
            self.neuron = bp.neurons.HH(size=1)
        else:
            raise ValueError("Unknown neuron type: " + neuron_type)
        
        # PAC参数
        self.pac_strength = pac_strength
        self.stimulus_phase = 0.0
        self.amplitude_modulation = 1.0
        
        # 记录膜电位历史
        self.membrane_potentials = bm.zeros(1000)  # 保存最近1000ms的数据
        
        # 暴露神经元的spike属性
        self.spike = self.neuron.spike
    
    def update(self, tdi, stimulus):
        """
        更新模型状态
        
        参数:
            stimulus: 外源刺激信号
            
        返回:
            spike: 神经元是否产生动作电位
        """
        # 计算刺激相位
        self.stimulus_phase = bm.arcsin(bm.clip(stimulus, -1, 1))
        
        # 相位-振幅耦合
        # 根据刺激相位调制神经元膜电位振幅
        self.amplitude_modulation = 1.0 + self.pac_strength * bm.cos(self.stimulus_phase)
        self.amplitude_modulation = bm.clip(self.amplitude_modulation, 0.1, 2.0)
        
        # 调整输入电流
        modulated_input = self.amplitude_modulation * bm.abs(stimulus)
        
        # 更新神经元状态
        self.neuron.update(tdi, modulated_input)
        
        # 保存膜电位
        self.membrane_potentials = bm.roll(self.membrane_potentials, -1)
        self.membrane_potentials = self.membrane_potentials.at[-1].set(self.neuron.V[0])
        
        return self.neuron.spike

def generate_stimulus(frequencies, amplitudes, duration=1000.):
    """
    生成多频率叠加的外源刺激信号
    """
    time_points = np.arange(0, duration, 1.)
    stimulus = np.zeros_like(time_points)
    
    for freq, amp in zip(frequencies, amplitudes):
        stimulus += amp * np.sin(2 * np.pi * freq * time_points / 1000.)
    
    # 归一化
    stimulus = stimulus / np.max(np.abs(stimulus)) if np.max(np.abs(stimulus)) > 0 else stimulus
    
    return stimulus, time_points

def calculate_plv(membrane_potentials, stimulus):
    """
    计算相位锁定值
    """
    from scipy.signal import hilbert
    
    # 计算解析信号
    analytic1 = hilbert(membrane_potentials)
    analytic2 = hilbert(stimulus)
    
    # 计算瞬时相位
    phase1 = np.angle(analytic1)
    phase2 = np.angle(analytic2)
    
    # 计算相位差
    phase_diff = phase1 - phase2
    
    # 计算PLV
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
    
    return plv

def decode_10hz_signal(membrane_potentials, threshold=0.5):
    """
    专用10Hz解码模块
    """
    from scipy.signal import hilbert, find_peaks
    
    # 计算膜电位的振幅包络
    analytic_signal = hilbert(membrane_potentials)
    amplitude_envelope = np.abs(analytic_signal)
    
    # 找到超过阈值的时间点
    above_threshold = amplitude_envelope > threshold
    
    # 计算相邻超过阈值事件的时间间隔
    rising_edges = np.where(np.diff(above_threshold.astype(int)) == 1)[0] + 1
    
    # 计算相邻脉冲的时间间隔
    if len(rising_edges) >= 2:
        time_intervals = np.diff(rising_edges)
        
        # 计算预期的10Hz间隔（100ms）
        expected_interval = 100.0
        
        # 检测符合10Hz间隔的脉冲
        valid_pulses = np.where(np.abs(time_intervals - expected_interval) < 20.0)[0]
        
        # 计算解码成功率
        decoding_ratio = len(valid_pulses) / len(time_intervals) if len(time_intervals) > 0 else 0
        decoding_success = decoding_ratio > 0.7  # 超过70%的脉冲符合10Hz间隔则判定为成功
        
        return decoding_success, len(rising_edges), decoding_ratio
    else:
        return False, len(rising_edges), 0.0

def run_experiment(neuron_type='izhikevich'):
    """
    运行PAC实验
    """
    print("=== 运行BrainPy PAC模型实验 ===")
    print(f"神经元类型: {neuron_type}")
    
    # 创建模型
    model = PACModel(neuron_type=neuron_type, pac_strength=0.7)
    
    # --------------------------
    # 条件a：10Hz纯节律刺激
    # --------------------------
    print("\n--- 条件a：10Hz纯节律刺激 ---")
    
    # 生成刺激
    stimulus_a, time_points = generate_stimulus(
        frequencies=[10],
        amplitudes=[1.0],
        duration=1000.
    )
    
    # 运行模拟
    runner = bp.DSRunner(model, monitors=['spike', 'neuron.V', 'neuron.u'])
    runner.run(1000., inputs=[stimulus_a])
    
    # 计算PLV
    plv_a = calculate_plv(runner.mon['neuron.V'], stimulus_a)
    
    # 解码
    decode_success_a, pulse_count_a, decode_ratio_a = decode_10hz_signal(runner.mon['neuron.V'])
    
    print(f"PLV值: {plv_a:.3f}")
    print(f"专用10Hz解码: {'成功' if decode_success_a else '失败'} (脉冲数: {pulse_count_a}, 成功率: {decode_ratio_a:.3f})")
    
    # --------------------------
    # 条件b：10Hz+3Hz叠加刺激
    # --------------------------
    print("\n--- 条件b：10Hz+3Hz叠加刺激 ---")
    
    # 生成刺激
    stimulus_b, time_points = generate_stimulus(
        frequencies=[10, 3],
        amplitudes=[0.8, 0.6],
        duration=1000.
    )
    
    # 运行模拟
    runner = bp.DSRunner(model, monitors=['spike', 'neuron.V', 'neuron.u'])
    runner.run(1000., inputs=[stimulus_b])
    
    # 计算PLV
    plv_b = calculate_plv(runner.mon['neuron.V'], stimulus_b)
    
    # 解码
    decode_success_b, pulse_count_b, decode_ratio_b = decode_10hz_signal(runner.mon['neuron.V'])
    
    print(f"PLV值: {plv_b:.3f}")
    print(f"专用10Hz解码: {'成功' if decode_success_b else '失败'} (脉冲数: {pulse_count_b}, 成功率: {decode_ratio_b:.3f})")
    
    # --------------------------
    # 绘制结果
    # --------------------------
    print("\n--- 绘制实验结果 ---")
    
    # 条件a
    plt.figure(figsize=(16, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(time_points, stimulus_a, label='10Hz Stimulus', alpha=0.7)
    plt.title('Condition a: 10Hz Rhythmic Stimulus')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(time_points, runner.mon['neuron.V'], label='Membrane Potential', alpha=0.7, color='red')
    plt.scatter(time_points[runner.mon['spike']], np.ones_like(time_points[runner.mon['spike']])*50, 
                marker='|', s=200, c='black', label='Spike')
    plt.title('Condition a: Neuron Membrane Potential')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 条件b
    plt.subplot(2, 2, 3)
    plt.plot(time_points, stimulus_b, label='10Hz+3Hz Stimulus', alpha=0.7)
    plt.title('Condition b: 10Hz + 3Hz Combined Stimulus')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot(time_points, runner.mon['neuron.V'], label='Membrane Potential', alpha=0.7, color='red')
    plt.scatter(time_points[runner.mon['spike']], np.ones_like(time_points[runner.mon['spike']])*50, 
                marker='|', s=200, c='black', label='Spike')
    plt.title('Condition b: Neuron Membrane Potential')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('brainpy_pac_experiment.png', dpi=300)
    print("实验结果图已保存为: brainpy_pac_experiment.png")
    
    # 对比柱状图
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # PLV对比
    plv_values = [plv_a, plv_b]
    axes[0].bar(['Condition a', 'Condition b'], plv_values, color=['blue', 'orange'])
    axes[0].set_title('Phase Locking Value (PLV) Comparison')
    axes[0].set_ylabel('PLV (0-1)')
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(plv_values):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    # 解码成功率对比
    decode_ratios = [decode_ratio_a, decode_ratio_b]
    axes[1].bar(['Condition a', 'Condition b'], decode_ratios, color=['blue', 'orange'])
    axes[1].set_title('10Hz Decoding Success Rate')
    axes[1].set_ylabel('Success Rate (0-1)')
    axes[1].set_ylim(0, 1)
    for i, v in enumerate(decode_ratios):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('brainpy_pac_comparison.png', dpi=300)
    print("对比柱状图已保存为: brainpy_pac_comparison.png")
    
    print("\n=== 实验完成 ===")
    return runner

def main():
    """
    主函数
    """
    # 运行Izhikevich模型实验
    runner = run_experiment(neuron_type='izhikevich')
    
    # # 运行HH模型实验
    # runner = run_experiment(neuron_type='hh')

if __name__ == "__main__":
    main()