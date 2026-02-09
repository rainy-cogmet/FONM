#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BrainPy基础神经元模拟示例 - 简化版
使用更简单的输入方式，避免JAX的tracer问题
"""

import numpy as np
import matplotlib.pyplot as plt
import brainpy as bp
import brainpy.math as bm

def test_hh_neuron():
    """
    测试Hodgkin-Huxley神经元模型
    """
    print("=== 测试Hodgkin-Huxley神经元模型 ===")
    
    # 创建HH神经元
    hh = bp.neurons.HH(size=1)
    
    # 直接生成输入电流数组
    time_points = np.arange(0, 500, 1.)
    input_current = np.zeros_like(time_points)
    input_current[(time_points >= 100) & (time_points < 200)] = 5.
    input_current[(time_points >= 300) & (time_points < 400)] = 10.
    
    # 运行模拟
    runner = bp.DSRunner(hh, monitors=['V', 'spike', 'm', 'h', 'n'], jit=True)
    runner.run(inputs=input_current)
    
    # 绘制结果
    plt.figure(figsize=(12, 8))
    
    plt.subplot(311)
    plt.plot(time_points, runner.mon['V'][:, 0], label='HH Neuron', color='red')
    spikes = time_points[runner.mon['spike'][:, 0] > 0.5]
    plt.scatter(spikes, np.ones_like(spikes)*(-60), marker='|', s=200, c='black', label='Spikes')
    plt.title('Hodgkin-Huxley Neuron Model')
    plt.ylabel('Membrane Potential (mV)')
    plt.legend()
    
    plt.subplot(312)
    plt.plot(time_points, runner.mon['m'][:, 0], label='m (Na+ activation)')
    plt.plot(time_points, runner.mon['h'][:, 0], label='h (Na+ inactivation)')
    plt.plot(time_points, runner.mon['n'][:, 0], label='n (K+ activation)')
    plt.ylabel('Gating Variables')
    plt.legend()
    
    plt.subplot(313)
    plt.plot(time_points, input_current, label='Input Current', color='blue')
    plt.xlabel('Time (ms)')
    plt.ylabel('Input Current (mA/cm²)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('brainpy_hh_neuron_example_simple.png', dpi=300)
    print("HH神经元示例结果已保存为: brainpy_hh_neuron_example_simple.png")
    
    print(f"\n检测到的动作电位数量: {len(spikes)}")
    if len(spikes) > 0:
        print(f"神经元最大膜电位: {np.max(runner.mon['V'][0]):.2f} mV")
        print(f"神经元最小膜电位: {np.min(runner.mon['V'][0]):.2f} mV")
    else:
        print("未检测到动作电位，可能需要增加刺激强度")
    
    return hh, runner, spikes

def test_izhikevich_neuron():
    """
    测试Izhikevich神经元模型
    """
    print("\n=== 测试Izhikevich神经元模型 ===")
    
    # 创建Izhikevich神经元
    izh = bp.neurons.Izhikevich(size=1, a=0.02, b=0.2, c=-65., d=6.)
    
    # 直接生成输入电流数组
    time_points = np.arange(0, 500, 1.)
    input_current = np.zeros_like(time_points)
    input_current[(time_points >= 100) & (time_points < 200)] = 50.
    input_current[(time_points >= 300) & (time_points < 400)] = 100.
    
    # 运行模拟
    runner = bp.DSRunner(izh, monitors=['V', 'spike', 'u'], jit=True)
    runner.run(inputs=input_current)
    
    # 绘制结果
    plt.figure(figsize=(12, 8))
    
    plt.subplot(311)
    plt.plot(time_points, runner.mon['V'][:, 0], label='Izhikevich Neuron', color='red')
    spikes = time_points[runner.mon['spike'][:, 0] > 0.5]
    plt.scatter(spikes, np.ones_like(spikes)*50, marker='|', s=200, c='black', label='Spikes')
    plt.title('Izhikevich Neuron Model')
    plt.ylabel('Membrane Potential (mV)')
    plt.legend()
    
    plt.subplot(312)
    plt.plot(time_points, runner.mon['u'][:, 0], label='u (Recovery variable)', color='green')
    plt.ylabel('Recovery Variable')
    plt.legend()
    
    plt.subplot(313)
    plt.plot(time_points, input_current, label='Input Current', color='blue')
    plt.xlabel('Time (ms)')
    plt.ylabel('Input Current (mA)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('brainpy_izhikevich_neuron_example_simple.png', dpi=300)
    print("Izhikevich神经元示例结果已保存为: brainpy_izhikevich_neuron_example_simple.png")
    
    print(f"\n检测到的动作电位数量: {len(spikes)}")
    if len(spikes) > 0:
        print(f"神经元最大膜电位: {np.max(runner.mon['V'][0]):.2f} mV")
        print(f"神经元最小膜电位: {np.min(runner.mon['V'][0]):.2f} mV")
    else:
        print("未检测到动作电位，可能需要增加刺激强度")
    
    return izh, runner, spikes

def test_rhythmic_stimulus():
    """
    测试节律性刺激下的神经元响应
    """
    print("\n=== 测试节律性刺激下的神经元响应 ===")
    
    # 创建Izhikevich神经元
    izh = bp.neurons.Izhikevich(size=1, a=0.02, b=0.2, c=-65., d=6.)
    
    # 生成节律性刺激
    freq = 10.0  # 10Hz
    amplitude = 50.0  # 刺激强度
    time_points = np.arange(0, 1000, 1.)
    input_current = amplitude * np.sin(2 * np.pi * freq * time_points / 1000.)
    
    # 运行模拟
    runner = bp.DSRunner(izh, monitors=['V', 'spike'], jit=True)
    runner.run(inputs=input_current)
    
    # 绘制结果
    plt.figure(figsize=(12, 6))
    
    plt.subplot(211)
    plt.plot(time_points, input_current, label='10Hz Stimulus', color='blue')
    plt.title('10Hz Rhythmic Stimulus')
    plt.ylabel('Input Current (mA)')
    plt.legend()
    
    plt.subplot(212)
    plt.plot(time_points, runner.mon['V'][:, 0], label='Izhikevich Neuron', color='red')
    spikes = time_points[runner.mon['spike'][:, 0] > 0.5]
    plt.scatter(spikes, np.ones_like(spikes)*50, marker='|', s=200, c='black', label='Spikes')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('brainpy_rhythmic_stimulus_simple.png', dpi=300)
    print("节律性刺激测试结果已保存为: brainpy_rhythmic_stimulus_simple.png")
    
    print(f"\n检测到的动作电位数量: {len(spikes)}")
    if len(spikes) > 0:
        print(f"平均脉冲频率: {len(spikes)/1.0:.2f} Hz")
        print(f"神经元最大膜电位: {np.max(runner.mon['V'][0]):.2f} mV")
    else:
        print("未检测到动作电位，可能需要增加刺激强度或调整神经元参数")
    
    return izh, runner, spikes

def main():
    """
    主函数
    """
    print("=== BrainPy基础神经元模拟示例（简化版） ===\n")
    
    # 测试HH神经元
    _, _, hh_spikes = test_hh_neuron()
    
    # 测试Izhikevich神经元
    _, _, izh_spikes = test_izhikevich_neuron()
    
    # 测试节律性刺激
    _, _, rhythmic_spikes = test_rhythmic_stimulus()
    
    print("\n=== 所有测试完成 ===")
    print(f"\n总结:")
    print(f"HH神经元检测到动作电位: {len(hh_spikes)} 个")
    print(f"Izhikevich神经元检测到动作电位: {len(izh_spikes)} 个")
    print(f"10Hz节律刺激下检测到动作电位: {len(rhythmic_spikes)} 个")

if __name__ == "__main__":
    main()