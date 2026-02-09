#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BrainPy基础神经元模拟示例
从官方示例开始，确保能够正确模拟神经元的动作电位产生
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
    
    # 定义输入电流
    def input_current():
        t = bp.share['t']
        if 100 < t < 200:
            return 5.
        elif 300 < t < 400:
            return 10.
        else:
            return 0.
    
    # 运行模拟
    runner = bp.DSRunner(hh, monitors=['V', 'spike', 'm', 'h', 'n'], inputs=input_current, jit=True)
    runner.run(500.)
    
    # 绘制结果
    plt.figure(figsize=(12, 8))
    
    plt.subplot(311)
    plt.plot(runner.mon.ts, runner.mon.V[0], label='HH Neuron', color='red')
    spikes = runner.mon.ts[runner.mon['spike'].flatten() > 0.5]
    plt.scatter(spikes, np.ones_like(spikes)*(-60), marker='|', s=200, c='black', label='Spikes')
    plt.title('Hodgkin-Huxley Neuron Model')
    plt.ylabel('Membrane Potential (mV)')
    plt.legend()
    
    plt.subplot(312)
    plt.plot(runner.mon.ts, runner.mon.m[0], label='m (Na+ activation)')
    plt.plot(runner.mon.ts, runner.mon.h[0], label='h (Na+ inactivation)')
    plt.plot(runner.mon.ts, runner.mon.n[0], label='n (K+ activation)')
    plt.ylabel('Gating Variables')
    plt.legend()
    
    plt.subplot(313)
    plt.plot(runner.mon.ts, [input_current(t) for t in runner.mon.ts], label='Input Current', color='blue')
    plt.xlabel('Time (ms)')
    plt.ylabel('Input Current (mA/cm²)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('brainpy_hh_neuron_example.png', dpi=300)
    print("HH神经元示例结果已保存为: brainpy_hh_neuron_example.png")
    
    print(f"\n检测到的动作电位数量: {len(spikes)}")
    print(f"神经元最大膜电位: {np.max(runner.mon.V[0]):.2f} mV")
    print(f"神经元最小膜电位: {np.min(runner.mon.V[0]):.2f} mV")
    
    return hh, runner

def test_izhikevich_neuron():
    """
    测试Izhikevich神经元模型
    """
    print("\n=== 测试Izhikevich神经元模型 ===")
    
    # 创建Izhikevich神经元
    izh = bp.neurons.Izhikevich(size=1, a=0.02, b=0.2, c=-65., d=6.)
    
    # 定义输入电流
    def input_current():
        t = bp.share['t']
        if 100 < t < 200:
            return 50.
        elif 300 < t < 400:
            return 100.
        else:
            return 0.
    
    # 运行模拟
    runner = bp.DSRunner(izh, monitors=['V', 'spike', 'u'], inputs=input_current, jit=True)
    runner.run(500.)
    
    # 绘制结果
    plt.figure(figsize=(12, 8))
    
    plt.subplot(311)
    plt.plot(runner.mon.ts, runner.mon.V[0], label='Izhikevich Neuron', color='red')
    spikes = runner.mon.ts[runner.mon['spike'].flatten() > 0.5]
    plt.scatter(spikes, np.ones_like(spikes)*50, marker='|', s=200, c='black', label='Spikes')
    plt.title('Izhikevich Neuron Model')
    plt.ylabel('Membrane Potential (mV)')
    plt.legend()
    
    plt.subplot(312)
    plt.plot(runner.mon.ts, runner.mon.u[0], label='u (Recovery variable)', color='green')
    plt.ylabel('Recovery Variable')
    plt.legend()
    
    plt.subplot(313)
    plt.plot(runner.mon.ts, [input_current(t) for t in runner.mon.ts], label='Input Current', color='blue')
    plt.xlabel('Time (ms)')
    plt.ylabel('Input Current (mA)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('brainpy_izhikevich_neuron_example.png', dpi=300)
    print("Izhikevich神经元示例结果已保存为: brainpy_izhikevich_neuron_example.png")
    
    print(f"\n检测到的动作电位数量: {len(spikes)}")
    print(f"神经元最大膜电位: {np.max(runner.mon.V[0]):.2f} mV")
    print(f"神经元最小膜电位: {np.min(runner.mon.V[0]):.2f} mV")
    
    return izh, runner

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
    
    def rhythmic_input():
        t = bp.share['t']
        return amplitude * np.sin(2 * np.pi * freq * t / 1000.)
    
    # 运行模拟
    runner = bp.DSRunner(izh, monitors=['V', 'spike'], inputs=rhythmic_input, jit=True)
    runner.run(1000.)
    
    # 绘制结果
    plt.figure(figsize=(12, 6))
    
    plt.subplot(211)
    plt.plot(runner.mon.ts, [rhythmic_input(t) for t in runner.mon.ts], label='10Hz Stimulus', color='blue')
    plt.title('10Hz Rhythmic Stimulus')
    plt.ylabel('Input Current (mA)')
    plt.legend()
    
    plt.subplot(212)
    plt.plot(runner.mon.ts, runner.mon.V[0], label='Izhikevich Neuron', color='red')
    spikes = runner.mon.ts[runner.mon['spike'].flatten() > 0.5]
    plt.scatter(spikes, np.ones_like(spikes)*50, marker='|', s=200, c='black', label='Spikes')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('brainpy_rhythmic_stimulus.png', dpi=300)
    print("节律性刺激测试结果已保存为: brainpy_rhythmic_stimulus.png")
    
    print(f"\n检测到的动作电位数量: {len(spikes)}")
    print(f"平均脉冲频率: {len(spikes)/1.0:.2f} Hz")
    
    return izh, runner

def main():
    """
    主函数
    """
    print("=== BrainPy基础神经元模拟示例 ===\n")
    
    # 测试HH神经元
    test_hh_neuron()
    
    # 测试Izhikevich神经元
    test_izhikevich_neuron()
    
    # 测试节律性刺激
    test_rhythmic_stimulus()
    
    print("\n=== 所有测试完成 ===")

if __name__ == "__main__":
    main()