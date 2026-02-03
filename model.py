import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class NeuronModel:
    """
    神经元模型类，模拟具有振荡膜电位的神经元
    """
    def __init__(self, frequency, threshold=0.8):
        """
        初始化神经元模型
        
        参数:
            frequency: 膜电位振荡频率 (Hz)
            threshold: 产生动作电位的阈值
        """
        self.frequency = frequency
        self.threshold = threshold
        self.phase = 0.0  # 当前相位
        self.time_step = 1e-3  # 时间步长 (1ms)
        self.omega = 2 * np.pi * frequency  # 角频率
    
    def update_membrane_potential(self, t):
        """
        更新膜电位
        
        参数:
            t: 当前时间 (s)
            
        返回:
            膜电位值
        """
        return np.sin(self.omega * t + self.phase)
    
    def receive_spike(self, spike_time, t):
        """
        接收输入尖峰并判断是否产生动作电位
        
        参数:
            spike_time: 输入尖峰的时间 (s)
            t: 当前时间 (s)
            
        返回:
            1 如果产生动作电位，0 否则
        """
        membrane_potential = self.update_membrane_potential(t)
        
        # 如果输入尖峰时间接近当前时间，且膜电位超过阈值
        if abs(t - spike_time) < 2e-3:  # 2ms窗口
            if membrane_potential >= self.threshold:
                return 1
        return 0

class NeuralSynchronyModel:
    """
    神经同步模型，模拟背侧和腹侧通路的信息传递
    """
    def __init__(self, gamma_freq=55, high_gamma_freq=200):
        """
        初始化神经同步模型
        
        参数:
            gamma_freq: 伽马振荡频率 (40-70Hz), 默认55Hz
            high_gamma_freq: 高伽马振荡频率 (180-220Hz), 默认200Hz
        """
        # 创建两种类型的神经元
        self.gamma_neuron = NeuronModel(gamma_freq)
        self.high_gamma_neuron = NeuronModel(high_gamma_freq)
        
        # 时间参数
        self.duration = 1.0  # 模拟持续时间 (s)
        self.time_points = np.arange(0, self.duration, 1e-3)  # 时间点数组
    
    def generate_spike_train(self, frequency, phase_lock=0.8):
        """
        生成相位锁定的尖峰序列
        
        参数:
            frequency: 目标振荡频率
            phase_lock: 相位锁定强度 (0-1)
            
        返回:
            尖峰时间数组
        """
        omega = 2 * np.pi * frequency
        spike_times = []
        
        for t in self.time_points:
            phase = omega * t
            
            # 计算尖峰概率：与目标相位越接近，概率越高
            target_phase = 0  # 目标相位为0（正弦波峰值）
            phase_diff = abs((phase - target_phase) % (2 * np.pi))
            probability = phase_lock * np.cos(phase_diff) + (1 - phase_lock) * np.random.rand()
            
            if probability > 0.8:  # 尖峰产生阈值
                spike_times.append(t)
        
        return np.array(spike_times)
    
    def simulate(self, mt_input=True, v4_input=True, phase_lock_strength=0.8):
        """
        模拟神经信息传递
        
        参数:
            mt_input: 是否包含MT输入
            v4_input: 是否包含V4输入
            phase_lock_strength: 相位锁定强度
            
        返回:
            gamma_response: 伽马检测器神经元的响应
            high_gamma_response: 高伽马检测器神经元的响应
        """
        gamma_response = []
        high_gamma_response = []
        
        # 生成输入尖峰序列
        mt_spikes = self.generate_spike_train(200, phase_lock_strength) if mt_input else np.array([])
        v4_spikes = self.generate_spike_train(55, phase_lock_strength) if v4_input else np.array([])
        
        # 合并所有输入尖峰
        all_spikes = np.concatenate([mt_spikes, v4_spikes])
        
        # 模拟每个时间点的神经元响应
        for t in self.time_points:
            # 伽马检测器神经元对V4输入的响应
            gamma_resp = 0
            for spike_time in v4_spikes:
                gamma_resp += self.gamma_neuron.receive_spike(spike_time, t)
            
            # 高伽马检测器神经元对MT输入的响应
            high_gamma_resp = 0
            for spike_time in mt_spikes:
                high_gamma_resp += self.high_gamma_neuron.receive_spike(spike_time, t)
            
            gamma_response.append(gamma_resp)
            high_gamma_response.append(high_gamma_resp)
        
        return np.array(gamma_response), np.array(high_gamma_response)
    
    def evaluate_performance(self, thresholds=np.arange(0.5, 1.0, 0.1), phase_locks=np.arange(0.5, 1.0, 0.1)):
        """
        评估模型在不同参数下的性能
        
        参数:
            thresholds: 阈值数组
            phase_locks: 相位锁定强度数组
            
        返回:
            性能评估结果
        """
        results = []
        
        for threshold in thresholds:
            for phase_lock in phase_locks:
                # 设置神经元阈值
                self.gamma_neuron.threshold = threshold
                self.high_gamma_neuron.threshold = threshold
                
                # 四种条件：仅MT, 仅V4, 两者都有, 两者都没有
                conditions = [
                    (True, False, "MT only"),
                    (False, True, "V4 only"),
                    (True, True, "Both"),
                    (False, False, "None")
                ]
                
                X = []
                y = []
                
                # 生成训练数据
                for mt_input, v4_input, label in conditions:
                    for _ in range(50):  # 每个条件生成50个样本
                        gamma_resp, high_gamma_resp = self.simulate(mt_input, v4_input, phase_lock)
                        
                        # 计算响应率（每秒尖峰数）
                        gamma_rate = np.sum(gamma_resp) / self.duration
                        high_gamma_rate = np.sum(high_gamma_resp) / self.duration
                        
                        X.append([gamma_rate, high_gamma_rate])
                        y.append(label)
                
                # 训练SVM分类器
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
                clf = SVC(kernel='rbf')
                clf.fit(X_train, y_train)
                
                # 预测并计算准确率
                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                results.append({
                    'threshold': threshold,
                    'phase_lock': phase_lock,
                    'accuracy': accuracy
                })
                
                print(f"Threshold: {threshold:.1f}, Phase Lock: {phase_lock:.1f}, Accuracy: {accuracy:.3f}")
        
        return results

def main():
    """
    主函数，运行模型示例
    """
    # 创建模型
    model = NeuralSynchronyModel()
    
    # 运行示例模拟
    print("Running simulation example...")
    gamma_resp, high_gamma_resp = model.simulate(mt_input=True, v4_input=True, phase_lock_strength=0.8)
    
    # 绘制结果
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(model.time_points, gamma_resp, label='Gamma detector neuron', alpha=0.7)
    plt.title('Gamma Detector Neuron Response')
    plt.xlabel('Time (s)')
    plt.ylabel('Spike count')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(model.time_points, high_gamma_resp, label='High-gamma detector neuron', alpha=0.7, color='red')
    plt.title('High-Gamma Detector Neuron Response')
    plt.xlabel('Time (s)')
    plt.ylabel('Spike count')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('simulation_results.png')
    print("Simulation results saved to simulation_results.png")
    
    # 评估模型性能
    print("\nEvaluating model performance...")
    results = model.evaluate_performance()
    
    # 保存评估结果
    np.savez('performance_results.npz', results=results)
    print("Performance results saved to performance_results.npz")

if __name__ == "__main__":
    main()