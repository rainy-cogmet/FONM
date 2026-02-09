import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.signal import hilbert, find_peaks
from scipy.stats import pearsonr

class NeuronModel:
    """
    神经元模型类，模拟具有振荡膜电位的神经元，支持相位-振幅耦合
    """
    def __init__(self, frequency, threshold=0.8, pac_strength=0.5):
        """
        初始化神经元模型
        
        参数:
            frequency: 膜电位振荡频率 (Hz)
            threshold: 产生动作电位的阈值
            pac_strength: 相位-振幅耦合强度 (0-1)
        """
        self.frequency = frequency
        self.base_frequency = frequency  # 基础频率，用于恢复
        self.threshold = threshold
        self.pac_strength = pac_strength  # 相位-振幅耦合强度
        self.phase = 0.0  # 当前相位
        self.time_step = 1e-3  # 时间步长 (1ms)
        self.omega = 2 * np.pi * frequency  # 角频率
        self.entrainment_history = []  # 记录频率变化历史
        self.amplitude_modulation = 1.0  # 振幅调制因子
    
    def update_membrane_potential(self, t, phase_modulation=0.0):
        """
        更新膜电位，支持相位和振幅调制
        
        参数:
            t: 当前时间 (s)
            phase_modulation: 相位调制因子
            
        返回:
            膜电位值
        """
        # 基础振荡
        base_oscillation = np.sin(self.omega * t + self.phase)
        
        # 振幅调制（PAC机制）
        modulated_oscillation = self.amplitude_modulation * base_oscillation
        
        return modulated_oscillation
    
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
    
    def receive_continuous_stimulus(self, stimulus_signal, t):
        """
        接收连续的刺激信号并判断是否产生动作电位
        
        参数:
            stimulus_signal: 刺激信号在当前时间点的值
            t: 当前时间 (s)
            
        返回:
            1 如果产生动作电位，0 否则
        """
        # 计算当前膜电位
        membrane_potential = self.update_membrane_potential(t)
        
        # 刺激信号通过相位-振幅耦合增强膜电位
        # 当刺激信号为正时，增强膜电位振幅；为负时，减弱振幅
        self.amplitude_modulation = 1.0 + self.pac_strength * stimulus_signal
        self.amplitude_modulation = np.clip(self.amplitude_modulation, 0.1, 2.0)  # 限制振幅范围
        
        # 计算调制后的膜电位
        modulated_potential = self.update_membrane_potential(t)
        
        if modulated_potential >= self.threshold:
            return 1
        return 0
    
    def phase_entrainment(self, stimulus_signal, t, entrainment_strength=0.8):
        """
        相位锁定Entrainment算法
        
        参数:
            stimulus_signal: 外源刺激信号
            t: 当前时间 (s)
            entrainment_strength: Entrain强度 (0-1)
        """
        # 计算刺激信号的瞬时相位
        # 简化：通过正弦波拟合得到刺激相位
        if np.abs(stimulus_signal) > 0.5:  # 只有当刺激强度足够大时才进行相位调整
            # 计算刺激信号在当前时间点的相位
            stimulus_phase = np.arcsin(stimulus_signal)
            
            # 计算目标相位与当前相位的差
            phase_diff = stimulus_phase - self.phase
            
            # 调整神经元相位，实现相位锁定
            self.phase += phase_diff * entrainment_strength * self.time_step
            
            # 保持相位在[-π, π]范围内
            self.phase = np.mod(self.phase + np.pi, 2 * np.pi) - np.pi
    
    def frequency_adaptation(self, stimulus_frequency, entrainment_strength, t):
        """
        模拟外源刺激导致的振荡频率自适应
        
        参数:
            stimulus_frequency: 外源刺激频率 (Hz)
            entrainment_strength: Entrain强度 (0-1)
            t: 当前时间 (s)
        """
        # 计算频率偏移
        frequency_shift = (stimulus_frequency - self.frequency) * entrainment_strength
        
        # 更新频率和角频率
        self.frequency += frequency_shift * self.time_step
        self.omega = 2 * np.pi * self.frequency
        
        # 记录频率变化
        self.entrainment_history.append((t, self.frequency))
    
    def reset_frequency(self):
        """
        重置振荡频率为基础频率
        """
        self.frequency = self.base_frequency
        self.omega = 2 * np.pi * self.frequency
        self.phase = 0.0
        self.amplitude_modulation = 1.0
        self.entrainment_history = []

class NeuralSynchronyModel:
    """
    神经同步模型，模拟背侧和腹侧通路的信息传递
    """
    def __init__(self, gamma_freq=55, high_gamma_freq=75):
        """
        初始化神经同步模型
        
        参数:
            gamma_freq: 伽马振荡频率 (40-70Hz), 默认55Hz
            high_gamma_freq: 高伽马振荡频率 (60-90Hz), 默认75Hz
        """
        # 创建两种类型的神经元
        self.gamma_neuron = NeuronModel(gamma_freq)
        self.high_gamma_neuron = NeuronModel(high_gamma_freq)
        
        # 时间参数
        self.duration = 1.0  # 模拟持续时间 (s)
        self.time_points = np.arange(0, self.duration, 1e-3)  # 时间点数组
    
    def generate_rhythmic_stimulus(self, frequencies, amplitudes, duration=None):
        """
        生成多频率叠加的外源节律性刺激
        
        参数:
            frequencies: 频率列表 (Hz)
            amplitudes: 振幅列表
            duration: 刺激持续时间 (s)
            
        返回:
            刺激信号数组
        """
        if duration is None:
            duration = self.duration
            time_points = self.time_points
        else:
            time_points = np.arange(0, duration, 1e-3)
        
        stimulus = np.zeros_like(time_points)
        
        for freq, amp in zip(frequencies, amplitudes):
            stimulus += amp * np.sin(2 * np.pi * freq * time_points)
        
        # 归一化刺激信号
        stimulus = stimulus / np.max(np.abs(stimulus)) if np.max(np.abs(stimulus)) > 0 else stimulus
        
        return stimulus, time_points
    
    def calculate_plv(self, signal1, signal2):
        """
        计算相位锁定值（Phase Locking Value）
        
        参数:
            signal1: 信号1
            signal2: 信号2
            
        返回:
            PLV值 (0-1)
        """
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
    
    def calculate_packet_loss(self, response_signal, target_frequency, threshold=0.3):
        """
        计算下游神经活动中目标频率信号的丢包率
        
        参数:
            response_signal: 神经元响应信号
            target_frequency: 目标频率 (Hz)
            threshold: 检测阈值
            
        返回:
            丢包率 (0-1), 丢包时间点列表
        """
        # 计算响应信号的功率谱
        fft_vals = np.fft.fft(response_signal)
        fft_freqs = np.fft.fftfreq(len(response_signal), d=1e-3)
        
        # 找到目标频率附近的峰值
        target_idx = np.argmin(np.abs(fft_freqs - target_frequency))
        peak_value = np.abs(fft_vals[target_idx])
        
        # 计算响应信号的包络
        analytic_signal = hilbert(response_signal)
        amplitude_envelope = np.abs(analytic_signal)
        
        # 找到低于阈值的时间段
        below_threshold = amplitude_envelope < threshold
        
        # 计算丢包率
        packet_loss_rate = np.mean(below_threshold)
        
        # 找到丢包的时间点
        packet_loss_points = self.time_points[below_threshold]
        
        return packet_loss_rate, packet_loss_points
    
    def simulate_entrained(self, stimulus, pac_strength=0.7, target_neuron='mt'):
        """
        模拟外源刺激Entrain内源振荡后的信息传递，仅保留相位-振幅耦合
        
        参数:
            stimulus: 外源刺激信号
            pac_strength: 相位-振幅耦合强度 (0-1)
            target_neuron: 目标神经元 ('mt' 或 'v4')
            
        返回:
            gamma_response: 伽马检测器神经元的响应
            high_gamma_response: 高伽马检测器神经元的响应
            pac_value: 相位-振幅耦合值
            membrane_potentials: 膜电位时间序列
        """
        gamma_response = np.zeros_like(self.time_points)
        high_gamma_response = np.zeros_like(self.time_points)
        
        # 选择目标神经元
        target_neuron = self.high_gamma_neuron if target_neuron == 'mt' else self.gamma_neuron
        
        # 设置PAC强度
        target_neuron.pac_strength = pac_strength
        
        # 存储膜电位历史用于PAC计算
        membrane_potentials = np.zeros_like(self.time_points)
        
        # 模拟Entrainment过程（仅PAC）
        for i, t in enumerate(self.time_points):
            # 接收刺激信号（触发PAC）
            if target_neuron == self.gamma_neuron:
                gamma_response[i] = target_neuron.receive_continuous_stimulus(stimulus[i], t)
            else:
                high_gamma_response[i] = target_neuron.receive_continuous_stimulus(stimulus[i], t)
            
            # 记录膜电位
            membrane_potentials[i] = target_neuron.update_membrane_potential(t)
        
        # 计算相位-振幅耦合值
        pac_value = self.calculate_pac(stimulus, membrane_potentials)
        
        # 重置神经元状态
        target_neuron.reset_frequency()
        
        return gamma_response, high_gamma_response, pac_value, membrane_potentials
    
    def calculate_pac(self, low_freq_signal, high_freq_signal):
        """
        计算相位-振幅耦合（Phase-Amplitude Coupling）
        
        参数:
            low_freq_signal: 低频相位信号
            high_freq_signal: 高频振幅信号
            
        返回:
            PAC值 (0-1)
        """
        # 计算低频信号的瞬时相位
        low_analytic = hilbert(low_freq_signal)
        low_phase = np.angle(low_analytic)
        
        # 计算高频信号的瞬时振幅
        high_analytic = hilbert(high_freq_signal)
        high_amplitude = np.abs(high_analytic)
        
        # 对相位进行分箱
        phase_bins = np.linspace(-np.pi, np.pi, 18, endpoint=False)
        pac_values = []
        
        for i in range(len(phase_bins)-1):
            # 找到当前相位区间内的样本
            mask = (low_phase >= phase_bins[i]) & (low_phase < phase_bins[i+1])
            if np.sum(mask) > 0:
                # 计算该相位区间内的平均振幅
                avg_amplitude = np.mean(high_amplitude[mask])
                pac_values.append(avg_amplitude)
        
        # 计算PAC值（振幅随相位变化的调制深度）
        if len(pac_values) > 1:
            pac_value = (np.max(pac_values) - np.min(pac_values)) / np.mean(pac_values)
            pac_value = np.clip(pac_value, 0, 1)
        else:
            pac_value = 0
            
        return pac_value
    
    def evaluate_decoding_accuracy(self, response_signal, target_frequency):
        """
        评估下游神经活动中目标频率信息的解码准确率
        
        参数:
            response_signal: 神经元响应信号
            target_frequency: 目标频率 (Hz)
            
        返回:
            解码准确率 (0-1)
        """
        # 计算响应信号的功率谱
        fft_vals = np.fft.fft(response_signal)
        fft_freqs = np.fft.fftfreq(len(response_signal), d=1e-3)
        
        # 找到目标频率附近的峰值
        target_idx = np.argmin(np.abs(fft_freqs - target_frequency))
        peak_value = np.abs(fft_vals[target_idx])
        
        # 计算总功率
        total_power = np.sum(np.abs(fft_vals))
        
        # 准确率定义为目标频率功率占总功率的比例
        accuracy = peak_value / total_power if total_power > 0 else 0
        
        # 归一化到0-1范围
        accuracy = np.clip(accuracy, 0, 1)
        
        return accuracy
    
    def decode_10hz_signal(self, membrane_potentials, threshold=0.5):
        """
        专门的10Hz信号读出模块
        
        参数:
            membrane_potentials: 神经元膜电位时间序列
            threshold: 振幅阈值
            
        返回:
            decoding_success: 是否成功解码10Hz信号
            pulse_count: 检测到的10Hz脉冲次数
            decoding_ratio: 解码成功率
        """
        # 计算膜电位的振幅包络
        analytic_signal = hilbert(membrane_potentials)
        amplitude_envelope = np.abs(analytic_signal)
        
        # 找到超过阈值的时间点
        above_threshold = amplitude_envelope > threshold
        
        # 计算相邻超过阈值事件的时间间隔
        rising_edges = np.where(np.diff(above_threshold.astype(int)) == 1)[0] + 1
        
        # 计算相邻脉冲的时间间隔
        if len(rising_edges) >= 2:
            time_intervals = np.diff(self.time_points[rising_edges])
            
            # 计算预期的10Hz间隔（100ms）
            expected_interval = 1.0 / 10
            
            # 检测符合10Hz间隔的脉冲
            valid_pulses = np.where(np.abs(time_intervals - expected_interval) < 0.02)[0]
            
            # 计算解码成功率
            decoding_ratio = len(valid_pulses) / len(time_intervals) if len(time_intervals) > 0 else 0
            decoding_success = decoding_ratio > 0.7  # 超过70%的脉冲符合10Hz间隔则判定为成功
            
            return decoding_success, len(rising_edges), decoding_ratio
        else:
            return False, len(rising_edges), 0.0
    
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
        target_phase = 0  # 目标相位为0（正弦波峰值）
        
        # 计算每个周期的峰值时间
        cycle_times = np.arange(0, self.duration, 1/frequency)
        
        spike_times = []
        for cycle_start in cycle_times:
            # 在峰值时间附近添加尖峰
            jitter = (np.random.rand() - 0.5) * 0.5 / frequency  # ±半个周期的抖动
            spike_time = cycle_start + jitter
            
            # 根据相位锁定强度决定是否添加尖峰
            if np.random.rand() < phase_lock:
                spike_times.append(spike_time)
        
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
        gamma_response = np.zeros_like(self.time_points)
        high_gamma_response = np.zeros_like(self.time_points)
        
        # 生成输入尖峰序列
        mt_spikes = self.generate_spike_train(200, phase_lock_strength) if mt_input else np.array([])
        v4_spikes = self.generate_spike_train(55, phase_lock_strength) if v4_input else np.array([])
        
        # 优化响应计算：只在尖峰时间附近检查
        for spike_time in v4_spikes:
            # 找到尖峰时间附近的时间点索引
            idx = np.abs(self.time_points - spike_time).argmin()
            # 检查前后2ms的窗口
            window = slice(max(0, idx-2), min(len(self.time_points), idx+3))
            for i, t in enumerate(self.time_points[window]):
                gamma_response[window.start + i] += self.gamma_neuron.receive_spike(spike_time, t)
        
        for spike_time in mt_spikes:
            idx = np.abs(self.time_points - spike_time).argmin()
            window = slice(max(0, idx-2), min(len(self.time_points), idx+3))
            for i, t in enumerate(self.time_points[window]):
                high_gamma_response[window.start + i] += self.high_gamma_neuron.receive_spike(spike_time, t)
        
        return gamma_response, high_gamma_response
    
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
                
                # 增加神经元采样数量 - 从6种条件扩展到12种条件
                conditions = [
                    (True, False, "MT only"),
                    (False, True, "V4 only"),
                    (True, True, "Both"),
                    (False, False, "None"),
                    (1.0, 0.8, "MT strong V4 moderate"),
                    (0.8, 1.0, "MT moderate V4 strong"),
                    (0.6, 0.2, "MT moderate V4 weak"),
                    (0.2, 0.6, "MT weak V4 moderate"),
                    (0.4, 0.4, "Equal moderate input"),
                    (0.3, 0.7, "Slight V4 bias"),
                    (0.7, 0.3, "Slight MT bias"),
                    (0.1, 0.1, "Very weak input")
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

def plot_fig4d(model):
    """
    绘制论文中的Figure 4D
    """
    print("\nPlotting Figure 4D...")
    
    # 四种条件
    conditions = [
        (True, False, "MT only"),
        (False, True, "V4 only"),
        (True, True, "Both"),
        (False, False, "None")
    ]
    
    gamma_rates = []
    high_gamma_rates = []
    labels = []
    
    # 为每个条件生成数据
    for mt_input, v4_input, label in conditions:
        gamma_resp, high_gamma_resp = model.simulate(mt_input, v4_input, phase_lock_strength=0.8)
        
        # 计算响应率
        gamma_rate = np.sum(gamma_resp) / model.duration
        high_gamma_rate = np.sum(high_gamma_resp) / model.duration
        
        gamma_rates.append(gamma_rate)
        high_gamma_rates.append(high_gamma_rate)
        labels.append(label)
    
    # 绘制散点图
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'red', 'green', 'gray']
    
    for i, (gamma_rate, high_gamma_rate, label, color) in enumerate(zip(gamma_rates, high_gamma_rates, labels, colors)):
        plt.scatter(gamma_rate, high_gamma_rate, s=200, color=color, label=label, alpha=0.7)
    
    plt.xlabel('Gamma detector neuron response rate (spikes/s)')
    plt.ylabel('High-gamma detector neuron response rate (spikes/s)')
    plt.title('Figure 4D: Response patterns for different input conditions')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figure_4d.png', dpi=300, bbox_inches='tight')
    print("Figure 4D saved to figure_4d.png")
    
    return gamma_rates, high_gamma_rates, labels

def plot_fig4e(model):
    """
    绘制论文中的Figure 4E
    """
    print("\nPlotting Figure 4E...")
    
    # 不同的阈值和相位锁定强度
    thresholds = np.arange(0.5, 1.0, 0.05)
    phase_locks = np.arange(0.5, 1.0, 0.1)
    
    # 存储准确率
    accuracy_matrix = np.zeros((len(thresholds), len(phase_locks)))
    
    # 评估每个参数组合的性能
    for i, threshold in enumerate(thresholds):
        for j, phase_lock in enumerate(phase_locks):
            # 设置神经元阈值
            model.gamma_neuron.threshold = threshold
            model.high_gamma_neuron.threshold = threshold
            
            # 生成训练数据
            X = []
            y = []
            
            conditions = [
                (True, False, "MT only"),
                (False, True, "V4 only"),
                (True, True, "Both"),
                (False, False, "None")
            ]
            
            for mt_input, v4_input, label in conditions:
                for _ in range(20):  # 每个条件生成20个样本
                    gamma_resp, high_gamma_resp = model.simulate(mt_input, v4_input, phase_lock)
                    
                    # 计算响应率
                    gamma_rate = np.sum(gamma_resp) / model.duration
                    high_gamma_rate = np.sum(high_gamma_resp) / model.duration
                    
                    X.append([gamma_rate, high_gamma_rate])
                    y.append(label)
            
            # 训练SVM分类器
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            clf = SVC(kernel='rbf')
            clf.fit(X_train, y_train)
            
            # 预测并计算准确率
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            accuracy_matrix[i, j] = accuracy
    
    # 绘制热力图
    plt.figure(figsize=(10, 6))
    im = plt.imshow(accuracy_matrix, extent=[0.5, 0.9, 0.9, 0.5], 
                    aspect='auto', cmap='viridis', vmin=0.7, vmax=1.0)
    
    # 添加颜色条
    cbar = plt.colorbar(im)
    cbar.set_label('Classification accuracy')
    
    # 设置坐标轴标签
    plt.xlabel('Phase lock strength')
    plt.ylabel('Neuron threshold')
    plt.title('Figure 4E: Classification accuracy across parameter space')
    
    # 添加数值标签
    for i in range(len(thresholds)):
        for j in range(len(phase_locks)):
            plt.text(phase_locks[j], thresholds[i], f'{accuracy_matrix[i, j]:.2f}',
                    ha='center', va='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figure_4e.png', dpi=300)
    print("Figure 4E saved to figure_4e.png")
    
    return accuracy_matrix

def run_entrainment_experiment():
    """
    运行外源节律性刺激Entrain内源振荡的实验
    """
    print("\n=== 运行Entrainment实验 (简化版PAC模型) ===")
    
    # 创建模型（MT区频率已调整为75Hz）
    model = NeuralSynchronyModel(high_gamma_freq=75)
    print(f"MT区振荡频率设置为: {model.high_gamma_neuron.frequency}Hz")
    
    # --------------------------
    # 条件a：10Hz纯节律刺激
    # --------------------------
    print("\n--- 条件a：10Hz纯节律刺激 ---")
    
    # 生成10Hz外源节律刺激
    stimulus_a, time_points = model.generate_rhythmic_stimulus(
        frequencies=[10],
        amplitudes=[1.0]
    )
    
    # 模拟Entrainment (仅PAC)
    gamma_resp_a, high_gamma_resp_a, pac_a, membrane_potentials_a = model.simulate_entrained(
        stimulus=stimulus_a,
        pac_strength=0.7,
        target_neuron='mt'
    )
    
    # 计算解码准确率
    decoding_accuracy_a = model.evaluate_decoding_accuracy(
        response_signal=high_gamma_resp_a,
        target_frequency=10
    )
    
    # 计算丢包率
    packet_loss_a, loss_points_a = model.calculate_packet_loss(
        response_signal=high_gamma_resp_a,
        target_frequency=10
    )
    
    # 专用10Hz解码
    decode_success_a, pulse_count_a, decode_ratio_a = model.decode_10hz_signal(
        membrane_potentials=membrane_potentials_a,
        threshold=0.5
    )
    
    print(f"PAC值: {pac_a:.3f}")
    print(f"解码准确率: {decoding_accuracy_a:.3f}")
    print(f"丢包率: {packet_loss_a:.3f}")
    print(f"专用10Hz解码: {'成功' if decode_success_a else '失败'} (脉冲数: {pulse_count_a}, 成功率: {decode_ratio_a:.3f})")
    
    # --------------------------
    # 条件b：10Hz+3Hz叠加刺激
    # --------------------------
    print("\n--- 条件b：10Hz+3Hz叠加刺激 ---")
    
    # 生成10Hz+3Hz叠加刺激
    stimulus_b, time_points = model.generate_rhythmic_stimulus(
        frequencies=[10, 3],
        amplitudes=[0.8, 0.6]  # 3Hz振幅为10Hz的75%
    )
    
    # 模拟Entrainment (仅PAC)
    gamma_resp_b, high_gamma_resp_b, pac_b, membrane_potentials_b = model.simulate_entrained(
        stimulus=stimulus_b,
        pac_strength=0.7,
        target_neuron='mt'
    )
    
    # 计算解码准确率
    decoding_accuracy_b = model.evaluate_decoding_accuracy(
        response_signal=high_gamma_resp_b,
        target_frequency=10
    )
    
    # 计算丢包率
    packet_loss_b, loss_points_b = model.calculate_packet_loss(
        response_signal=high_gamma_resp_b,
        target_frequency=10
    )
    
    # 专用10Hz解码
    decode_success_b, pulse_count_b, decode_ratio_b = model.decode_10hz_signal(
        membrane_potentials=membrane_potentials_b,
        threshold=0.5
    )
    
    print(f"PAC值: {pac_b:.3f}")
    print(f"解码准确率: {decoding_accuracy_b:.3f}")
    print(f"丢包率: {packet_loss_b:.3f}")
    print(f"专用10Hz解码: {'成功' if decode_success_b else '失败'} (脉冲数: {pulse_count_b}, 成功率: {decode_ratio_b:.3f})")
    
    # --------------------------
    # 绘制实验结果
    # --------------------------
    print("\n--- 绘制实验结果 ---")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 条件a：刺激信号和膜电位
    axes[0, 0].plot(time_points, stimulus_a, label='10Hz Stimulus', alpha=0.7)
    axes[0, 0].set_title('Condition a: 10Hz Rhythmic Stimulus')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(time_points, membrane_potentials_a, label='MT Neuron Membrane Potential', alpha=0.7, color='red')
    axes[0, 1].axhline(y=0.5, color='gray', linestyle='--', label='Threshold')
    axes[0, 1].set_title('Condition a: MT Neuron Membrane Potential')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Membrane Potential')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 条件b：刺激信号和膜电位
    axes[1, 0].plot(time_points, stimulus_b, label='10Hz+3Hz Stimulus', alpha=0.7)
    axes[1, 0].set_title('Condition b: 10Hz + 3Hz Combined Stimulus')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(time_points, membrane_potentials_b, label='MT Neuron Membrane Potential', alpha=0.7, color='red')
    axes[1, 1].axhline(y=0.5, color='gray', linestyle='--', label='Threshold')
    axes[1, 1].set_title('Condition b: MT Neuron Membrane Potential')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Membrane Potential')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('entrainment_experiment_results.png', dpi=300)
    print("实验结果图已保存为: entrainment_experiment_results.png")
    
    # --------------------------
    # 绘制对比柱状图
    # --------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # PAC对比
    pac_values = [pac_a, pac_b]
    axes[0, 0].bar(['Condition a', 'Condition b'], pac_values, color=['blue', 'orange'])
    axes[0, 0].set_title('Phase-Amplitude Coupling (PAC) Comparison')
    axes[0, 0].set_ylabel('PAC (0-1)')
    axes[0, 0].set_ylim(0, 1)
    for i, v in enumerate(pac_values):
        axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    # 解码准确率对比
    accuracy_values = [decoding_accuracy_a, decoding_accuracy_b]
    axes[0, 1].bar(['Condition a', 'Condition b'], accuracy_values, color=['blue', 'orange'])
    axes[0, 1].set_title('Decoding Accuracy Comparison')
    axes[0, 1].set_ylabel('Accuracy (0-1)')
    axes[0, 1].set_ylim(0, 1)
    for i, v in enumerate(accuracy_values):
        axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    # 丢包率对比
    loss_values = [packet_loss_a, packet_loss_b]
    axes[1, 0].bar(['Condition a', 'Condition b'], loss_values, color=['blue', 'orange'])
    axes[1, 0].set_title('Packet Loss Rate Comparison')
    axes[1, 0].set_ylabel('Loss Rate (0-1)')
    axes[1, 0].set_ylim(0, 1)
    for i, v in enumerate(loss_values):
        axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    # 专用10Hz解码成功率对比
    decode_ratios = [decode_ratio_a, decode_ratio_b]
    axes[1, 1].bar(['Condition a', 'Condition b'], decode_ratios, color=['blue', 'orange'])
    axes[1, 1].set_title('10Hz Specific Decoding Success Rate')
    axes[1, 1].set_ylabel('Success Rate (0-1)')
    axes[1, 1].set_ylim(0, 1)
    for i, v in enumerate(decode_ratios):
        axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('experiment_comparison.png', dpi=300)
    print("对比柱状图已保存为: experiment_comparison.png")
    
    # --------------------------
    # 保存实验数据
    # --------------------------
    np.savez('entrainment_experiment_data.npz',
             # 条件a数据
             stimulus_a=stimulus_a,
             high_gamma_resp_a=high_gamma_resp_a,
             pac_a=pac_a,
             decoding_accuracy_a=decoding_accuracy_a,
             packet_loss_a=packet_loss_a,
             loss_points_a=loss_points_a,
             membrane_potentials_a=membrane_potentials_a,
             decode_success_a=decode_success_a,
             pulse_count_a=pulse_count_a,
             decode_ratio_a=decode_ratio_a,
             # 条件b数据
             stimulus_b=stimulus_b,
             high_gamma_resp_b=high_gamma_resp_b,
             pac_b=pac_b,
             decoding_accuracy_b=decoding_accuracy_b,
             packet_loss_b=packet_loss_b,
             loss_points_b=loss_points_b,
             membrane_potentials_b=membrane_potentials_b,
             decode_success_b=decode_success_b,
             pulse_count_b=pulse_count_b,
             decode_ratio_b=decode_ratio_b,
             time_points=time_points)
    
    print("\n实验数据已保存为: entrainment_experiment_data.npz")
    print("\n=== 实验完成 ===")

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
    
    # 绘制Figure 4D
    gamma_rates, high_gamma_rates, labels = plot_fig4d(model)
    
    # 绘制Figure 4E
    accuracy_matrix = plot_fig4e(model)
    
    # 保存数据
    np.savez('figure_data.npz', 
             gamma_rates=gamma_rates,
             high_gamma_rates=high_gamma_rates,
             labels=labels,
             accuracy_matrix=accuracy_matrix)
    print("\nAll figure data saved to figure_data.npz")
    
    # 运行Entrainment实验
    run_entrainment_experiment()
    
    print("\nDone!")

if __name__ == "__main__":
    main()