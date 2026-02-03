# 神经同步模型 - 复现 Khamechian et al. 2019

这是一个基于Python的神经同步模型，复现了Khamechian等人2019年发表在PNAS上的论文中的计算建模部分。

## 论文摘要

论文提出了一种机制，通过不同频率的神经同步来路由信息流，使得高等灵长类皮层能够建立"功能标记线"（functionally labeled lines），从而动态调节皮层信息传递和多路复用聚合的感觉信号。

## 模型原理

模型模拟了两种类型的前额叶皮层（PFC）神经元：

1. **伽马检测器神经元**：膜电位在伽马频率（40-70Hz）振荡，主要响应来自腹侧通路（如V4区）的输入
2. **高伽马检测器神经元**：膜电位在高伽马频率（180-220Hz）振荡，主要响应来自背侧通路（如MT区）的输入

当输入尖峰与神经元的振荡相位锁定时，神经元更容易产生动作电位，从而实现信息的选择性传递。

## 安装依赖

```bash
pip install numpy matplotlib scikit-learn
```

## 使用方法

### 基本模拟

```python
from model import NeuralSynchronyModel

# 创建模型
model = NeuralSynchronyModel(gamma_freq=55, high_gamma_freq=200)

# 运行模拟：同时包含MT和V4输入
gamma_response, high_gamma_response = model.simulate(
    mt_input=True, 
    v4_input=True, 
    phase_lock_strength=0.8
)
```

### 评估模型性能

```python
# 评估不同阈值和相位锁定强度下的模型性能
results = model.evaluate_performance()
```

### 运行完整示例

```bash
python model.py
```

这将生成模拟结果图表和性能评估数据。

## 文件说明

- `model.py`：主要的模型实现代码
- `README.md`：本说明文档
- `simulation_results.png`：模拟结果图表（运行示例后生成）
- `performance_results.npz`：性能评估数据（运行示例后生成）

## 关键参数

- `gamma_freq`：伽马振荡频率，默认55Hz（40-70Hz范围）
- `high_gamma_freq`：高伽马振荡频率，默认200Hz（180-220Hz范围）
- `phase_lock_strength`：相位锁定强度，范围0-1，默认0.8
- `threshold`：神经元动作电位阈值，范围0-1，默认0.8

## 模型验证

模型通过支持向量机（SVM）分类器来评估性能，能够区分四种输入条件：
1. 仅MT输入（背侧通路）
2. 仅V4输入（腹侧通路）
3. 同时包含MT和V4输入
4. 无输入

分类准确率越高，说明模型越能有效区分不同来源的神经信息。

## 参考论文

Khamechian, M. B., Kozyrev, V., Treue, S., Esghaei, M., & Daliri, M. R. (2019). Routing information flow by separate neural synchrony frequencies allows for “functionally labeled lines” in higher primate cortex. Proceedings of the National Academy of Sciences, 116(25), 12506-12515.

DOI: https://doi.org/10.1073/pnas.1819827116