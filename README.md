# 神经振荡相位-振幅耦合计算建模

## 项目简介

本项目实现了神经振荡的相位-振幅耦合（Phase-Amplitude Coupling, PAC）计算建模，研究外源节律性刺激对神经振荡的调制效应。项目基于Khamechian等人2019年发表在PNAS上的论文“Routing information flow by separate neural synchrony frequencies allows for 'functionally labeled lines' in higher primate cortex”进行扩展和改进。

## 核心实验

### 实验目标
研究外源节律性刺激如何通过相位-振幅耦合调制神经振荡，并评估下游神经活动中信息的可解码性。

### 实验条件

#### 条件a：10Hz纯节律刺激
- 外源输入为10Hz的纯节律信号
- 成功实现相位-振幅耦合（PAC值=1.000）
- 下游神经活动中可清晰解码出10Hz信息（解码成功率=100%）

#### 条件b：10Hz+3Hz叠加刺激
- 外源输入为10Hz和3Hz的叠加信号
- 相位-振幅耦合效果减弱（PAC值=0.889）
- 下游神经活动中10Hz信息出现“丢包”现象（解码成功率=33.3%）

## 模型架构

### 核心模块

#### 1. 神经元模型
- 支持振荡频率配置（默认60-90Hz范围）
- 实现相位-振幅耦合机制
- 膜电位振幅随外源刺激相位动态调制

#### 2. 刺激生成模块
- 支持多频率叠加信号生成
- 精确控制各频率分量的振幅和相位
- 实时输出连续刺激信号

#### 3. 解码模块
- 专用10Hz读出模块，检测振幅脉冲时间间隔
- 传统功率谱解码方法作为对照
- 定量评估解码准确率和信息丢包率

## 安装使用

### 环境依赖
```bash
pip install numpy matplotlib scikit-learn scipy
```

### 运行实验
```bash
python model.py
```

### 查看结果
- `entrainment_experiment_results.png`：实验结果图
- `experiment_comparison.png`：指标对比柱状图
- `entrainment_experiment_data.npz`：完整实验数据

## 关键参数

| 参数 | 默认值 | 范围 | 描述 |
|------|--------|------|------|
| mt_frequency | 75Hz | 60-90Hz | MT区内源振荡频率 |
| v4_frequency | 55Hz | 40-70Hz | V4区内源振荡频率 |
| pac_strength | 0.7 | 0-1 | 相位-振幅耦合强度 |
| threshold | 0.5 | 0-1 | 神经元动作电位阈值 |

## 结果指标

### 主要指标
- **PAC值**：相位-振幅耦合强度（0-1），值越高表示耦合效果越好
- **解码成功率**：专用10Hz解码模块的成功率（0-1），值越高表示10Hz信息越清晰
- **丢包率**：下游神经活动中信息丢失的比例（0-1），值越高表示信息丢失越严重

### 辅助指标
- **PLV值**：相位锁定值（0-1），衡量相位同步程度
- **解码准确率**：传统功率谱方法的解码准确率（0-1）

## 创新点

1. **简化的PAC模型**：移除相位锁定机制，仅保留相位-振幅耦合，更符合生理特性
2. **专用解码方法**：直接检测振幅脉冲时间间隔，比传统功率谱方法更敏感
3. **频率比优化**：采用75Hz与10Hz的频率比，接近生理上常见的PAC频率比

## 参考论文

Khamechian, M. B., Kozyrev, V., Treue, S., Esghaei, M., & Daliri, M. R. (2019). Routing information flow by separate neural synchrony frequencies allows for "functionally labeled lines" in higher primate cortex. Proceedings of the National Academy of Sciences, 116(25), 12506-12515.

DOI: https://doi.org/10.1073/pnas.1819827116

## 项目结构

```
.
├── model.py              # 核心模型实现
├── README.md            # 项目说明文档
├── .gitignore           # Git忽略规则
├── venv/                # Python虚拟环境
├── requirements.txt     # 依赖声明
└── examples/            # 使用示例（计划中）
```

## 开发计划

- [ ] 添加更多频率比的实验
- [ ] 实现更复杂的嵌套振荡模型
- [ ] 开发交互式可视化工具
- [ ] 支持更多解码算法对比
- [ ] 完善文档和示例

## 许可证

本项目采用MIT许可证，详见LICENSE文件。