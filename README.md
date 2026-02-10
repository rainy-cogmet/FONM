# 神经同步模型（Khamechian et al. 2019 复现与扩展）

本仓库包含两条主线：
1. 神经同步/相位锁定模型（`model.py`），复现论文中的双频率路由机制。
2. 相位-振幅耦合（PAC）扩展模型（`FONM/model.py`），用于研究外源节律刺激对下游信息可解码性的影响。

此外，`FONM/brainpy_*.py` 为 BrainPy 实验脚本，作为可选探索路线。

## 目录导览

- `model.py`：同步模型主入口（Fig.4D/4E 复现）
- `FONM/model.py`：PAC/entrainment 扩展实验
- `FONM/brainpy_*.py`：BrainPy 版本实验脚本
- `docs/`：讨论稿与 BrainPy 研究报告
- `requirements.txt`：基础依赖
- `requirements-brainpy.txt`：BrainPy 可选依赖

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

可选（BrainPy 实验）：

```bash
pip install -r requirements-brainpy.txt
```

### 运行同步模型（Fig.4D/4E）

```bash
python model.py
```

### 运行 PAC/entrainment 实验

```bash
python FONM/model.py
```

生成的 `*.png` / `*.npz` 结果会写入当前工作目录，已在 `.gitignore` 中默认忽略。

## 模型简述

- 同步模型模拟两类 PFC 神经元：伽马检测器（40–70Hz）与高伽马检测器（180–220Hz）。
- PAC 模型模拟外源节律刺激对振荡振幅的调制，并评估下游 10Hz 信息的可解码性与丢包率。

## 参考论文

Khamechian, M. B., Kozyrev, V., Treue, S., Esghaei, M., & Daliri, M. R. (2019). Routing information flow by separate neural synchrony frequencies allows for “functionally labeled lines” in higher primate cortex. Proceedings of the National Academy of Sciences, 116(25), 12506-12515.

DOI: https://doi.org/10.1073/pnas.1819827116

## 许可证

MIT License. 详见 [LICENSE](LICENSE)。
