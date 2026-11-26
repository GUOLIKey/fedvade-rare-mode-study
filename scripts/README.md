# 代码目录

```
common/     九份共享模块：距离、指标、模型、采样、数据、训练循环、局部 GMM、评估
lines/      八条实验线，每条一个文件夹，内含 main.py 与 README.md
```

**一条线路一个文件夹，主逻辑写在 `main.py`，数据与评估这类通用部分放在 `common/` 复用。**

---

## common/

| 文件 | 内容 |
|---|---|
| `data.py` | Fashion-MNIST 下载与缓存、长尾构造、标准化、Clusterlets 划分、客户端构造、对半分 |
| `distances.py` | 最近邻距离、自身最近邻、1-近邻纯度。全部用 Gram 恒等式，避免广播产生的巨型临时数组 |
| `metrics.py` | ARI / Purity / NMI / ACC |
| `model.py` | 从 checkpoint 构造作者 VaDE、取后验均值、算责任度、分量统计表 |
| `sampling.py` | 从服务器 GMM 采潜向量、解码、一步生成 |
| `local_gmm.py` | DPGMM 拟合与剪枝、同 K 普通 GMM 对照 |
| `train.py` | **唯一的联邦训练循环**：复制作者源码→打补丁→注入钩子→跑 300 轮，含逐客户端损失、NaN 扫描、checkpoint |
| `evaluate.py` | 对一个 run 算 Global ARI / Purity / NMI（10 个推理种子） |

> `train.py` 覆盖了原来两个 runner（普通 FedVaDE 与 DPGMM）的全部功能，
> 通过 `--local-gmm author|dpgmm|samek_gmm` 切换；采样与损失补丁通过 `--local-gmm` 之外的开关施加。

## lines/

| 文件夹 | 线路 | 问的问题 |
|---|---|---|
| `fmnist_benchmark/` | 碎片化基准 | 换稀有类可分的数据集，还会不会丢类 |
| `latent_width/` | 潜空间宽度 | 覆盖是不是被 10 维潜空间卡住 |
| `client_dpgmm/` | 客户端 DPGMM | DP 先验能否替代"用标签定 K" |
| `client_acd/` | 客户端自适应聚类 | ACD 切簇后转成高斯参数能否改善全局 |
| `server_multica/` | 服务器端层次聚合 | 解析合并替换重采样 |
| `server_reweight/` | 服务器端重加权 | 强制稀有分量被采到 |
| `coverage_metric/` | 覆盖率指标 | 自造指标可不可信 |
| `diagnostics/` | 丢失定位 | 生不出来的类，问题在哪一环 |

每个文件夹的 README 写明：问的问题、怎么运行、得到的结论、以及对应的报告文件。

## 运行顺序

```
common/data.build_dataset                    ← 先准备数据
  ↓
lines/fmnist_benchmark  check                ← 训练前必做的可用性检验
  ↓
lines/*/main.py  run|driver                  ← 训练（内部调用 common/train）
  ↓
common/evaluate.evaluate_run                 ← 聚类指标
lines/coverage_metric/main.py                ← 生成覆盖指标（含自检）
lines/diagnostics/main.py                    ← 定位诊断
```

## 原则

1. **不修改 `FedVaDE_source/`**。训练脚本把四个 `.py` 复制到 run 目录后才打补丁，原文件保持不变。
2. **共享代码只在 `common/` 写一份**，线路文件只做组织与命令行。
3. **自造指标必须内建自检**（见 `lines/coverage_metric`）。
