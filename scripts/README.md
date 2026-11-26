# 代码目录说明

所有脚本按用途分三类：`runner/`（跑实验与评估）、`analysis/`（事后分析）。
文件名使用能直接读懂含义的英文名，不再使用 c1/a1 之类编号。

## runner/

### 数据

| 文件 | 作用 |
|---|---|
| `data_fmnist_prep.py` | 下载 Fashion-MNIST、展平 784 维、构造长尾版、缓存为 npz。**纯 numpy 解析 IDX，不依赖 torchvision** |
| `data_clusterlet_partition.py` | 碎片化划分：每个类在特征空间用 K-Means 切 4 个子簇，再贪心分配给 10 个客户端，保证零样本重叠 |
| `data_client_builder.py` | 把上面两者组装成作者 `main.py` 期望的 `(K, [DataLoader], (X,y), sizes)`；同时支持调用作者原版 `make_noniid_clients` 作为对照划分 |

### 打补丁（不改作者源码）

| 文件 | 作用 | 对应线路 |
|---|---|---|
| `patch_client_local_gmm.py` | 把 `pre_train` 里"用标签定 K 再拟合 GMM"换成我们的局部 GMM 构造 | 客户端 DPGMM |
| `patch_server_sampling.py` | 给 `gmm_sampling` 加采样下限、给 `ELBO_Loss_server` 加责任度下限 | 服务器端重加权 |

### 训练

| 文件 | 作用 | 对应线路 |
|---|---|---|
| `run_fmnist_clusterlet.py` | 在 Fashion-MNIST 上跑一次完整联邦训练（300 轮），含逐客户端损失记录与 NaN 扫描 | 碎片化基准 |
| `run_fmnist_dpgmm.py` | 同上，但客户端局部 GMM 换成 DPGMM 或同 K 普通 GMM | 客户端 DPGMM |
| `driver_three_partitions.py` | 三组 × 三种子：平衡版·碎片化、长尾版·碎片化、长尾版·作者划分 | 碎片化基准 |
| `driver_latent_width.py` | 潜空间 10 / 20 / 30 维各 3 种子 | 潜空间宽度 |

### 评估

| 文件 | 作用 | 对应线路 |
|---|---|---|
| `eval_clustering.py` | Global ARI / Purity / NMI，10 个推理种子 | 全部 |
| `eval_coverage_matched.py` | **修正后的覆盖率**：真实数据对半分为参考集与查询集，比较"生成集"与等大真实集的最近邻距离比值。阈值、样本量、类内几何全部对齐，内建自检（生成集=参考集时必须精确为 1.0） | 覆盖率指标 |
| `check_data_precondition.py` | **训练前必做**：每类 1-近邻纯度、集中式 KMeans/GMM 的 ARI、划分统计。用来判断该数据是否支持要测的现象 | 全部 |
| `precheck_client_dpgmm.py` | 离线拟合 DPGMM，统计各 prior 下每客户端保留多少个分量 | 客户端 DPGMM |

### 诊断

| 文件 | 作用 | 对应线路 |
|---|---|---|
| `diagnose_mode_loss.py` | 查"某些类生成不出来"发生在哪一环：服务器分量构成、客户端局部 GMM、潜空间可分性、每分量期望采样数 | 丢失定位 |
| `diagnose_decoder_reach.py` | 查解码器能否渲染某类：重构误差、oracle 覆盖率（把真实样本编码再解码，绕开 GMM 采样）、服务器解码器与客户端解码器对比、逐类 PCA oracle | 丢失定位 |

## analysis/

| 文件 | 作用 | 对应线路 |
|---|---|---|
| `server_multica_core.py` | 服务器端层次聚合的完整实现：对角高斯 2-Wasserstein、尺度归一化 Bures、complete-linkage 凝聚、加权矩匹配、宏/微两级保留 | 服务器端层次聚合（Multi-CA） |
| `acd_local_experiment.py` | ACD 局部实验：调用 Fed-HIRE 的 FCPL，用矩估计把簇归属换算成 `(π, μ, σ²)`，含六组对照与接口验证 | 客户端自适应聚类（ACD） |

## 运行顺序

评估与诊断类脚本都是只读的，可以任意顺序跑；训练类脚本之间的依赖是：

```
data_fmnist_prep  →  data_clusterlet_partition  →  data_client_builder
                                                        ↓
                              check_data_precondition（先确认数据可用）
                                                        ↓
                                     run_fmnist_clusterlet / run_fmnist_dpgmm
                                                        ↓
                                    eval_clustering / eval_coverage_matched
                                                        ↓
                                    diagnose_mode_loss / diagnose_decoder_reach
```

**注意**：`run_*` 不修改 `FedVaDE_source/`，它把四个 `.py` 复制到 run 目录后才打补丁；补丁内容通过 `patch_*.py` 施加。
