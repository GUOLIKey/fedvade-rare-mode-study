# 客户端 DPGMM

**问的问题**：用 Dirichlet 过程自动决定局部簇数，能否替代作者"用真实标签定 K"。

方法（DPGMM 拟合、剪枝、同 K 普通 GMM 对照）在 `common/local_gmm.py`；训练循环在 `common/train.py`。

**子命令**

| 命令 | 做什么 |
|---|---|
| `dump` | 跑 50 轮预训练，导出每客户端 `local_mu`，停止 |
| `precheck` | 离线拟合 DPGMM，统计各 prior 保留多少分量 |
| `driver` | 对选定 prior 跑完整 300 轮（可含同 K 对照） |
| `eval` | 计算 Global ARI |

**结论**：DP 先验从 0.001 扫到 100 **完全无差别**；每客户端保留 11–15 个分量，
全局 131–132 个，服务器每轮每分量只能分到 0.76 个点 —— 与 ACD 失败机制相同。**未进入完整训练。**
详见 `docs/reports/18_dpgmm_precheck.md`。
