# 2026-11-26 修复交付与恢复说明

本次完成已授权的P1–P4：保全、文档修正、代码修复、保存权重重评。没有新增正式研究训练，没有GitHub推送，没有修改作者源码或历史实验输出。

## 代码与证据位置

- Mac代码：`/Users/cheeseowl/Desktop/科研/fuxian/reproduction_runs/audit_repair_20261126/code`
- Mac原研究仓库同步入口：`/Users/cheeseowl/Desktop/科研/fedvade-rare-mode-study`
- 服务器代码：`/root/autodl-tmp/fedvade_project/reproduction_runs/audit_repair_20261126/code`
- Mac私人备份与日志：上述Mac修复目录的同级`backups/`、`evidence/`。大文件不进入Git。
- 服务器原始材料仍在`/root/autodl-tmp/fedvade_project/`，禁止以修复目录覆盖它们。

## 核查结果

11项回归测试通过；六种CPU小型集成运行通过；CUDA小型运行完成且无非有限值。相同toy配置下，未改动历史runner与修复runner的服务器/客户端参数逐位相同，指标差为0。这些是工程验证，不代表重新完成300轮研究训练。

9个Fashion-MNIST基准模型的旧口径ARI与原metrics完全对齐；另做每记录一次口径，共18组评价。15个模型各3个生成种子，共45次生成评价。33份最终逐模型JSON均自动记录相同代码包指纹，数值摘要与先前复核一致。见[报告20](reports/20_saved_checkpoint_reevaluation.md)。

真实数据缺类对照推翻了“新ratio在1–1.5就可排除类别丢失”：删掉凉鞋类仍约1.10。此反例不证明实际FedVaDE必然丢类。见[报告21](reports/21_metric_missing_class_control.md)。

原decoder“完全相同因此可排除聚合问题”的说法也撤回；单模型参数检查发现差异，不能仅凭重构误差接近推断因果。两划分的微小浮点处理差异已标注，影响不确定。

## 保全范围与能力

- 清单内1515个历史原件修复前后SHA一致。
- 1484个轻量源码/配置/日志等文件已备份Mac并校验。
- 27个服务器模型评价状态导出、额外1个C13第300轮导出、3个Fashion-MNIST缓存已复制到Mac并SHA校验；导出时与原模型状态逐张量核对。
- Mac原有C13-s1完整基线checkpoint与服务器原件SHA一致。

这些评价状态支持冻结模型重评，不保存完整客户端、优化器、调度器或随机数状态，不能替代服务器完整checkpoint或承诺精确续训。原始完整checkpoint继续保留在服务器。详情见`results/audit_20261126/backup_integrity_final.json`。

## 使用与恢复

数值检查使用已有服务器环境，不在Mac安装CUDA依赖：

```bash
cd /root/autodl-tmp/fedvade_project/reproduction_runs/audit_repair_20261126/code
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /root/autodl-tmp/fedvade_project/.venv-gpu/bin/python -m unittest discover -s tests -v
/root/autodl-tmp/fedvade_project/.venv-gpu/bin/python scripts/common/evaluate.py --help
```

重评必须显式指定run、data-dir和新的output；旧重复记录与unique_full不能混报。训练入口和可用选项见[scripts说明](../scripts/README.md)。不要用rsync --delete同步研究目录。

灾后重评恢复时，在新目录中组合`backups/server_evidence/`保存的原run配置/实际source、`backups/evaluation_backups/`中的对应导出和`backups/fmnist_cache/`。依据清单保留相对映射、验证SHA后使用显式路径，勿以新导出冒充原始完整训练状态。作者交付PK的原始格式与导出wrapper不同，不可盲目替换。

## 仍然缺少的东西

已检查的原run中部分第300轮权重未找到；Fashion-MNIST缓存到原始train IDX的索引映射未恢复；干净环境重建未测试；ACD/Multi-CA若干历史流程仍是归档脚本而非通用CLI。这些限制在复现说明中列明。

碎片化的因果影响、init×flow贡献、方法端到端多种子效果仍需另立受控训练方案；本次没有以猜测补足这些结论。
