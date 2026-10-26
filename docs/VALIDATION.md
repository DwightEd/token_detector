# 工程验证记录

记录时间：2026-10-26T15:31:03+08:00。对象为当前独立仓库，以下均为本次实际运行结果。

## 已完成

- 构建 wheel 并以 `pip install --no-deps --no-index --target …` 安装至独立目录；未修改共享 Python 环境。
- 从项目目录外运行安装包的两个 CLI，导入路径确认指向独立安装目录。
- 对最终安装包运行全部 CPU 测试：**115 passed，9 subtests passed**，pytest 耗时 19.97 秒。包括微型随机 Llama 的真实前向、来源观测/路由、原 token 位置、标签隔离、断点续跑、数据适配和评价。
- 主线 `myproject/*.py` 与三任务 `references/*.npz` 均与初始 `ASSEMBLY_MANIFEST.json` 哈希一致。
- 使用安装包和显式外部历史路径复验：参考数组、原 token 身份和最终分数的 **dtype 与字节均完全一致**。

| 任务 | DEV token | TEST token | 参考、身份、分数 |
|---|---:|---:|---|
| QA | 147,981 | 124,817 | 逐位一致 |
| Summary | 132,721 | 121,248 | 逐位一致 |
| Data2txt | 213,197 | 178,343 | 逐位一致 |
| 合计 | 493,899 | 424,408 | 918,307 token |

- 使用最终安装包，从现有的全部 **11 个 HalluRAG 官方 test pickle** 导出 1,782 条记录，与原 JSONL 按原生 ID 逐条比较，内容完全一致。随后 prepare-dataset 得到 1,782 条、81 个来源；1,779 条句标注可对齐，3 条仍保留不可对齐状态。未修补标注或改写回答。
- 准备 manifest 已绑定导出 receipt，receipt 记录官方成员与导出文件哈希。
- 两轮独立代码审查未发现 Critical/Important 问题；已补充 dtype/字节检查、导出 receipt 的 provenance 和中断恢复说明。

## 运行环境与证据

Python 3.11.15；NumPy 2.4.6；PyTorch 2.8.0+cu126；Transformers 4.57.1；
scikit-learn 1.9.0；pytest 9.1.1。上述运行设置 `CUDA_VISIBLE_DEVICES=-1`，无 GPU 推理。
最终 wheel SHA256：`fd97eea0a27df4f7c87cd70bc9c2c4c6a9d94ea384db4de8cfb301cf6a21a108`。

本机详细命令、退出码和输出日志位于 Git 忽略的 `artifacts/final_package/`：
`installed_checks.json`、`frozen_parity.json`、`hallurag_export_equivalence.json`。
这些是本机验证产物，不随源码发布；复验命令见根 README。

## 范围边界

冻结结果复验读取已有观测与分数，不读取真假标签；没有重跑完整 8B 观察模型，
没有产生新数据集 AUROC，也不是新方法提升的证据。
GitHub Actions 已配置但尚未在远端执行；本次使用现有研究环境的依赖，
不声称完成了空白机器上的依赖下载与安装。代码许可证仍未指定。
`ASSEMBLY_MANIFEST.json` 保留初始组装记录，新增工程文件以当前源码和本记录为准。
