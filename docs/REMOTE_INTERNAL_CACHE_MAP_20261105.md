# 远端内部缓存与同题多采样：2026-11-05 更正

此前库存没有覆盖 `research/reanchor`。不能据旧 F039–F042 或仅检查 `token_detector` 的结果，断言远端没有内部表征。以下按实际文件区分，下载与计算资格是不同状态。

远端根：`/share/home/tm902089733300000/a903202310/lys/`。网页文件存储对应 `tm902089733300000/a903202310/lys/`。本机统一根：`C:/ResearchData/token-detector-remote-20261103/06_same_question_multisample/`。

| 远端相对路径 | 数据与真实保存量 | 当前状态 |
|---|---|---|
| `research/reanchor/outputs/samples_20260911_145421_235/` | RAGTruth QA，source 14304/14315/14325/14375；同一 Llama-3.1-8B-Instruct，seed 0/1/2/3，temperature=.7、top_p=.9、最长512；16个原始生成NPZ，含全部32层32头注意力、token IDs、prompt length、top5、chosen logit、完整词表log-normalizer | 19文件含3份元数据，1,168,475,270 bytes，已完整下载及逐文件SHA；本机子目录 `reanchor_samples_20260911_145421_235` |
| `research/reanchor/outputs/states_samples_20260911_145421_235/` | 对同16个生成序列做固定前缀重放；hidden[33,N−1,4096]FP16，覆盖prompt及R[:-1]；末slot为finalnorm后，不等同于原生成hidden或Q/K/V | 17文件/2,401,783,169 bytes下载及资格完成；原轴/签名与24共同prefix控制通过，无新native重放；本机子目录同远端basename |
| `research/reanchor/outputs/attributed_samples_20260912/` | QA source14375的seed0/1；完整A[32,32,N,N]、V[32,N,8,128]、residual[33,N,4096]、post_attention/MLP[32,N,4096]，BF16计算/FP32存储，N730/681；末residual为finalnorm前 | 34文件/6,509,800,561 bytes全部下载，原manifest SHA及保存数组数值资格全部通过（各自原轴/BF16/因果/残差舍入）；旧原权重bodySHA未证明。没有Q/K或W_O权重，不是全16例完整graph |
| `research/graph/outputs/same_question_path_conflict/` 与 `_focused_v2/` | 派生的轨迹、head_source_writes、effects及报告；input_settings指向以上原始多采样 | 只用metadata追溯，不按旧effects/错误标签选样、选头或调方法 |

当前恢复源码表明原生成先得到logits/attention再抽样，保存raw logit/LSE，温度和top_p用于采样分布。源码与旧运行之间尚须核历史身份，不把当前文件存在当历史源码SHA已证明。固定prefix hidden属于另一次重放，不能冒称原生成hidden。

## 如何使用

1. 先检查 `00_manifests/` 下下载计划、逐文件ledger和completion，核大小/SHA及实际headers。沿原token轴读取，不重新生成回答。
2. 全部保留4source×4seed。共同prompt和首次分叉之前的共同prefix是身份负控；不同回答的第t个token不能硬对齐。所有层和头先保留。
3. 注意力、V、source主体归属、层间状态和原token概率分别解释。高关注或稳定状态不等于关系正确；没有Q/K/V或MLP输入输出的文件不能用于伪造相应量。
4. 这16条是重新抽样的回答，**不能复制RAGTruth官方原回答的token标注**。资格核验/机制探索不能报新的检测AUROC。原始自然任务的冻结评分和评价仍走 `experiments` 主线。

三批合计70文件、10,080,059,000 bytes，约9.39GiB；raw只在C，不push Git。原16记录共1532个生成决策、13种不同R轨迹；source14325四个seed的9-token轨迹相同。原生成与replay在各自共同前缀控制上通过，不能因此把replay当原时刻hidden。

H018此前实际运行的是Data2txt来源13603下5682/5684两条官方原回答及5684的来源字段交换，**不是这16条QA多采样**。target已独立审计，但必要主方向反证，该固定分支关闭，剩余control不再推理；[反例与边界](results/h018_owner_specificity.md)。S0与已完成target不得重跑。完整900 Data2txt旧比较仍见[检测结果](results/h017_nli_detection.md)。

另恢复 `reanchor/outputs/ragtruth_population_20260912/` 的元数据与官方5684探针：声明17,790答，但尚非本机逐文件全量清点。5684原R相同，旧P769、当前H018 P803；只存全层source/remote attention池统计、message/MLP norm及IDs/offsets，没有raw V/hidden/MLP向量。桌面canonical归档的train manifest2497答，生成器仅llama-2-7b-chat，保存post-token稀疏attention且floor=.01；不能用它补出被删除的小权重或冒充prechoice完整表征。

## D盘整理

已迁移并SHA校验 `D:/Desktop/ragtruth_llama31_8b_canonical.tar.zst` 到统一根的兄弟目录 `07_local_archives/`，12,222,647,079 bytes，SHA256 `c18627668b5702f0b0a364674ad37389fc904572fa0108359e2b6ed35a6b61cc`。尚未由文件名推断包内实际张量，未在D盘解压。

已迁移OEM驱动**安装源** `D:/Drivers` 到 `C:/DiskCleanupArchive/20261105/Drivers`，1,687文件、5,428,914,435 bytes，逐相对路径/大小核对；Windows当前驱动文件未操作。两项释放约16.4GiB。C盘对应目录有迁移收据。研究缓存、模型、科学raw及失败记录保留。


## 保存数组资格与跨graph精度边界

三类资格索引已更新至v2：[索引](data/multisample_qualification_20261105.json)、[graph2资格](data/multisample_graph2_qualification_20261105.json)。184,942,592个残差加法坐标在BF16_RNE后完全一致；FP32直接相加可差.75。graph residual末slot为finalnorm前；replay末slot为finalnorm后。

H019另对已封存的64个小ΔV表做精度诊断；同P535不同N730/681下1536向量只有48相等、1488不同，maxabs.078125。各自产物资格不证明跨graph深层sourceV相同，不能把跨回答幅度差仅归因query/head routing。它不隔离full-N数值kernel/capture/历史模型身份，也看不到individualV共同平移。[完整有限机制结果](results/h019_head_carrier_association.md)。
