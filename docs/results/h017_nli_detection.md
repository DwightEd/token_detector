# H017：完整900回答的文本核验检测评价

本轮已完成实际幻觉检测评价，不是角色问答准确率。全部900条原回答、150个来源、6个生成器、178343个原始有效token均保留；真值只在完整评分封存并通过独立无标签审计后打开。

| 方法 | AUROC | AP | TP | FP | FN | Recall |
|---|---:|---:|---:|---:|---:|---:|
| R4（历史冻结分数） | 0.798318 | 0.190392 | 2088 | 6830 | 5319 | 28.1896% |
| V3（历史冻结分数） | 0.755716 | 0.145209 | 1720 | 7198 | 5687 | 23.2213% |
| DeBERTa NLI（外部监督对照） | 0.744306 | 0.097168 | 1038 | 7880 | 6369 | 14.0138% |

所有方法统一使用全局最高5%的8918个token报警；TP/FP/FN来自同一原轴与官方字符标注。AUROC不是二元判断正确率，FP表示未与发布正跨度重叠，不能据此证明整句事实正确。NLI按完整句广播分数，不能宣称唯一错误token定位。

## 不确定性与任务范围

以下为NLI减基线的150来源成组配对bootstrap，2000次，seed17；没有重新生成R4/V3分数，也没有融合。历史测试人群已暴露，结果仍属探索性证据。

- 相对R4：ΔAUROC=-0.054011，95%CI=[-0.07907388299266993, -0.0291244363775072]；ΔAP=-0.093224，95%CI=[-0.11936316200037513, -0.06783771923853059]。
- 相对V3：ΔAUROC=-0.011410，95%CI=[-0.0369327598931804, 0.016237054215237093]；ΔAP=-0.048041，95%CI=[-0.06431088430941354, -0.030842269639716608]。

只新跑Data2txt。QA、Summary、FinQA、TATQA、HalluRAG没有本轮NLI结果；六任务各0.8的无监督目标尚未完成。这个模型在MNLI/FEVER/ANLI等外部任务上受过监督训练，只作比较器，不能包装成本项目的新无监督机制。不同观察器、监督目标与上下文预算也不能解释成单一模块的因果效应。

## 固定方法与实现

1. 保留全部已有完整句scope，把原8247个测量分块按既有句边界归并为8240句，不重新调用生成模型抽取事实。
2. 每句作为hypothesis，完整来源分窗覆盖；每对至多512 token，假设不截断，重叠为min(64,来源预算//4)。不做样本或窗口去重。
3. 调用主线HFEntailment.prepare_pairs/classify与三分类归一化。句风险为1−max(window entailment probability)，neutral并入不支持风险；原token由唯一所属句接收分数。
4. 复用myproject.measure.token_scores、原预测校验/封存与experiments.evaluate.evaluate_dataset，不引入新评价内核。阈值、投影与人群均在开真值前冻结。

源码：[collector](../../experiments/nli_dataset.py)、[共享NLI后端](../../experiments/nli_qualification.py)、[主线评价](../../experiments/evaluate.py)。安装可用`pip install ".[experiments,nli]"`；CLI见`r4-experiment transfer predict-local-nli-dataset --help`。

## 资源中断、恢复与成本

原进程完成801条和第802条部分结果后，主存余量536465408B低于512MiB保护线；显存仍空闲6354370560B。这是forward前拒绝，不是GPU OOM。原失败完整保留；显式恢复全部16160成功raw，仅新增2003次，旧两次canary不重跑，也不冒称fresh resident资格。
合计18163次成功forward、7473887 framed输入token、逐调用计时合计492.396秒。此时间不包含分词、恢复验证、文件封存、审计和评价，不能当完整端到端时延。另有1次旧资源拒绝，物理接口共18164次；本轮无新拒绝。unknown token/句/回答均为0。

Windows主要等待来自数万小文件的校验与封存；模型与数据已在本地，不依赖重新下载。精度、内存门限、窗口和评分规则未降低或调整。

## 固定错例诊断


[逐例原文、三类概率与限制](h017_nli_error_cases.md)：营业时间错误风险已达0.968276，但被更高的概括/否定句风险挤出预算。来源明确WiFi=no，正确否定句仍被判矛盾0.853761；三个固定FP整句均无发布正span，不能都归因于真错误向邻词的句级广播。这里只识别比较器的排序与语义判断问题，不解释原Llama/GPT的生成成因，也没有证实max窗口是因果原因。

NLI相对R4的AUROC下降置信区间不跨0；相对V3的AUROC差区间跨0，不能称显著下降；AP相对二者均下降。不能因这份较弱的DeBERTa对照而宣称所有直接文本核验无效或内部方法已有新优势。

选例预先固定为每类按原始顺序的前三个不同来源；完整机器诊断保存在本地`local-nli-datatxt900-diagnostics-v1/examples.json`，包含NLI FN/FP、互相挽回的错例和5670小时条件案例。事后解释不能用于本批参数搜索、重选阈值或丢弃困难token。

## 证据与复核

[结构化结果及完整SHA索引](h017_nli_detection_summary.json)。生产者为外置安装的140文件wheel；完整旧前缀、801份预测字节、partial802、全部原轴/窗口、三类logit归一化与风险投影均由独立无标签脚本核对。原Llama机制观测的60次forward单独记账；本轮18163次为不同的DeBERTa观察器。

评价第一次误用了无scikit-learn的推理环境，import阶段即退出、未读标注；失败560445f保留。随后使用既有CPU sklearn1.9.0/NumPy2.4.6环境另claim542e6415完成同一冻结规则，未安装依赖或重跑模型。本机完整CPU测试达到明确300s上限，仍记未完成；源码提交[`b362eff`](https://github.com/DwightEd/token_detector/commit/b362efff7ddb8e3806f04fd9ddb9bf0fd1969bd7)的[完整Linux Tests CI](https://github.com/DwightEd/token_detector/actions/runs/41724862885)已通过，不使用历史CI或局部测试替代。[验证收据](h017_nli_ci.json)。
