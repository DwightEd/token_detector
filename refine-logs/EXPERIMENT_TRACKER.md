# 幻觉检测执行 Tracker：完整900评分与最终评价完成

**快照：2026-11-04 05:42:47，Asia/Shanghai。** 全900评分、恢复、独立无标签审计、最终CPU评价及同轴诊断均已完成。评分和审计阶段未读目标真值；全部分数封存且审计通过后，最终评价已按冻结合同打开官方标注。本版有真实自然检测指标，外部NLI仍只是比较器，新的主线无监督方法未准入。

本轮结果已写入研究记忆F054、决策D050及[恢复点](../research-state/RESUME.md)，所有active_jobs为None。正式结果见[完整900检测报告](../docs/results/h017_nli_detection.md)及[结构化汇总](../docs/results/h017_nli_detection_summary.json)；固定错例见[公开分析](../docs/results/h017_nli_error_cases.md)及[对应JSON](../docs/results/h017_nli_error_cases.json)。下一问题以本机`outputs/h017-relation-20261103/NLI_NEXT_QUESTION.md`为准，新内部方法/模型实验仍NOT_ADMITTED。

对应[同版计划](EXPERIMENT_PLAN_20261104_054247.md)；[04:52续跑快照](EXPERIMENT_TRACKER_20261104_045230.md)与更早历史保留。

| 实际阶段 | 身份 | 状态 | 实际结果/边界 |
| --- | --- | --- | --- |
| M0–M4机制/资格历史 | 见04:10历史tracker及结果 | DONE / 原边界保留 | M3所测K路径负结果、M4单例描述及严格FP64 inconclusive保持；不重跑 |
| M5五来源范围对照 | f47b032f首轮、7cf99af7续跑 | DONE / supported_scope |5model/5head=2复用+3新、三主差及独立audit通过；约3GiB无OOM，不是自然AUROC或新风险方法 |
| Data2txt900完整规划 |b1c3c43b… | DONE / sealed |900答/150来源/6生成器/8240原完整句/8247测量unit/178343原targets，K18162、unknown0、512最大窗口 |
| 原评分v1 |8902c781… | STOPPED / preserved |801完整预测+partial802；成功16160模型结果，额外guard1在forward前拒绝，失败root55d00a…不删除 |
| 兼容计划v2 |caba7e54… | DONE / supported_scope / 0 MODEL |seal6c9441f2…，原单位/协议/modelidentity保持、140源/旧AST/运行时资格通过 |
| 显式评分续跑v2 |c6b4520fb08d4fcf70988292c2e88a05b5f10b83f3b5fb0433e5d0522e863dcb | **DONE / complete900** | 复用16160旧成功raw含partial802，仅新增2003；完整score seal a12c6a29…、sidecar93c3d9f6…；原801预测/完成unit保持byte一致，无新拒绝/无新canary |
| 独立nonlabel全量审计 |audit SHA0673c5621ad0256d38616c0be40fc22d5ea09b372f42ace20cf7d68d2df75f37 | **PASSED** |900/8240/178343、所有原轴/窗口/完整3类raw/归一化/风险投影/复用与实际成本通过；0模型、0目标真值值读取 |
| 第一次GPU环境最终评价 |560445fca9273ceca5d833827fead2800df976128f677d4e20b36bcc59847cc5 | FAILED / preserved | 缺sklearn在读标签前失败；不是方法成绩，不删除，不安装/修改原环境 |
| 已有CPU环境最终评价 |542e64150afd88823e516c1206673a092c31515f7aef0f8f2dd8189eeca8e284 | **DONE** | 既有evaluate_dataset，2000次source bootstrap/seed17/native；完整178343轴AUROC0.744306164、AP0.097168289 |
| 同轴R4/V3对照与固定错例 |7d56ec25d9d615ea952d09692418ab917f2fdfa1c78d560bb07f2ae332fcde25 | **DONE / posthoc** | 复用原封存R4/V3、统一8918报警预算；配对CI和固定FN/FP/5670时间例保留，不重新评分/拟合 |
| 新代码完整本地安装CPU suite |installed045716尝试 | **INCOMPLETE** | 明确300秒timeout，约17%；无assertion failure不代表全套通过。29相关测试仅是局部范围 |
| 精确新代码CI | root待完成 | **PENDING** | 不能挪用旧9dc6600 CI1101pass给新代码；root负责最终检查/发布 |
| 新主线无监督内部方法 | 无新准入 | **NOT_ADMITTED** | 不救NLI、不追加M5世界/换头层，不把外部监督模型或本批错例补丁当新机制；六任务各0.8未完成 |

## 真实检测表

Data2txt TEST900、150source，所有178343原有效token保留，已暴露人口仍为exploratory。三方法统一全局top5%报警8918token：

| 方法 | AUROC | AP | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| R4（旧封存） |0.798317534 |0.190392333 |2088 |6830 |5319 |
| V3（旧封存） |0.755715886 |0.145209394 |1720 |7198 |5687 |
| DeBERTa NLI（外部监督对照） |0.744306164 |0.097168289 |1038 |7880 |6369 |

NLI−R4 ΔAUROC=-0.054011，95% source配对CI[-0.079074,-0.029124]；NLI−V3=-0.011410，CI[-0.036933,0.016237]跨0。AP点估计低于两者，完整区间见[正式结果](../docs/results/h017_nli_detection.md)和[结构化汇总](../docs/results/h017_nli_detection_summary.json)。不把AUROC理解为“全部正误都判断对”；FP仅指没有发布正跨度重叠，不证明严格事实正确，NLI按句广播不提供唯一坏token定位。

只有Data2txt有本轮新NLI成绩，其他五任务没有。外部MNLI/FEVER/ANLI监督来源、不同观察器/精度/上下文预算完整披露，不把这个分数当项目无监督新方法。

## 成本与失败恢复

合计16160旧成功forward+2003新forward=18163个成功物理模型结果；另有1次旧pre-forward资源拒绝，物理接口总数18164。合并predict字段/stdout含cached replay，不是恢复进程重新执行18163次。旧2canary只作资格复用，本次没有fresh-resident canary；原阈值/FP32/batch1/512窗口/512MiB门限保持。unknown句/token/回答均0，仅为覆盖结果，不表示0漏检/0误报。

framed输入token7473887，逐forward时间合计492.396秒；不含tokenization、恢复验证、I/O封存、独立审计和评价，不作为端到端时延。原失败时host536465408B低于512MiB门、GPU仍空闲6354370560B；是主存保护线触发，不是GPU OOM。旧失败、partial802、原801预测与成功raw均保留，没有重复成功模型调用。

## 下一动作及禁止晋升

1. 固定FN/FP与原S/raw的事后分析完成后，只记录证据、未知与不确定性；不调报警阈值、类别映射、max、窗口、source格式或融合权重，不重跑本批NLI救分。
2. 下一候选仍未准入。零模型证明只以5682/13603（not=65、WiFi=67）与同源5684（or=65、lot=66）补齐source许可字段、条件/极性闭合及已有物理观测匹配，复用库存与已知缺失，不重清点、不拿QA层15或回答历史局部attention替代来源观测。必须给出区别词法/位置/回声的预测与直接文本核验之外的原轴价值；否则停止，不追加模型。
3. root完成精确新代码完整安装CPU/CI检查及已授权发布。现有结果已经封存，不因测试、文档、bootstrap或环境切换重评分。

六任务各AUROC≥0.8、新无监督机制、顶会水准或内部量增益均未达成。MiniCheck新下载/网络重试仍停止，当前不自动启动任何新GPU实验。
