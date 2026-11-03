# 幻觉检测执行 Tracker

**快照：2026-11-04 04:10:29，Asia/Shanghai。** 当前有一个真实完整900例外部NLI评分job；未完成、未评价。新主线无监督方法仍未准入，六任务各0.8未达成。对应[同版计划](EXPERIMENT_PLAN_20261104_041029.md)；此前计划与tracker原文保留在timestamp及before-update版本。

| 实际任务 | run/证据身份 | 状态 | 真实产物与边界 | 下一动作 |
| --- | --- | --- | --- | --- |
| M0三任务结构库存 |80913833 | DONE / supported_scope |2700答、450来源、5806显式字段；散文语义unknown，不是自动事实gold | 复用，不重复 |
| M1 QA缓存3+897重放 |5f7cf0ad、48f5ce11 | DONE / supported_scope |900答资格通过、五坏实现拒绝；不是检测准确率 | 复用 |
| M2 QA原轴3+897对齐 |7d065e2e、8d6662cb | DONE / supported_scope |900答、124817原R token；语义所属未自动许可 | 复用 |
| 原模型prefix/head/cache/no-op |1fdf3766、7f222ddf | DONE / supported_scope | 数值仪器/路径资格，不是新风险或效能 | 不重复资格调用 |
| M3四来源Q/K/V路径 |24dd6e0b | DONE / refuted_scope |48 full model/44 head；所测数字K选择性最终输出介导门失败；不推出所有K无角色信息 | 负结果保留，不换头/层救结果 |
| M4自然时间补充观测 |31185253 | DONE / descriptive_scope |1 full model/1 head；ago低概率、首选BC。生产FP32身份通过，严格FP64附加界inconclusive保留 | 不等于旧计划全人口检测M4 |
| M5五来源范围对照首轮 |f47b032f349bf2fc75baaeed2a84c0253ddf375f0033041cbbc48e5f05c89cc6 | PARTIAL_RETAINED / I/O failure | native与S_minus两组2model/2head原样保存；进度文件禁止覆盖触发，非OOM | 两组已独立核查并精确复用，0重算 |
| M5其余三来源续跑 |7cf99af794c5b21a02c6654694827c9b178b8e356cae6eb0afa6305fbeb9e3a3 | DONE / supported_scope |3新model/3head；连同2复用组成完整5+5，三项固定主比较全过；独立925概率与全层AV/attention审计通过 | [结果/图表](../docs/results/h017_scope_results.md)已归档 |
| LOCAL DeBERTa资产盘点 |LOCAL_DEBERTA_ASSET_INVENTORY_20261104.json，8bef1cfa… | DONE / seven_files_pinned |380001591B、七SHA全核；不需网络，不以旧34例目标成绩选模型 | 使用新full900合同，旧34保持独立 |
| Data2txt900规划 |local-nli-datatxt900-plan-v1 | DONE / sealed_tokenizer_plan |900答/150来源/8240完整既有句/8247旧测量unit/178343原targets；K18162、unknown0、最长512 | 规划通过不等于模型评分已通过或0错误 |
| Data2txt900 FP32 NLI评分 |8902c7819378d476ac0c2a78e31eb8bbd4d6284598ae7c31ddbbe8d4dc623a4f | **RUNNING** | 固定18163 classifier forwards，NO dedup；两次最长frame同FP32接口canary通过，adapter结果复用一次。04:10日志247/900、5235calls，仅快照 | 等完整分数/seals；不得把局部结果当AUROC |
| 完整900分数独立审计 | 未申领评价；待score seal | WAITING | 原id/原token/offset/唯一句子membership、全部chunk风险、unknown、code/model/plan、调用/费用 | 不跑模型，不读取目标标签值 |
| 完整900主线最终评价 |evaluate_dataset，2000bootstrap/seed17/native | WAITING | 标注文件已仅核存在/hash；完整178343轴AUROC/AP、预算FN/FP、coverage/cost，和封存R4/V3同轴表 | scores与审计完成后才开值 |
| 新主线无监督检测器 | 无新方法准入 | **NOT_ADMITTED** | M5 margin/有限source变体不是原输入单次风险；DeBERTa外部训练比较器也不是新无监督方法 | 冻结可验证新机制、风险接口/coverage/cost后再审查 |

## 已完成M5的解释与资源

三项预定S_plus减周一、周二、周六例外的主差分别3.313881、3.300304、2.382473，均过约0.0185的冻结工程界。五世界全部保留，不能择优展示。该结果拒绝此有限assay的严格全局完整时段计数不变解释；位置/顺序、来源评论时效、后文Sunday声明以及自动前端/自然覆盖仍未解决。不是通用范围核验电路或检测AUROC。

真实总调用5model/5head，2组复用加3组新算，没有再次算五组。显存实测约3GiB，无OOM；首轮103.90秒、续跑149.79秒含各自启动。初次停止仅为I/O编排错误，原失败和输出保留。独立审计进程及M5模型进程已退出。

## LIVE NLI合同和最终评价入口

- 输入原S和既有完整R句子；完整source-token窗口、hypothesis不截断、512含special；旧64token cap的7额外片段只按既有scope恢复，不新分句/LLM改写。
- 原目标risk为`1−max_chunk P_entail`，neutral同非entail；预定未知0.5保留完整分母，不能把资源/模型/数值错误填unknown冒充完成。planning unknown0仅指分词规划。
- 本次为固定外部MNLI/FEVER/ANLI监督DeBERTa，FP32 GPU/batch1；不等同旧FP16 pilot，不与R4/V3盲加权、不充作内部新模块。
- scoreroot：`outputs/h017-relation-20261103/local-nli-datatxt900-score-v1`；planroot同目录`local-nli-datatxt900-plan-v1`。partial预测正在产生；截至本快照尚未有完整completion/score审计或自然AUROC。
- 标注路径：`D:/projects/research/outputs/fusion-cache-20261102/prepared/test_Data2txt/annotations.jsonl`；SHA `d6b2f44d12d68b520e5a28dd4d642b03a94854a2ec5b208a3ceee3f73be2b9d2`匹配原manifest。只hash未解析值；既有evaluator在score seal后严格join900 roster/response SHA。
- 下一顺序固定：full score seal → 独立code/source/score/axis/cost审计 → 主线AUROC/AP/FN/FP/coverage/cost与R4/V3表。bootstrap或审计不触发再次模型评分。没有新增MiniCheck下载/重试。

## 费用与状态不能混写

M5的Llama full/head计数与当前NLI classifier计数分开。当前NLI18163是预算，5235是04:10已记录进度，不是最终费用；实际wall/GPU/host峰值待完整run与审计。6GiB allocator cap和512MiB host/GPU floors是执行合同，不是全程实际峰值。旧缓存概率、M5模型条件概率及外部NLI entailment分数属于不同观察器/算术合同。

旧C1/C2与完整自然检测M4的历史计划不自动成为新run许可。root登记、内容封存和实际完成收据优先；本tracker只记录这一时间快照，不改任何模型、运行状态或研究记忆。
