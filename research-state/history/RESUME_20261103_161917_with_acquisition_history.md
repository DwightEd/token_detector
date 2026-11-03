# 当前恢复：关系错绑的内部计算设计；数据收尾并行

**最新指令D036优先：先思考关系错绑如何检验，不把改进限定于V3。** 读 [RELATION_BINDING_DIRECTION.md](RELATION_BINDING_DIRECTION.md)、[编码前科学准入](relation_misbinding_admission.json)、[当前计划](../refine-logs/EXPERIMENT_PLAN.md)。H017仅为已审查设计，未claim、未产生新模型运行/分数；active_hypothesis仍为空。目标是原回答幻觉检测相对直接文本核验的定位/检测/成本价值；H015/H016问答支线不恢复。

四量分开：观察器条件概率、数值关注、事实主体/条件归属、持续性。先查条件接收位置到prevalue的事实所属主体，区分条件失败、检索切换、值/读出失败及post-token回声；不假定正确关系原向量距离更近。M0来源许可/unknown与M1RoPE/GQA/因果/精度资格先于解释。M2观测进入M3只需不同预测与对照spec，真实角色交叉与M3统一补采；最终机制门在联合结果后。旧错例只作研究动机，不按目标标签挑头/参数。

**下载仍继续，先查实际状态，勿重复启动：** `outputs/remote-complete-download-20261103/rag-mirror-state-v3.json` 已传完23.04GB；收尾器 **acquisition-finisher-v2-state.json** 自动接续v4/核验，最终 **acquisition-finisher-v2-final.json**。rootwide mirrorlock 与PID/startUTC必须一致。v1收尾器因读取契约纠错被正常停止，证据保留；不是停止payload下载。C根 `C:/ResearchData/token-detector-remote-20261103/`，51,812selectedfiles/30.82GB；00含索引与核验。observer-config另补56KB并匹配原metadata SHA。

**合同纠错必须保留：** Q/K是RoPE前，不能raw点积当attention。source阻断作用于实际执行0–15层，只保存15层；nodes/query仅回答T+1行，缺完整来源Q/hidden和early-stop最终logp。Llama观察器teacher-force已有R，不是GPT/Mistral/Llama2原生成器内部原因。Summary/Data的lag8量是局部回答历史attention，非同范围来源QKV。原公开交接中的“仅15层阻断”以此纠正，不能沿用。

实验代码在experiments，候选diagnostic在experiments/research，复用主线原轴/基线/封存/评价。memory负责身份去重和证据，不自动判科学创新；每个新模块编码前补科学准入/审查，真实入口与SHA齐后才check/claim，标签最终评价。三旧日期integrity脚本的当前副本已删，H002完整副本不动；installed-wheel全CPU验证进行中，不自称通过。

下方D034/D035为下载历史范围快照；进程/完成以v2收尾器实际state+final为准。

**用户当前指令/D034、范围修正D035：先用RAGTruth所需数据，统一分类保存。** 数据根 `C:/ResearchData/token-detector-remote-20261103/`，当前缓存读取依赖清点已闭合为 **51,812文件/30,817,905,041字节（约30.82GB）**；不是整个远端镜像或严格重算依赖。`00_manifests` 保存分类索引、不可变清单和核验收据。原D盘历史缓存保留，不改sealed plans或seal。

**实际进程先核对，禁止重复启动：** 活动镜像 `outputs/remote-complete-download-20261103/rag-mirror-state-v3.json`；根级 `.http-mirror.lock` 绑定它的PID。排队补充为 `manifest-RAG-v4-additive.json`。有界一次性收尾器状态 `acquisition-finisher-state.json`，最终结果 `acquisition-final-result.json`（均在同一outputs控制目录）；先读实际文件/PID/started_utc/manifestSHA，不能根据此快照推测已完成。收尾器待审查/启动，若状态文件不存在需核对最新交接，不能自称已在运行。

核心v2 **16,824文件/7.36GB已全部下载并通过11个互不重叠前缀的清单/SHA/原封存引用/header核验**；416远端目录二次指纹也全部未变。报告 `verify-core-v2-20261103-153626/`。v3 **29,277文件/23.04GB传输中**：三任务当前TEST各900个完整原生10文件包及450来源；QA TEST900 paired block15坐标。v4 **5,711文件/420.26MB排队**：Summary/Data2txt TEST各900份attention NPZ/JSON及参考；P005/P006的196个TRAIN父输入包/99来源，687原SHA引用已核齐。

05_graph_inputs中的QA坐标是第15层native/该层source-read-blocked双世界，含residual/attention-write/MLP-write、Q/K/V、lag8注意力；不是全部层。Summary/Data2txt已找到的是原生标量和局部attention，未确认同范围QKV。06_auxiliary_train单列196TRAIN对照（QA64/Summary67/Data2txt65，与TEST0重合）。旧197GBfullQA主产物只有DEV1008/FIT1295、TEST0/900，是PROGRESS停在2303且无FREEZE的未完成运行；本轮只保留7根元数据，204GB旧v1范围已经废弃，勿恢复传输它。

本阶段只恢复文件、核哈希/头信息，**没有新模型调用、方法优化或检测分数**。先冻结方法/参数/全部分数后才可读取真值最终评价。H015/H016问答支线保持关闭；复用已知负结果，不换名重跑。完成实际文件核验后再设计原回答幻觉风险的判别实验。认证材料独立`...-access`，不得入数据包/Git；保留至少15GiB空闲；Windows长路径仅工具内处理，不改系统。

**用户最新纠正/D032：需要的是幻觉检测效果。** 输入是原始来源/实际可用问题及已有回答，输出为该回答原评价轴上的风险分数。验证器的角色问答正确率不能代替检测效果。后续先读 [DETECTION_OBJECTIVE.md](DETECTION_OBJECTIVE.md)；QA/NLI可作为内部组件，但必须接入原回答评分并直接评价自然数据检测表现。

**用户新增问题/F039/D033：远端哪些数据集保存了内部量？** 先读 [INTERNAL_STATE_INVENTORY.md](INTERNAL_STATE_INVENTORY.md)。已核2700RAG及356finance主缓存headers仅delta/route/原轴；另有P005/P006的196RAG小面板末层hidden/写入向量历史落盘。旧myPorject fullroute/vector目录仅查到父listing，字段未核；Hallu原NPZ未下载。先恢复和核对已有独立特征的manifest/header，不据本机缺文件就说远端没有，也不直接重跑模型。没有新评分或新QA实验。

## 当前状态

- 没有新实验在运行，没有新run登记，active_hypothesis为空，所有H015/H016控制器已结束。不要自动重跑旧请求。
- H016/F038完整结束：8来源toy、312前向，四种角色正确数9/16、16/16、15/16、16/16；仅作辅助诊断，不是自然数据检测准确率或六任务提升。保留原观察/评价seal及失败记录，停止新增独立角色问答实验，不再把该支线作为主线。
- 当前尚无优化后新检测器的六任务全量结果。六任务各AUROC≥.8仍未达成，不能用case排序或某任务最佳配置包络替代完整检测器。
- 上次代码与数据交接已push：69aead38079e5f4c52afa9123c96f9eaa376f2d5。其GitHub完整CPU CI run41578874802实际成功：992passed/1skip/9subtests；201.68s。receipt在outputs/h016-paired-scope-20261103/github-ci-success.json。本机180秒全套超时仍是incomplete，不改写成pass。
- 最新目标纠正是文档/状态更新，尚未产生算法或分数变化。未来需保持源码、预测和评价身份分开。

## 下一步准入

1. 复用已有自然数据FN/FP机制审计与完整缓存；不要重数已知错例作为新研究进展。
2. 围绕一个尚未解决的错误成因设计一个无标签模块和必要对照，先明确如何产生原回答的token/span风险分数、适用范围和unknown。
3. 冻结方法、配置、人口、观察预算及全部分数后再打开目标真值。主结果是六个声明任务逐项AUROC/AP、原分母、coverage、成本，及固定规则下FN/FP/召回；面板结果明确只是面板。
4. 若模块只能提升内部问答而不能形成可运行的原回答检测评分，停止追加其模型实验，补齐接口与论证。自然数据的重复值、合法多对一、null与作用域不得硬套双射。

## 必须复用的记录

- 检测问题：[FNFP_MECHANISM.md](../docs/source-reading/FNFP_MECHANISM.md)、[PERFORMANCE_REVIEW.md](../docs/source-reading/PERFORMANCE_REVIEW.md)；完整事实F001–F039和先前实验索引在本目录。
- P002残差/cosine、P003删token依赖、P005方向/whitening、P021来源重连、P026repair/reset等旧方向不得换名重跑。原等权融合已拒绝。
- H004/H005/H006自然面板结果负或混合；H012只有自然擦值库存，QA准入0。它们不能当作新方法已有效。
- H007删suffix严格复用旧b0，0新调用；H011/H014与H016均保留为已完成记录。H013严格概率门失败未晋升成功；H015仅工程资格。
- 六任务旧R4/V3 AUROC：QA .8929/.8470；Summary .7621/.7623；Data .7983/.7557；Fin .6617/.8027；TAT .7110/.8430；Hallu answerable .807066/.7669。未被H016更新。
- 五组成对缓存2992答/448202原token；Hallu主要891答，旧R4原数组缺失、V3 full_index未全恢复。不得换新token轴或用full_gap冒充R4。
- FRED原生成问题未恢复；已暴露人口继续是探索性。真值仅最终评价，不能由已知错例拟合权重/阈值/prompt。

## 换机与制品

远端输入/native/score/model清单见 [REMOTE_EXPERIMENT_HANDOFF.md](../REMOTE_EXPERIMENT_HANDOFF.md)。近期本地H006/H015/H016云端位置未确认。桌面15_H016_Paired_Scope_20261103/START.html含全部32原NPZ和181sealed文件，约105MB；这些原始材料保留，不需重复生成。

主项目D:/projects/research/token_detector；CPU .venv，GPU独立env token-detector-cu128。用户允许正常关闭WPS，主窗口已退出；主VSCode/Codex、网络和系统进程保留。继续前核对实际状态，不根据旧PID或留存lock推断任务仍运行。

持续目标未完成；用户授权的无监督检测、合理模块设计/验证和换机交接继续有效。旧goal工具blocked不代表当前研究已完成或获准改目标，不重建/假称complete。
