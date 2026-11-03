# 当前恢复：回到已有回答的幻觉检测；角色问答支线已关闭

**用户最新纠正/D032：需要的是幻觉检测效果。** 输入是原始来源/实际可用问题及已有回答，输出为该回答原评价轴上的风险分数。验证器的角色问答正确率不能代替检测效果。后续先读 [DETECTION_OBJECTIVE.md](DETECTION_OBJECTIVE.md)；QA/NLI可作为内部组件，但必须接入原回答评分并直接评价自然数据检测表现。

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

- 检测问题：[FNFP_MECHANISM.md](../docs/source-reading/FNFP_MECHANISM.md)、[PERFORMANCE_REVIEW.md](../docs/source-reading/PERFORMANCE_REVIEW.md)；完整事实F001–F038和先前实验索引在本目录。
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
