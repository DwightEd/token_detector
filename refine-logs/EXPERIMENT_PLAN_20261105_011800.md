# 细粒度绑定研究：先补既有知识，再验证新干预

2026-11-05 01:18，Asia/Shanghai。无监督原回答检测、六任务接近或超过.8和完整可验证方法仍是目标，尚未达成。当前F059/F060、[缓存地图](../docs/REMOTE_INTERNAL_CACHE_MAP_20261105.md)及[H018反例](../docs/results/h018_owner_specificity.md)为准。

## 待支持的主张

1. 细粒度source关系计算包含超出词面、位置和整体source显著性的可区分信息。最低证据是有竞争预言的绑定专属干预、语义/同义与数值负控；generic patch能改数值不够。
2. 该信息能改善原回答的定位、检测或成本。必须有冻结无标签方法、全轴分数、最终标签评价与模块消融；未有新risk，不能报新AUROC。

| 顺序 | 已有数据/成本 | 进入下一阶段的门 |
|---|---|---|
| 恢复知识 | 新reanchor三批70文件10.08GB已下；0模型 | 原16+replay16已资格；2完整graph核数值合同。恢复旧ownership/Relation终局、N/A/G失败及I02未运行入口，不能重演旧Vswap/source-null |
| 新物理观测 | 同q/GQA head×完整source carrier的双边marginal保持关联擦除；仅proposal | 必须和旧X/E/XE/endpoint/suffix明确区分。保持原轴、原token条件化、source词面/总mass；先独立审查与memory claim，不直接解释为truth |
| 语义许可入口 | 先核I02 source-first协议是否确实未跑，以及位置/同义/unknown与合法多对多关系 | 无可靠语义许可不能包装检测器；若文本前端已完成判断，必须证明内部量额外收益，否则仅机制诊断 |
| 完整检测/消融 | 仅机制及原轴接口准入后 | 全样本和unknown保留，评分冻结后开标签；新16QA采样不能套官方旧R标签，六任务仍各自报告 |

H018 target的必要方向已resolved反证，剩余control无需推理；原S0/target/全部失败均复用，不翻号选点。旧源码151 exact installed、65targeted通过；完整CPU300s incomplete，当前批次push另核完整Linux CI。新方法未armed模型预算。
