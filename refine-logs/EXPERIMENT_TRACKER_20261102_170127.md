# 性能优先任务表

日期2026-11-02。允许多模块，目标各主任务AUROC≥0.8；真实标签仅冻结后的最终评价使用。

| ID | 任务 | 状态 | 运行条件／已有证据 |
|---|---|---|---|
| S001 | R4/V3六主任务与Hallu三组统计 | RECOVERED_NOT_RERUN | R4完整Hallu TEST已恢复；旧DEV132单列 |
| S002 | 真实FN/FP来源与机制审计 | SOURCE_GROUNDED_AUDIT_IN_PROGRESS | 当前优先事项：原始来源／回答／标注／测量及传播阶段，不先设计新分数 |
| P001 | 恢复成对token分数／观测 | RESTORED_FIVE_TASKS | 2992答／448202原token；score/native全部记录hash及轴匹配；Hallu旧R4分数仍缺 |
| P002 | 固定等权秩融合CPU验证 | COMPLETED_REJECTED | 五任务固定0.5＋0.5；金融AUROC/AP相对V3区间均负，未晋升 |
| P003 | 冻结基础模型语义核验器合同／采集 | PROPOSED_NOT_RUN | 未采集、未评分、未评价；先完成真实FN/FP机制审计 |
| P004 | 完整评分封存、最终效果与消融 | FIVE_TASK_EVAL_COMPLETED | R4／V3／固定融合均已封存评价；Hallu及核验器／模块消融仍未执行 |
| U001–U003 | 四条件操作／预测诊断 | OPTIONAL_SUPPORTING | 可辅助模块设计，不作为唯一允许路线 |
| R001–R008 | 旧关系组合 | DEFERRED_RELATION_ASSUMPTION_UNRESOLVED | 仅闭合语义合同下可选 |

六主任务历史点值：R4为2/6、V3为3/6，最好点值包络4/6并非新方法。已完成固定CPU基线在五个已评任务中仅1项点值达到0.8，Hallu待配对缓存；全部达标尚未实现，未采用该基线。

执行证据：[归档负结果](../docs/source-reading/CPU_FUSION_REVIEW.md)、[公开汇总JSON](../docs/results/cpu_fusion_evidence.json)。本地缓存：`D:/projects/research/outputs/fusion-cache-20261102/`；实际运行：`outputs/rank-fusion-cpu-20261102/`。用户最新优先事项是讲清真实FN/FP成因，不把加权包装成新方法。
