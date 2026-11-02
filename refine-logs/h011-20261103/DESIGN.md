# H011：固定回答擦值问题的来源重建桥

本文件在任何 H011 神经观测前登记；仅资格实验，不是自然检测效果或论文新颖性主张。遵守目标标签只在封存评价时释放、不调模板/权重/温度、不删样例的约束。

## 唯一问题和竞争解释

H008 在同3B模型、严格toy前端下，S定义的16问题全部重建正确。尚未知道问题从R而来并承载完整回答后缀时是否仍可恢复该能力。H011固定一个擦值问题编译器；不重新验证A/B/C焦点核验、不重跑P021的原R续写、不搜索score readout。

H007 inference完整48原答、8来源、两PERSON槽。已知事件角色保持正确、来源明确是双射；这些是构造域前提。所有两人完整字面occurrence擦成目标[TARGET]和另一人[OTHER]，原事件、标点、空白、0/4/12正确后缀完整保留。问题固定为：

    Using only the source, give the full person name for [TARGET] in the masked statement. [OTHER] denotes a different source person.
    Masked statement:
    {masked_response}

候选只由S列出两个人名，排序固定。S解析器不向题面导出真绑定或替代模型分数。原R人名仅用于擦除和事后对齐，不在问题prefix中重新暴露。'[OTHER]不同人'是显式toy前提，不能推广自然场景或冒充唯一编辑位置识别。

## 完整人口、冻结和成本

- 48答×2槽×2候选=192逻辑completion实例；两个R方向擦除后题面逐字相同，48唯一Q/96完整姓名+EOS请求。保留全部原答和96槽映射。
- 人名残留、占位符冲突、重叠、未知别名/语法/后缀或不能精确坍缩时，保留unknown审计且整包资格门失败；不得先筛好例再forward。
- 沿用H008完整候选/EOS和float64全词表normalizer、无缓存、固定3B NF4/FP16/SDPA。全部96tokenmetadata及实际messages先冻结，之后才forward。
- 对已封存H008消息、候选、完整tokenframing和数值合同做精确请求身份审计。消息不同才新增模型请求；不能因protocol/sourceversion不同就宣称旧请求不可复用。至多96新增forward，无模型/提示/参数搜索。
- 唯一必要对照复用封存H008全部16Q/32complete原结果，不重跑，不把H007不同接口当匹配对照。

## 固定评价和停走

主结果逐burden0/4/12各16唯一QA的correct/wrong/tie/unknown、每层全部sourceflip8与eventflip8的完整候选rawLL和方向。保留poolmass、候选token长度、未知与成本。次级为全部48答/96槽固定risk及构造AUROC/AP；48答不是48独立重建。只有分数、实现和CPU评价冻结后才读/hash既有H007 design/truth与H008参考。

0负担失败则说明S-defined→R-masked接口迁移失败，不据此说基础模型不懂来源；不换prompt救分。若仅大负担失败，说明该QA接口也对后缀敏感，未唯一识别内部机制。全部通过也仅进入自然问题锚、候选完整性/别名/算术/非断言审查，不晋升自然检测器。H009四臂都16/16没有增益，不再加大score偏置。

先前审查和文献重叠见 outputs/h010-natural-span-20261103/NATURAL_CONTRACT_NEXT.md；QAGS/FEQA/Q²/QuestEval/QAFactEval已有QA核验与实体遮蔽思想，本实验不能声称这些是新方法。
