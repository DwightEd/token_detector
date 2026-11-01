# 独立审查与修订记录

reviewer：次级Codex代理，请求/编排报告`gpt-5.6-sol`、`xhigh`；与执行者同属模型家族。正面结论至多`provisional`，不代表跨模型接受或顶会认可。没有单独运行时部署身份报告，不虚构不同模型。无taste anchors，calibration=none。

| 轮次 | 主要结论 | 本轮处理 | 分数 |
|---|---|---|---|
| 0 | 保序/协方差推导成立；诊断不能代替检测器；近期USP/SHAM/时序融合有重合 | 修订目标为来源支持性；核对原论文；收缩新颖性；增加AUC精确分解 | 未给分 |
| 1 | 普通fullgap最终JSON漏掉计算出的分解；缺非零复制的独立oracle；段数命名易误解 | 修复最终序列化并加测试；增加copy/refresh有限枚举；明确known-token段数与跨段补集 | 未给分 |
| 2 | 26项指定CPU测试独立通过，主要修复成立；提示全unknown序列化None风险 | 增加防护及第27项fixture；核实真实evaluate_dataset会先拒绝无可评价群体，未把它说成已发生回归 | 未给分 |
| 3 | 窄范围代码检查通过；独立复现27 passed | 确认普通JSON/seal输出、None防护与真实无标签拒绝路径；科研门槛仍待独立证据 | 未给分 |

执行者补充核查发现：默认Observer和fullgap collection不是同一测量路径。前者present eager/absent SDPA；后者已记录matched_sdpa并调用两世界同SDPA的measure_responses。reviewer在后续轮次复核并收缩先前宽泛判断。没有仅因为reviewer同意就跳过实际代码核验。

完整请求、原始回答、meta与hash保存在本地`.aris/traces/research-refine/20261102_run01/`，亦复制到桌面材料。对外公开的是本摘要及完整方案，原始历史会话/账户信息不进入Git。

尚未通过：新的自然人类来源支持性确认、冻结的source等权主估计量、完整unknown/纯片段/边界覆盖、逐运行观测身份核查、预算匹配和足够新颖性。重复文案修订不改变这些状态。
