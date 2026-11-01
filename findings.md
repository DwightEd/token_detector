# Research findings

## 2026-11-02 理论与源码审计

- 固定片段的交换混合专家后验严格保留原full-gap排序；不能靠调Beta先验改善同片段AUROC。证明与有限CPU核对见docs/source-reading/THEORY.md。
- 来源依赖、专家责任和provided-source支持性不是同一个随机变量；已有精确积分不识别事实后验。
- pooled-AUC可精确拆成同片段与全部跨片段正负对；分解已进入封存评价报告，保持原分数。
- 默认Observer后端有eager/SDPA差异，但fullgap collector已有匹配SDPA实现，历史运行身份须按plan/artifacts核查，不能一概否定。
- 21个主库文件完成逐行讲解；主库外77个原文件及新增工具/测试仍pending。独立代码审查窄范围通过，完整149 passed/1 skipped/9 subtests；无新自然数据检测收益或venue readiness。
