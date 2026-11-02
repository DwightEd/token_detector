# 本轮结果与后续门槛

2026-11-02。原始代码基线1210120；本轮保留主库算法/默认方法，只扩充评价诊断、测试与说明。完整方案见round-2-refinement.md，原始问题锚点保持不变。

**已经落实**：主库21个Python文件、2421行、128函数/嵌套定义逐行讲解；方法流程、张量/缓存/位置轴、默认与可选分支的统计含义；交换共享保序与导数推导；AUC分解；已有proof_revision的受限语法与KL投影解释。新增27项测试，完整149 passed/1 skipped/9 subtests，另有三个受限语法/KL fixture。独立次级review的最终窄范围代码检查通过，同家族结论只标provisional。

**研究迭代带来的修正**：不再把来源敏感性称事实后验；不再认为调Beta先验可以改善同片段排序；不把监督response-level TPA当token基线；纳入USP、SHAM-Bench和近期时序融合的重合；明确source bootstrap不等于source等权点估计。修复新评价的最终JSON漏项，补全非零复制oracle，核实fullgap有匹配SDPA路径，收缩了reviewer起初过宽的后端混淆判断。

**未成立**：没有新的自然支持性标注、没有新的真实检测效果、没有完成新算法的新颖性确认、没有达到或证明顶会就绪。保序定理、KL投影及AUC身份是正确计算与解释工具，不被包装成原创检测突破。

**尚未完成的逐行范围**：原清单主库以外77文件/508函数，及本轮新增两测试和渲染工具；这些明确pending，不能把本轮讲解扩写为整个仓库全部完成。下一阅读顺序建议：experiments/run.py与pipeline.py → data/datasets/evaluate.py → fullgap_dataset.py与local_evaluation.py → span_resolution.py →其余候选与历史实验驱动。继续时先读SOURCE_INVENTORY.json核对hash，不重新讲解已经完成的主库。

**继续研究的具体顺序**：先核验每次历史结果的sealed plan、源码与观测身份；冻结source等权主估计量、未知/纯片段/字符边界覆盖；在未参与开发的来源上收集人工supported/contradicted/absent/uncertain标注；以匹配预算/后端的raw/shared/一个序列模型/LUMINA评价局部对比，监督CORTEX单列。若新增语义verifier，先对比verifier alone与语义保持重写，不直接拼成新默认模型。

研究状态保留evidence_pending，未把代理评分当科学验收。4次review（0–3）没有强制数字分数；默认最多5轮的限制没有被用来制造“9分通过”。缺少新外部证据时继续反复润色，不会推进剩余科研门槛。


2026-11-02用户进一步确认：坚持无监督，真值只用于最终评价。当前方案与状态已按USER_CONSTRAINTS.md收紧；历史锚点/轮次保留原文，标注训练与风险校准分支退出主线。此次是用户约束确认，没有新增模型运行、审查评分或科研结果。
