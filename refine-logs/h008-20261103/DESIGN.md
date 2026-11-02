# H008：来源独立的有限候选多 token 重建资格

状态：**纯 kernel 与无模型语法可行性已验证；source-QA 模型观察未执行。** 本文不是效果报告或主方法晋升。实际模型观察由 root 在独立 prior/代码审查、协议和运行身份封存后决定。本轮不搜索模型、模板、温度、长度惩罚或混合权重。

## 主张与判别问题

H007 已否定该冻结 A/B/C judge 的“仅靠精确焦点即可抗回答负担”解释。H008 改变被观测的随机变量：在读取原回答槽值前，用 **来源 S + 从 S 编译的问题 Q**，对两个人名的完整候选序列计分，得到来源答案分布；之后才把分布映射回原回答的两个实体槽。

拟检验的唯一主张是：在这个明确、无歧义的构造域内，固定观察器是否具有能同时通过 **source flip 与 event flip** 的 source-only 重建能力。不是重新调 A/B/C verbalizer，也不是把原回答在改写来源下重新 teacher-force。成功仅允许说该新观测在这些 toy 关系上通过必要能力资格；不推出自然数据有效、原生成器内部因果、六任务各 AUROC≥0.8 或新颖性。

一个关键限制必须先说明：严格语法 parser 已能读出这些 toy 关系的逻辑绑定，因而规则本身可以回答 toy 问题。H008 有意只导出人名/事件集合来构建全部候选请求，**不把 parser 的正确答案放入 QA 字段或替代模型分数**。这只是测量神经 QA 仪器，不能拿 toy parser 的完整覆盖冒充自然语义覆盖，也不能据此证明此域需要 LLM。

## prior 审查与不重复边界

已逐项读取 `research-state/prior_experiments.json` 中 31 项的实际状态与新信息定义，并定向复核 P021/P026、I02、H002/F011/F012 及相关实现。

| 既有方向 | 与 H008 的关系与禁止重做的部分 |
|---|---|
| P001、P007、P009–P020、P030 | 同一概率世界的归一化、先验、共享、持久性或权重读出。H008 不改这些读出，不融合旧风险。 |
| P002–P006 | 隐状态、删除依赖、source-write 等新测量，不是有限候选完整姓名的来源 QA。不给旧测量改名。 |
| P008、P022–P025、P027–P028 | 有限数字/关系证书、约束、语法前端及覆盖限制；不能将少量新模板说成广泛自然覆盖，也不重跑旧 Gaussian、KL 或因子核。H008 仍受这一局限约束。 |
| **P021** | 已完成并拒绝的关系 null/source-world 实验。`experiments/relational_transfer.py` 先 `encode_response(sample.response)`，再在原/空/重连来源下 `score_continuation(..., response['answer_ids'])`；它测的是原 R continuation 的 likelihood。H008 请求不接收 R、不重排或修补来源，teacher-forcing 的目标是从 S 列举的姓名候选，而不是原 R 的历史。共同使用 teacher-forcing 不构成相同观测。 |
| **P026** | 已做 repair/reset 与 source × response-repair 的固定未来 likelihood 交互。H008 不修改 R、不重放未来 R token，也不把删后缀当改善。 |
| P029、P031 | 外部观察器比较和旧 broad-verifier 提案；后续 H003–H007 的 direct judge 已经实测。不能用 P031 旧条目的 proposed 状态声称 judge 未测。 |
| **I02、H002** | source-first/answer-blind 思路已有提案先例，不是本轮新原理。H002 报告明确 direct/blind/leaky source-QA 调用均为 0：失败的是自由位置/角色提议前端，不能把它说成 source reconstruction 失败。H008 用程序精确槽位绕过前端，首次资格测量的范围限于本 toy 域。 |

结论：所核查记录中没有发现“全部 S-event 问题先于 R 封存、两姓名完整候选加 EOS、再对齐原 R 两槽”的已完成观测。此结论只是本地 prior 准入边界，不是全领域新颖性证明。原 H002 自然 v2 保留原问题上下文；本 H008 是完整明确的构造域 source-only 问题合同，不应擅自迁移到缺失问题的金融材料。

## 域、完整覆盖与纯 kernel

输入仅使用 H007 **inference** 的原 48 行，不打开 construction truth 或任何自然 benchmark 标签/预测。来源必须完整匹配以下有限语法，包括全部固定 12 条 filler 及顺序：

- winner：两个不同全名参与两个不同 Cup final；每场有明确且唯一的胜者；胜者在两人间形成双射，禁止自反参赛或第三人。
- sole_signer：两个不同 Accord，各有唯一全名签署者；两事件与两人形成双射。
- 所有其他子句、缺失 filler、额外人员、重复事件、非唯一绑定或未支持措辞均为 `unknown`，不猜测、不修复。

`experiments/research/source_reconstruction.py` 是纯函数 kernel：

```python
parse_source(source)                 # people/events/relation；全 source 消费
source_queries(source)               # 每个合法 S 的两个事件问题
candidate_requests(source)           # 两问题 × 两规范人名 = 四请求
align_response(source, response)     # 之后才读取 R：首句两个 exact 字符槽
candidate_distribution(query, candidate_logprobs)
```

成功 `status='ok'`；未支持输入或不合法候选分数返回 `status='unknown'` 和 reason。人名与事件都按源码字符串排序；候选顺序不随 S flip、R flip 或首次出现次序改变。R 模板以确定性正则完整匹配首个 `focus.response_units` 单元，槽位来自捕获组原生 start/end，不自由生成 offsets。同一姓名重复出现在两个槽时保留两个不同位置；源外人名/事件、重复事件和歧义则保留 unknown。

无模型实际可行性核对：48/48 回答、8/8 独特来源全部通过；2 家族各 4 来源，得到 **16 个独特 S-event query、32 个候选请求、96 个 exact 实体槽**。这是构建与覆盖事实，不是模型准确率。

## QA target-value leak 审查

来源及两名合法候选必然包含与原 R 相同的人名，不能声称整个输入字面上不含这些字符串。约束是 **不得从 R 或其另一槽值向 QA 新增线索**：

1. 先将输入行投影成仅含 S 的 source records。source planner 不接收 R、masked R、claim、另一个实体槽、factor cell 的正确方向、construction truth 或优选候选。
2. 对全部 8 来源的全部 2 事件建立问题和全部 2 候选，完成字节级 manifest/请求封存，之后才调用 `align_response`。不让 R 选择“要问哪个事件”。原始 48 行 hash 可作顶层溯源，不进入模型 messages 或 source-QA execution key。
3. 问题固定为 `Who was the sole winner of the {source_event} final?` 或 `Who was the sole signer of the {source_event}?`，**question 不含任一候选人名**。messages 的事实 payload 只有 `source` 和 `question`，不含 masked claim、answer prefix、优选项或候选列表。候选作为两条独立强制 completion，均计分。
4. 变换任意 R flip、0/4/12 filler 或换成源外人名，S 不变时 source query、候选、完整 messages、EOS policy 与 observation key 必须逐字节不变。源外 R 只影响事后 alignment 状态。

这建立固定来源条件下的程序数据流隔离，不保证模型不利用来源位置、姓名习惯或预训练知识。source 本身给出真实绑定，是合法证据；不得另加 `correct_person` 字段。

## 最小 `score_fullcandidate` 后端合同

接口建议 `score_fullcandidate(messages, candidate, eos_policy)`。root/另 agent 实现 backend 与 collector；本 kernel 不加载 tokenizer、权重或执行模型。必须固定 H007 同一模型/runtime 身份后再比较观测合同，不自动把辅助 7B 加进本 run。

记 chat prefix token 为 P，候选姓名 token 为 C，明确 EOS 为 E=`tokenizer.eos_token_id`。每个候选只评分完整字符串加 **恰好一个 E**，不加别的 special token、标点、解释或长度惩罚。冻结 tokenizer revision、P/C/E IDs 及完整拼接对齐；`encode(prefix+candidate)` 必须保持 prefix 边界，不能静默截断或改名。

一种避免 shift 错误的实现是输入 `P+C`（不把 EOS 送进输入），目标为 `C+[E]`。若 `m=len(C)`，从模型保留最后 **m+1** 个位置的 logits，依次预测第一枚候选 token 到 EOS。`use_cache=False`、eval、单完整 prefill；逐位置全词表稳定 log-softmax 再 gather。**现有 `score_labels(..., logits_to_keep=1)` 不能复用来当多 token 分数**。也可取全 logits 后正确切片，但必须保存对应位置并通过手算 toy shift/EOS 检查。

后端最少返回：P/C/E 与 input/target IDs 或其精确 hash、每 token log probability（含 EOS）、完整序列和、输入 token 数、scored completion 数、生成 token 数=0、真实 forward 数、耗时、峰值和 runtime/model/source 身份。不要伪装成 A/B/C 的 next-token observations。

对于事件 j、两候选 a/b，固定

\[
\ell_j(c)=\sum_{t=1}^{|C_c|+1}\log p((C_c,E)_t\mid S,Q_j,(C_c,E)_{<t}),\qquad
q_j(c)=\frac{e^{\ell_j(c)}}{e^{\ell_j(a)}+e^{\ell_j(b)}}.
\]

归一化只使用两条完整序列；不除 token 长度、不调温度、不裁剪 epsilon、不用目标真值校准。此 q 是条件候选分布，不能当全答案空间质量或校准事实概率。缺失/非有限 logprob、EOS 不一致、前缀不对齐和超长都要保留工程 unknown，不丢槽/丢行。

事后每个 R 人名槽可记录 `1-q_j(claimed_name)`，完整保留 96 个槽和原 48 行。此时尚未定义或实现联合整答概率、非姓名 token 风险或全任务探测器；两个槽不应未经独立性证明就相乘。

## 最小判别与先验抵消的准确范围

主资格须同时检查每个词汇 scenario 的两来源方向 × 两事件，逐格报告全部 **16 个 query** 的候选 log odds、顺序和 unknown。只在 source flip 上改变回答还不足：总复制来源首个人名的策略可能让首事件看似成功，但会在另一事件失败。event flip 是必要保护；不靠挑一个事件或平均后掩盖失败通过。

设规范人名为 a/b、两个事件为 e0/e1，`L[s,e]=ell[s,e](a)-ell[s,e](b)`。双差

\[
D=(L_{0,e0}-L_{0,e1})-(L_{1,e0}-L_{1,e1})
\]

消去与 source/event 绑定无关的 additive 姓名/长度先验，以及各自不随另一因子变化的项。其预期方向必须由固定生成规则和规范事件顺序预先确定，不能看分数后翻符号。**这种抵消在线性 log-odds/完整平衡对比中成立；不能直接声称 normalized q 的任意平均也严格抵消**。source×event×候选的其他词汇交互、位置和预训练偏好仍可能存在，不能称已去偏或已定位因果。

R flip 保留两候选对称的正确/错误槽映射。R burden 不影响 QA 输出是设计隔离的数学结果，不是又一次实测的“抗负担成功”。新的经验问题是来源 QA 是否正确识别两事件绑定。H007 已有 0-burden/judge 观察只作封存参照；不重跑、挑模板或宣称等成本匹配。

## 顺序、成本与停止边界

1. **已完成，无模型**：31 prior 记录核查、纯 kernel、21 个自包含逻辑测试（0.12 秒）；仅 inference 的实际 48/96 与 8/16/32 构建检查。未打开 construction 目录。
2. **root 准入前**：独立审查 source-only 数据流；tokenizer-only 核对所有候选多 token/EOS 与边界；用手算 logits 的 fake backend 检查第一 token、EOS、位置切片和整序列和。封存全部 16 queries、32 requests、原 48 行及 96-slot map、unknown 规则和模型/runtime 身份。
3. **root 明确冻结后**：一次性完整观察；串行单候选 full-prefill 估计 **32 unique forwards**。32 是请求结构估算，不是已发生的前向、耗时或显存事实；若 batching/cache 改变物理调用数要另外记录。不得因早期分数改模板、候选顺序、EOS 或重试未知。
4. 全部预测与成本封存后才开放构造真值核对 source/event 双翻转与 48 槽映射。若必要能力失败，保留负结果；不把 H002 前端失败当作本轮已知结论，也不以换名重开 P021/P026。通过也只允许进入下一项独立审查，不能跳到自然覆盖或主方法晋升。

## 本轮精确证据身份

- H007 inference inputs：`0d298932778c5ad60a43b059cc860b31679e7649350a825f5b1de72d2585cf66`。
- prior31 registry：`43d20b4e67e806915dd4876e0d0e7d5cec00a7176c1d2d29d1d15a5292d555e3`。
- H002 aggregate qualification report：`491198f664006a86927567634e3345bfbad558b10d018e26c78679eeebcf3e1e`。
- P021 实现 `experiments/relational_transfer.py`：`360c5e339b1021dca9e8a2a195bf16f85ef31a55ca98f4c57eab4b9c7c9ea004`。
- kernel：`08ac925b1eb55e08de9cfd29b1838c8ef0c431717866fb222da683b4840dd705`；tests：`b984123666fea325fdd82ef8026ee84e9ac6ec859f0504b7dbad68d7310fe288`。

测试命令：`.venv/Scripts/python.exe -m pytest tests/test_source_reconstruction.py -q`；实际输出 `21 passed in 0.12s`。实际 inference 可行性检查进程没有导入 torch、transformers 或 bitsandbytes。上述是本轮文件字节身份；后续若改代码或换行规范，必须保留此记录并由 root 重新绑定实际冻结版本，不能冒称旧 hash。
