# H013：单个肯定谓词conjunct的固定题面桥

前瞻登记；观察尚未执行。主对照只使用封存H011 **burden0的全部16问题/32候选**，不把去掉suffix当成新的机制改善。H008完整对照保留为背景。全48原答/96原槽映射仍保留，重复burden属于编译复用，不是48个独立新问题。

H011最短复合句仍含两谓词/两角色，7/8来源的两问题都选同名。竞争解释：模型不能可靠读这种cloze；或者完整的另一conjunct/角色与上下文负担妨碍了本来可恢复的单角色。已有H005/H007只切到句子，未单独保留一个肯定谓词；P026观察原R未来续写，输出合同不同。详见独立prior-review.md；不声称新颖性。

## 唯一固定结构变化

source、source-only排序两候选、system、H011 QUESTION_TEMPLATE（包括[OTHER]不同人的解释句）、同3B/NF4/FP16/SDPA、完整姓名/EOS、float64全词表normalizer与风险均不变。只改变masked-statement字段：

1. 从已严格验证的原R首句取目标对应的一个完整肯定conjunct，记录原rawslice。第一个conjunct无独立句末，补固定period＋space；第二个保留原period＋space。
2. 原person literal擦为[TARGET]。只留下目标event条件；该atomic statement里没有[OTHER]。题目前缀关于[OTHER]的说明仍保留，明确是dangling instruction条件，不能悄悄删词救分。
3. 不从S中选择真姓名/绑定写进Q，不改source，不按标签、模型结果或slot效果筛选。原48×2slot×2name=192实例完整保留；Rswap及三burden坍缩到16Q/32完整completion。

这是删掉完整非目标conjunct、另PERSON/关系/不同人耦合、缩短prefix和确定标点规范化的**组合变化**。不能唯一归因为谓词数、attention、sourcefirst bias或内部生成因果。一般自然否定、析取、量词、引用、指代与修辞不能直接用肯定conjunction elimination；严格toy规则不是自然coverage。

## 条件数学性质，不是效果保证

仅对本明确双射域，令c_i*为两事件的不同真姓名，q_i是在完整来源两候选上的条件质量。若每个来源的两个问题均满足q_i(c_i*)>1/2，且错误回答严格交换两姓名，则固定max-slot risk满足：

\[
r_{\rm supported}=\max_i(1-q_i(c_i^*))<\tfrac12,\qquad
r_{\rm swapped}=\max_i(1-q_i(c_{1-i}^*))>\tfrac12 .
\]

因而若所有16问题均正确，构造回答会完全分离。假设本身需要实测；质量不是事实真值概率，不能将此条件引理推广为自然检测保证。H011的同名角色坍缩已使“两个角色都正确”前提失败，不能通过换聚合或权重来伪造前提。

## 固定运行、判别和停走

- 独立审核纯compiler、shared collector/CPU reader，先用synthetic及outside-checkout installed package验证。冻结全部32metadata/消息/候选/源/协议后才模型。
- 对H008与H011现有消息＋候选＋完整framing＋数值身份做exactcache审计；等价时必须复用，不因protocol/source哈希变化重算。至多32新forward，无采样/模板/模型/阈值搜索。
- 观察和独立CPU冻结后才释放构造参考。完整16Q的greedy/wrong/tie、sourceflip8/eventflip8、rawLL/poolmass/长度/成本、全部48答/96槽保留；主比较H011b0，其余旧结果不覆盖。
- 若单洞仍失败，否定该cloze接口必要能力，不换词/7B补分。若恢复，只支持此组合条件下的toy接口改善，仍不证明自然前端、需要LLM、新颖性或唯一内部机制。
- H014是拟议必要缓存控制：原封使用H009已预声明双射energy读H011新矩阵，区分共同name偏好是否遮住正确交互；不新模型、不选更强bias、不与H013择赢家。其设计/claim须独立冻结后才算结果。
