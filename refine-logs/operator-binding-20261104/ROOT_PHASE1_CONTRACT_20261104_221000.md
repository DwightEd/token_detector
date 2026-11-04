# H018 第一门：原回答的来源→Norm+MLP局部使用读出

2026-11-04，Asia/Shanghai。代码基线69835f2。上游唯一提案为`PROPOSAL_20261104_215536.md`（d5cae090a4550e17704fe184fc2247e23099a3298a915922979b43b9587800ff），独立科学收据5ba047946449a77e924dc55eccd43d7bdb0f494c07fd1cf4816a70bcad7dc874。本文件只准入候选实现和有限工程/预定反例第一门，未准入第二门、完整900或“负作用即幻觉”的结论。须独立复核本合同、最终实现、实际资源并另claim后才有模型执行。

## 科学对象与原轴

仅5682、5684，来源13603，完整原S/P/R字节保留。来源已有字段许可、弱OR unknown、原R时序以及缺失物理缓存直接引用`FRONTEND_OBSERVATION_CONTRACT.md/.json`；零模型主线tokenizer门另取得实际P/N/T，不能把旧native的R轴冒称已保存prompt轴。所有有效原token都评分，原索引不删、不用目标真值选位置。prechoice q=P+t−1才输出R[t]的风险；post-property只描述。valet-not73的属性已在过去，可作支持型否定反例；Wifi67和lot66/102是未消费property时的真实词选择，不称已完成的关系判断。

主scope固定完整source_mask。每层使用原native softmax、V、GQA和真实W_O分区，得到source/history/other写入；source不重新归一化。复用`experiments.source_write_observe.partition_writes`，bias如存在不得分配多次，当前资格仅接受已核无bias的原Llama。自R注意力可另记，但history组含self，不能误称全部earlier R。

完整forward显式使用原生eager attention backend，离开context后恢复；不允许SDPA返回None时静默fallback。本次使用真实native注意力和v_proj输出（完整N、8KV头），与真实pre-O的32头AV核对，再核对三组O写入；无需再把QK/RoPE重建称作新科学量。eager FP32不是历史SDPA/BF16逐bit复现。额外分区3次W_O/层/例（合计192次）及64次AV重建分别记账，不计作免费或新的完整forward。

每层post-attention/pre-MLP residual为x，F=原post-attention RMSNorm接原MLP。采用相同T行reference和counter：delta=F_T(x_R)−F_T(x_R−m_S)。D由32层delta按层序以FP64累加，再一次转FP32；H−D用原FP32残差算术。H必须来自native finalnorm输入，不能取postnorm last_hidden_state。risk固定sigmoid(e)，e=logp(g(H−D),original target)−logp(g(H),original target)，g为原finalnorm/full-vocab head。输入source/原历史/attention skip以及其他层输入不被改写。这个held-native-input surrogate不是传播后的新世界，也不区分Norm与MLP独有的作用；attention可能组合、MLP可能复制。

## 第一门控制与冻结成本（覆盖提案旧6-head预算）

每例每层除native的一次MLP外，严格4次额外官方T行F：reference、source-removal、固定norm-sham、独立走counter路径的m=0。sham复用已审`matched_noise(m_source, seed=17+layer)`：CPU FP32 generator后移至device，逐receiver匹配整个O写入向量L2范数，不从结果选种子、方向、层或scope。source向量为0时sham亦0；目标非0而噪声范数为0必须失败，不填epsilon。

新增P006同forward direct source-write control是必要的新量隔离，不能将已知H−write endpoint或signed source support当创新：累计全部32层m_S按原P006的FP32加法顺序，构造H−W_S。只增加2次head，不增加模型或MLP；不重跑P006的196回答结论。

为消除finalnorm形状差，保留native **完整N行H_raw**，四个finalnorm重放都保持相同N和native contiguous布局；只在原prechoice R行放入D、D_sham或W_S，其他行保持原H。m=0的完整N norm必须逐bit复现同forward native finalnorm输出；若失败停止，不改成T-row/放宽政策。m0的D严格为0且norm输入输出已逐bit相同，因此head复用native结果，不假称另作过head/noop调用。

每例完整head最多4次，各自为T原target行×全128256词表：native实际postnorm、source-removal、norm-sham、P006 direct-write。不得分块计算head；允许在head算完后按最多16行流式搬运/落盘其完整logits，这只改变存储，不改变算术。native与counter同T形状。各分支保存实际token logit、全词表LSE、logp、H/D范数及norm分母，拆开e的分子和分母，不将分母变化包装成独有抑制feature。同forward native NLL留作既有置信度基线，不拟合去相关或改变符号。

**总上界：2次native完整forward；8次完整head；256次额外局部MLP和256次对应post-attention norm；8次额外完整N finalnorm；另有native32×2层MLP及2次native finalnorm包含于完整forward。** 分别记录调用、行数、耗时、transfer字节。不得自动追加源世界、层头、canary、QA或第二门。每例只执行一次，部分成功原始文件必须保留，失败后不自动重跑。

## 数值门：先冻结、后观察

全部native/input/replay/AV/logits为FP32算术；原参数BF16、官方RoPE FP32，TF32/AMP/grad关闭。完整原权重、Torch2.8.0+cu128、Transformers4.57.1与SM120合同沿用H017资格，tokenizer元数据固定；这不是历史BF16概率逐bit重现，也不是原GPT4/Mistral/Llama2生成机制。

以下为固定操作性资格界，不是语义阈值/浮点定理，任何失败只封存/停止，不看新输出调宽：

| 项 | 冻结门 |
|---|---|
| dtype/shape/finiteness、Q/R/token/offset/source mask、future attention | 精确；future质量0 |
| native x_mid=layer_input+native_attention、layer_output=x_mid+native_MLP | 对采集的真实FP32行逐bit |
| source/history/other分区O之和对native O；同forward native headAV重放 | abs≤0.002+0.0001·abs(native)，所有坐标通过，继承旧AV操作门但不冒称覆盖MLP |
| 本地T-row RMSNorm输出对native post-attention RMSNorm输出（即MLP输入）、T-row reference MLP对真实native write | 前者abs≤1e−6+1e−5·abs(native)；后者abs≤2e−5+2e−4·abs(native)，所有坐标通过；不得将norm输出与其输入x_mid比较 |
| 独立m0局部counter对同T reference | 逐bit，delta_m0严格0 |
| 完整N finalnorm m0对native postnorm/last_hidden_state | 逐bit（形状/布局相同）；若失败停止 |
| D逐层累积 | FP64顺序累积与独立CPU同序重放逐bit，保存FP64 D与最终FP32 D；不得改用某层子集 |
| norm-sham幅度 | abs≤1e−6+1e−5·目标范数，0范数单列 |
| e分解和sigmoid | 固定GPU FP32 logp=z_target−torch.logsumexp(full_vocab_logits,-1)，e=logp_cf−logp_native，risk=torch.sigmoid(e)；单列decomposed_e=(z_cf−z_native)−(LSE_cf−LSE_native)，允许不同减法次序的舍入；固定误差门为32·float32_eps·(abs(z_cf)+abs(z_native)+abs(LSE_cf)+abs(LSE_native)+1)，全部finite。独立CPU FP64原logits重放只是audit oracle，不生成方法分数 |

数值误差、参考/native偏差及D尺度全部公开；小到资格误差量级的符号只作numerically unresolved的描述，不删token、不改主risk或挑方向。模型或数值资格失败不能用0.5伪装完成。两个合法否定/True共事实若出现明显反例，只能报告该signed proxy未具备晋升依据，不能换号/阈值/seed、选层头、改字段scope、重新做NLI或source-format来救援。

## 保存、资源与实施边界

复用主线encode_source/encode_response/answer_axis/token_scores、既有partition_writes和FP32 streamed observer；候选kernel在experiments/research、orchestration在experiments并经统一CLI接入。不复活旧myproject模块别名。新增hook必须保持native forward输入/输出不变，递归replay有显式guard与finally清理；全部层/头在结果前固定。KV/source-prefix及source/other/history/self身份保留，source-prefix只能作精度允许的causal一致性证据，不能反推原生成器。

实际N/T由已授权零模型门填入资源计划。沿用6GiB GPU allocator预算、512MiB host/GPU floor和逐tensor mmap/单层物化；不压精度、不截R/来源、不并发。CPU raw搬运最多16行full-vocab或32MiB缓冲，按层顺序落盘并释放临时CPU数组；D/source累积在GPU。保留完整raw logits和必要native/replay输入输出，尽量少量按层文件，避免重复扫描4万小文件。模型前metadata-only资源复核与代码fresh review仍必要。

本次输出包含两原R的全部风险和unknown记录，但**第一门不开GT、不算自然AUROC/0.8，不把两个合法关系或weakOR当已证明正例集**。第二门source×operator/同义/同域owner对照的文本及预算尚未冻结，仍NOT_ADMITTED。只有后续符号/特异性门独立通过，才考虑完整900的分数冻结和最终标签评价。当前已有M5、NLI/P006负结果与数据库存直接复用，不重验其已知现象。
