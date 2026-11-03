# M5 范围过度泛化：文献与拒绝条件（提案前，只读调研）

2026-11-03。仅检索一手论文/作者资源并阅读本地P021/P026，不准入算法、执行模型或修改研究状态。下面是文献判断与待审实验建议，不是新方法或已经验证的机制。

**结论：广义“关系反事实分数”已有充分先例。** 来源条件似然差、事实扰动、原子claim核验和保义一致性都不能单独作为新颖性。可能值得区分的是：在完全匹配值频率及语义有效性后，内部计算是否对**所声明范围的成员资格**敏感，并能比同一语义前端上的直接规则/文本核验更可靠地定位、更低成本地检测。当前没有该优势证据。

## 六个最相关的一手来源

| 来源与核验范围 | 已有重叠 | 本问题仍需额外检验的部分 |
|---|---|---|
| [CoCo：Factual Consistency Evaluation via Counterfactual Estimation](https://arxiv.org/html/2108.13134v2)，2021，全文§3.3/Algorithm1、§4.4 | 选择回答关键token，对其相关来源作mask，再比较同一原回答前缀下的条件概率；还明确采用独立于原生成者的评分模型。来源反事实+原token似然差已存在。 | 删除/遮蔽使模型依赖来源，不自动证明值属于正确范围。作者报告mask范围过小可从上下文恢复、过大可近零；这不是合法关系翻转或内部中介检验。 |
| [FIB：Evaluating the Factual Consistency of LLMs Through News Summarization](https://aclanthology.org/2023.findings-acl.322/)，Findings ACL2023，官方摘要/论文入口 | 比较同一来源下正确与错误摘要的模型分数；文章明确指出错误摘要若在来源逐字出现，LLM仍可能偏好它。条件似然偏好与事实一致性并非同义。 | 不能把“偏好正确条件续写”包装为新检测器；需要同词袋、同频率、源范围改变和保义改写控制，区别成员检查与字面复制。该文不是内部计算机制证明。 |
| [MetaRAG](https://arxiv.org/html/2509.09360v1)，2025 preprint，全文§3 | 将既有R分解为factoids，对其同义/反义变体依据retrieved context核验，并聚合成定位/回答分数；原子化+语义变换+上下文核验已有直接重叠。 | 其主要改变claim，而当前拟保持原R、改变已独立许可的来源。这个接口差异不足以证明创新；“反义必被矛盾”对缺失信息并不成立，unknown必须保留。本项是从方法定义推出的适用性限制。 |
| [RAGChecker](https://proceedings.neurips.cc/paper_files/paper/2024/hash/27245589131d17368cccdfa990cbf16e-Abstract.html)，NeurIPS2024 Datasets/Benchmarks；[论文正文](https://arxiv.org/html/2408.08067v1)§3.2 | text-to-claim extraction与claim-entailment checking已有明确分工；原子事实粒度不是新贡献。其部分整体指标依赖ground-truth answer，不能整套照搬为本项目无监督detector。 | 对本项目可用的source-only直接核验baseline应只给S/Q/原R，不给参考答案。提取器若把“Mon–Sat B”原样当一个无范围结构的原子claim，范围遗漏仍未解决；需保留量词、范围和例外，不能先丢条件再声称内部量有优势。 |
| [HELP：Monotonicity Reasoning](https://aclanthology.org/S19-1027/)，*SEM2019，官方摘要/论文入口 | 量词/短语替换的逻辑单调性及模型对此类推断的困难已有系统基准。对全称命题缩小成立范围与扩大成立范围的非对称性不是新数学发现。 | HELP是推理数据/训练研究，并非原回答无监督幻觉检测或内部介导检验。不能借一个合法集合式就宣称模型内存在相同算法。 |
| [DICD：Faithful detection of hallucinations in event extraction via causal intervention](https://www.sciencedirect.com/science/article/pii/S0950705126015455)，Knowledge-Based Systems2026，DOI10.1016/j.knosys.2026.116819 | 出版商摘要已主张事件tuple原子分解、counterfactual context manipulation、对trigger falsification敏感而对entity perturbation保持特异性；“结构干预检测关系幻觉”的宽泛叙述直接重叠。 | 此次仅能核验出版商索引的摘要；正文抓取403，不能断言其精确分数、同词袋对照、内部变量或因果识别已覆盖本方案。正式新颖性判断前仍须读全文。不得把未读到的细节报告为不存在。 |

以上只列六项近邻，不是穷尽检索或优先权证明。CoCo/FIB的方法定义足以否定“条件似然对照本身新颖”；MetaRAG/RAGChecker否定“原子化+核验本身新颖”；DICD要求更窄地核对结构干预的近邻重叠。论文效果不能移植成本项目结果。

## 必须先确定的语义合同

在完整、单值、无覆盖例外的日程中，将原范围主张写作

$$C(S,R)=\mathbf 1[\forall d\in[\mathrm{Mon},\mathrm{Sat}],\;v_S(d)=B].$$

这只是研究者对许可语法的判定，不是学习到的白盒算法或新检测分数。缺失日程、开放世界、多个并存值、跨周、边界非包含、优先级/例外不明时均不能用这个式子擅自补齐事实。对来源仅支持Tue–Sat B，扩成Mon–Sat B增加了未许可的成员；若Mon A且A/B互斥，则有显式反例，而不只是缺失支持。

将原七天表集合简写为`Mon A / Tue–Sat B / Sun C`，仅将A/B两组互换后，固定`Mon–Sat B`仍为false，不能冒充support flip。逐日B仅五天、且A/C均不同于B的完整七天单值表，也不可能在保留该日频率时让六天全B成立。这里的三组写法不是原始来源行数。词袋相同不保证语义有效；这正是P021必须排除的混杂。

父任务目前另行讨论的匹配对改为：

| 提案世界 | Mon | Tue–Sat | Sun | 固定Mon–Sat B子主张 |
|---|---|---|---|---|
| S− | A | B | B | false，Mon是反例 |
| S+ | B | B | A | true |

两者都含六个B日、一个A日。**上表Tue–Sat只是本笔记的集合简写；native已经是七个显式day keys，提案实际表示也应逐日明确列出，并非native中存在一个压缩区间句。** 两个逐日表均为B6/A1，仍需实际tokenizer核对token/长度/位置，字符长度相同不等于token对齐。它们改变了native来源且都与原R后续的Sun C冲突，所以只能讨论原回答中、Sunday之前的固定Mon–Sat B子主张，绝不能宣称整个原R真假翻转。原native世界须另列为不匹配锚点，而非额外同合同对照。

## 与本地已关闭实验的界线

- **P021（relational null worlds）**已对FinQA/TATQA运行来源行列/数字重排，结果低于同run shared对照；其失败记录明确指出保持数字词袋并不保证反事实语义有效。将`Δlog p(original R | permuted S)`重新命名为relation score，就是重复。只有独立许可的成员关系及明确反例/保义控制，才增加新的判别条件；并未由此保证方法有效。
- **P026（repair / prefix reset）**已运行source×response repair、未来token交互与local reset；错误组只3/5过、正常guard3/6失败。不得修成Tue–Sat、替换日名/范围、删掉先前R、选择最利于对比的后缀，再称发现新范围机制。保持原R与原prefix的来源对照不等于该旧repair实验，但若后续主要收益仍来自repair/reset，应按旧支线处理。
- **H017 M3/M4**限制仍在：数值位置K-role选择性最终输出门失败；自然300/years/ago观察并非高概率错误单位。当前范围例须独立检查是否真有“强支持值、错误范围”的新现象，不从前例或文献替代实测。

父任务随后报告了一个新的**既有缓存**动机：old BF16 Data2txt5670/source894原轴中，minute `00`（target55）的source p=0.997455、full gap=−0.205618；endhour `21`（target57）的source p=0.999670、full gap=−1.022722。这表明高概率原续写可以实际出现，不能把M4 ago的低概率结果推广为不存在这种问题。本次文献调研未重新读取或资格核验该缓存，数值只按父任务的匹配审计结果引用；单词续写概率不等于全称范围主张置信度，更不证明interval-frequency copying或某个内部路径。该新动机也不自动重开P021/P026或准入反事实方法；手工许可域的直接规则核验仍然很简单。

本地依据：`research-state/prior_experiments.json` 的 `P021_RELATIONAL_NULL_SOURCE_WORLDS` / `P026_RETROSPECTIVE_REPAIR_AND_PREFIX_RESET`；本轮只读这些已封存记录，没有目标真值访问。

## 一个以拒绝为先的最小对照建议（尚未准入）

最小提案先只审上述S−/S+两份**七天显式键表**，保持原Q和原R所有字节；不能把native再“逐日展开”当作新控制，因为那是no-op。若需要一个保义必要对照，可另行审查每个世界相同七条事实的固定行序反转，得到2个语义世界×2个行序共4份来源；全部保持B6/A1、逐日成员与完整性不变。反序改变绝对位置，因此只作为预测/读出稳健性控制，不能把不同位置的坐标直接互patch。这里仍只是最小提案建议，不是新增四世界run准入。

若研究重点另设为“值字面提及频率”，那是**非最小的另一个编码实验**：必须先明确新建压缩视图，如把Tuesday、Wednesday、Thursday、Friday、Saturday五个日名显式列在同一个B clause，Mon与Sun各自成行。这样两个语义世界的压缩视图均B2/A1，而两个逐日视图均B6/A1；七个具体成员保持相同。该视图不是native已有格式，尚未进行语义前端审查，不应隐含加入本次提案或把其构造当作已完成事实。

固定终点仅限原Mon–Sat B子主张及其原前缀。研究者的集合membership规则是透明baseline；固定source-only文本entailment核验器使用完全相同S/Q/R子主张与上下文。条件似然对照只算已有方法家族的必要控制，不是新风险模块。所有视图、原token投影、unknown和数值界应在任何调用之前另行审核；本建议没有提供运行许可。

优先拒绝以下结果：

1. 若似然/所谓关系量只在一个行序有正确方向，保义反序后翻转或消失，则位置/呈现解释尚未排除，不包装成稳定成员推理。B的全局日频率与字面计数在S−/S+本就匹配；不能把不存在的native压缩区间复制当作已证成因。未来若另审压缩视图，也必须将其字面频率控制与本次行序控制分开。
2. 若已给许可表的直接规则完整解决该域，而内部候选没有额外自然覆盖、定位或成本证据，则拒绝白盒必要性；受控正确率不代替原回答检测。
3. 若错误子主张本身已是低概率，或全R后续Sun C污染整体分数，则停止“高置信范围错绑”的动机和whole-response support-flip解释，不加prompt/模型/repair救援。
4. 若通过这四个世界，也只说明受控边界成员敏感性可行。一个Monday边界例可以由字面边界匹配解决，尚不能证明一般全称聚合、否定/多值/内部例外处理或独特电路。

## 内部计算究竟还能增加什么

可区分的研究假设应更窄：**原范围完成前，完整事实闭合状态是否保留“范围内反例成员”，而不同范围外成员/压缩频率的状态不能替代该作用？** 这要求将值命中、完整范围可用性、当前query已整合的条件及最终原token读出分开；单看B位置attention、Q/K相似度、source mass或一组logit差都不能识别它。

如果未来另行申请因果实验，至少需要“范围内反例closure”与匹配的“范围外反例/保义格式”控制，并以原R终点的最终效应而非局部attention作判断；新读出不能靠目标标签挑头/层，不能假设K=纯地址或V=纯值。可携带语义的closure状态被访问也不证明具体成员被编码或使用。若最终作用不能区别scope membership与词频/位置/既有直接文本核验，这条内部主张应被拒绝。

因此可能的贡献不是给`counterfactual relation score`命名，而是一个被反例约束、语义覆盖明确、能在自然原回答上证明额外定位/检测/成本价值的证据接口。当前上述每项都仍未建立。
