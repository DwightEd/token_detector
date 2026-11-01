# 原始论文对照与新颖性审查

检索日期：2026-11-02。优先读本地 PDF，再核对出版方/arXiv 方法正文。ReDeEP 的两个 PDF 按 arXiv:2410.11414 去重。三份只读文献提取分别覆盖本地论文、近邻检测方法、统计基础；最终结论由主执行者合并。检索不是系统综述，不能据此声称文献穷尽或已有方法不存在。

arXiv helper 已实际运行：`arxiv_fetch.py search 'all:hallucination AND all:token AND all:context' --max 5`。原始 JSON 在本次本地 `outputs/source-reading-20261102/arxiv-search.json`。Zotero/Obsidian 连接器未发现，未访问。API 元数据和正文版本可能不同，下面显式区分读到的版本。没有为扩大目录而批量下载论文。

## 直接相关的检测论文

| 规范标识 / 论文 | 已读范围与实现机制 | 对本项目的约束 |
|---|---|---|
| [arXiv:2606.31033 CORTEX](https://arxiv.org/html/2606.31033v1)；Furumai/Haruta/Matsumoto/Kamisaka，2026预印本 | §2–4.3及附录：有/无来源最终隐藏态差；减去参考条件注意力传播的历史差；拼接后监督 MLP；二状态标签持续性平滑。 | 来源删除对比、上下文传播、链平滑均不是空白方向。须单列监督训练条件，不能把其效果归给无监督推断。保序命题不适用于其顺序平滑。 |
| [arXiv:2509.21875 LUMINA](https://arxiv.org/html/2509.21875v3)；Yeh/Li/Mallick，[ICLR2026正式记录](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f5846131aa6a72d1df3bd6d43a4a960b-Abstract-Conference.html) | 方法与附录C/D及官方代码：检索文档与随机文档替换的下一token分布核MMD²；logit-lens内部知识信号；两者组合。 | 随机文档替换不是来源缺失。其条件动机不是事实标签识别定理。必须使用真实实现、相同原token轴和成本记录。论文/代码在top-k未重归一及层数细节有差异，应固定版本。 |
| [arXiv:2410.11414 ReDeEP](https://arxiv.org/abs/2410.11414)；Sun等，[ICLR2025论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/7daf60e805e596c3bd1e843e72ea5560-Paper-Conference.pdf) | 本地正式版1–10、17–18、20–23页及arXiv前8页：attention选来源token的ECS，FFN前后logit-lens JS的PKS，选层/头加权组合；AARF是另一个生成干预。 | 附录J使用验证标签做头/层选择与超参网格，不因generator冻结就变成完全无监督。其机制解释不提供逐token事实保证。 |
| [SiGHT](https://proceedings.mlr.press/v300/chen26d.html)；Zi-Ying Chen/Meng-Fen Chiang/Wen-Chih Peng，AISTATS2026，PMLR300 | 本地PDF1–8、14–16页：语料段落正例与LLM改写错误负例；Word2Vec/POS图，3层GAT、全局均值池化和二元训练。 | 名称含self-supervised仍有合成二元监督。最终是段落/图分类，token图节点不等于验证过token定位；不直接当无监督局部基线。 |
| [arXiv:2512.07515 TPA](https://arxiv.org/html/2512.07515v2)；Lu/Lu/Liu/Zhang，[ACL2026正式记录](https://aclanthology.org/2026.acl-long.1159/) | 读v2方法§3和训练§4.1：通过残差探针差分作概率望远镜分解，头logit softmax分摊，七来源按18类POS聚合126维，监督XGBoost集成。 | 概率守恒不识别事实；最终是response-level，不能误称token定位对手。arXiv helper返回更晚更新，当前提取绑定v2与正式记录，未声称审计了所有版本差异。 |
| [USP at SHROOM-Visions](https://aclanthology.org/2026.uncertainlp-main.47/)；Rafael Macário Fernandes，Uncertainty-Aware NLP workshop，2026-10-29 | 原始9页PDF与方法/结果：图像代理、监督tagger与oracle粒度对照；区分句级信息和精确字符跨度定位。 | “有用区域代理未必提升精确定位”已有直接概念重合；跨到RAG不能自动构成新颖性。本文是多模态、字符级相关指标，与本项目固定片段成对AUC身份有不同估计对象。 |
| [arXiv:2608.18115 Temporal Multi-Signal Fusion](https://arxiv.org/html/2608.18115)；Igor Itkin，预印本 | 方法/训练/结果：33维文本、NLI、LM信号输入BiGRU等监督序列模型，比较顺序打乱与容量；主报pooled token AUC。 | 加序列结构、外部NLI、顺序控制已被做过。可用局部分解审查其结论，但未下载运行作者模型；本文不重复其报告性能作本地结果。 |
| [arXiv:2610.27370 SHAM-Bench](https://arxiv.org/html/2610.27370)；预印本 | §3–5及限制：六来源数据集九个测试集合、六类任务的参考型span评测，讨论跨域和边界不一致。 | 多任务span benchmark本身已存在。合并旧集合不等于本项目新的独立确认，尤其RAGTruth已用于方法开发。 |

## 理论工具：可继承什么、还缺什么

| 规范标识 / 原始来源 | 读到的结论 | 不能借名跳过的假设 |
|---|---|---|
| [Neyman–Pearson 1933](https://errorstatistics.com/wp-content/uploads/2014/08/neyman-pearson-1933_on-the-problem-of-the-most-efficient-tests-of-statistical-hypotheses.pdf)，§II/III(a)，printed298–301 | 指定简单H0/H1时，按两类密度比阈值作最强检验。原出版方不可用，使用原文扫描。 | q_absent/q_present 不是事实类条件密度比，除非另外建立桥梁；posterior也还需类别先验。 |
| [Huber Robust Estimation](https://ir.cwi.nl/pub/13042/13042D.pdf)，§6 printed17 | 指定TV污染邻域下最不利分布的LR呈截断结构。读取原始机构存档索引正文；直接PDF后来超时，未复核1965原证明。 | 不能给LLM分数任意clip就声明minimax鲁棒；要指定污染集合、依赖结构、阈值与错误类型。 |
| [Candès/Fan/Janson/Lv 2018 supplement](https://faculty.marshall.usc.edu/jinchi-lv/publications/JRSSB-CFJL18_Supplementary.pdf)，§F.1 | 条件独立原假设下，正确条件替换使检验统计交换，(1+超越次数)/(K+1)有效。 | 任意删除、随机文档、对抗替换并非正确条件样本；固定的回答前缀本身可能受来源影响，不能据此宣称CRT有效。 |
| [Conformal Risk Control，arXiv:2208.02814v4](https://arxiv.org/html/2208.02814v4)，§1.1/2.1及证明 | 交换例子上，对单调、右连续、有界损失曲线，用含一例修正的经验阈值控制新例边际期望风险。 | 需要独立已标注校准；token不独立时以完整来源/回答为单位。漏检比例可单调，FDR不自动单调；不是逐例、高概率或条件风险保证。 |
| [Pearl 2009](https://escholarship.org/content/qt36w8n7pg/qt36w8n7pg.pdf)，§3.2.1–3.2.2 | do干预替换结构方程；识别要求所有与观察规律相容的模型给出相同目标。 | 提示删除测到的是行为敏感性。固定原回答历史不是重新生成历史的总效应；支持性语义关系需额外约束。 |

## 本轮得出的研究判断

这些文献支持把当前项目定位为“有严格推断内核、缺少目标识别与充分新颖性证据的研究实现”。新增的保序命题和AUC身份可帮助排除错误解释，不能凭它们宣布检测器已达到顶会水准。优先补独立自然人类支持性标注、匹配后端/预算，以及最近方法的局部再评价；没有新证据时反复改标题、公式名称或审稿分数没有科研进展。
