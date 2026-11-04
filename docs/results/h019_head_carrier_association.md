# H019：head–carrier 关联擦除的有限结果

**计算试验与独立审计通过，但幅度不能稳定区分预定的 SEM/EQUIV；尚无新幻觉检测器、风险分数或 AUROC。** 这轮用完整内部缓存检验一个受限的计算问题，没有增加模型推理或读取真值。

固定来源为 RAGTruth QA source14375，两条重新采样的回答为00012/00013，均由 Llama-3.1-8B 产生；它们不是发布数据中的原回答，不能继承原回答标注。完整图来自独立 BF16 full forward，F32文件无损保存其BF16数值；没有混用原incremental采样或fixed-prefix replay的浮点状态。历史模型路径/文件元数据不等于原权重body SHA已经确认。

## 做了什么

M0保留全部343个原targets的入选、排除或unknown理由。词法规则检查chosen target是否含ASCII数字、其已消费同句prefix是否有ASCII字母，得到6+4=10个原决策；原t使用 `q=P+t−1`，不是post-token receiver。这个选域是事后描述规则，不声称模型在q知道未来token。

来源许可在看效果前人工冻结：完整局部关系已闭合的同ID `minutes` carrier为126/277/303/369，组成六对，其中重复simmer指令277–303是EQUIV，其余五对为SEM。全部候选与unknown仍保留；77主体不明、342同句后置条件未闭合而不获许可。这里的类别是局部来源关系许可，既不是自动前端，也不是原R对错标签，不强加全局双射。

每个原q、每层，取同一GQA组内全部六对Q heads：8组共48对，32层全算。对两head×两carrier四cell A，单独替换为：

$$
B=rc^\top/M,\quad r=A\mathbf1,\quad c=A^\top\mathbf1,\quad M=\mathbf1^\top A\mathbf1.
$$

该反事实保留两侧边际读取量，其它cell不变。48个重叠head-pair是分别定义的实验，不能相加成一个世界。两head共享V时，pre-O差为 `[-κ(Va−Vb), +κ(Va−Vb)]`，其中 `κ=(ad−bc)/M`；报告各128维head及2×128拼接范数，不能把两条128维差相加后冒充residual消息。W_O未独立确认，本轮没有4096维O投影、完整后缀或输出概率读出。

M0固定F=6、Q=10；M1完成 `32×48×6×10=92,160` 个分别定义的反事实。原数值门、双边守恒、direct AV与因子分解以及同ID layer0零控全部通过。独立审计不调用候选kernel，以原A/V坐标和80位Decimal逐tuple复核，92,160全部通过。**这些是算术、覆盖和身份资格，不是语义或检测资格。**

## 全部预定对照

下表mean仅由已封存sum/tuples相除，max直接来自预定描述；没有新增评分规则、阈值、显著性检验或挑选头层。

| case | source pair | 预定类别 | mean concat norm | max concat norm |
|---|---|---|---:|---:|
| 00012 | 126-277 | SEM | 0.000457177 | 0.063091322 |
| 00012 | 126-303 | SEM | 0.000414698 | 0.061708474 |
| 00012 | 126-369 | SEM | 0.000561648 | 0.102787710 |
| 00012 | 277-303 | EQUIV | 0.000237953 | 0.068853181 |
| 00012 | 277-369 | SEM | 0.000499219 | 0.083463296 |
| 00012 | 303-369 | SEM | 0.000377295 | 0.068261956 |
| 00013 | 126-277 | SEM | 0.000631699 | 0.170769969 |
| 00013 | 126-303 | SEM | 0.000530669 | 0.099142159 |
| 00013 | 126-369 | SEM | 0.000308832 | 0.019112351 |
| 00013 | 277-303 | EQUIV | 0.000384344 | 0.100276582 |
| 00013 | 277-369 | SEM | 0.000386989 | 0.051364183 |
| 00013 | 303-369 | SEM | 0.000329130 | 0.059321641 |

00012中EQUIV均值0.000237953低于全部五个SEM；00013中EQUIV均值0.000384344却高于126–369的0.000308832和303–369的0.000329130，EQUIV最大值0.100276582也高于四个SEM。不能只展示第一个回答的均值，或把总体nonzero称为绑定检测。这否定当前幅度proxy可稳定语义分离的晋升理由，不证明所有内部关系信息都不存在。

幅度本身等于 `sqrt(2)×|κ|×||Va−Vb||`，混合head-address关联与上下文化payload距离。同字面carrier、同q及边际守恒没有排除key位置/RoPE、完整前文、注意力大小及V范数；EQUIV只许可局部指令等价，不代表全部状态一致。全部头层是相关单元，92,160个tuples不是92,160个独立自然样本。

## 跨图精度限制

另一次独立claim只比较已封存的64份source ΔV小表，不重跑模型或M1，不读母A/V或全部M1 chunks。全部32层×6pairs×8KV组=1536个向量、196,608个坐标均保留；执行/覆盖通过不表示数值相等。

| 已封存精度摘要 | 结果 |
|---|---:|
| bitwise及普通数值均相同的向量 | 48 / 1536 |
| 不同向量 | 1488 / 1536 |
| bit-mismatch坐标 | 180,043 / 196,608 |
| 最大绝对差 | 0.078125 |
| 最大相对L2差 | 0.37815079167550525 |

因此，跨R的消息幅度差**不能只归因query/head路由**，ΔV因子本身并不相同。该诊断没有隔离full-N数值kernel、捕获或历史身份原因；源前缀在精确因果计算中不依赖未来R，不能把这些差异解释成未来回答反向改变来源语义。它也没有比较individual V或共同平移。相对量0/0保留null/flag，无epsilon或事后容差。

## 成本、范围与来源

| 阶段 | 已完成成本 |
|---|---|
| M1 | 工具内8.714s；launcher13.39s；数组31,427,200B；private峰值264,634,368B |
| 独立审计 | 165.607s；80位Decimal；0数值违规 |
| ΔV精度诊断 | 工具内0.284s；launcher0.817s；64表、1536向量 |
| 三阶段新增模型/head/MLP/tokenizer/真值/风险 | 全部0 |

原始图与run输出统一保存在C盘可删除数据目录 `ResearchData/token-detector-remote-20261103/06_same_question_multisample/`，公开文件只保存小摘要。M0/M1/audit/precision的run前缀分别为 `02e68af9` / `709af13` / `9678946` / `d1de688b`；完整seal、metadata SHA、源码SHA以及全部预定pair描述与小表身份见[机器摘要](h019_head_carrier_association_summary.json)。审计JSON SHA为 `57549447845fa880f52dbfc0ee78815df70be2884ac1fc51a5921377f7d9fa23`；精度JSON SHA为 `83da20bc4b8b2397e08d4f53bb5ca861fed111fce09c1e10eafe1247c76dcec8`。大原文件复用已完成资格的body-hash witness和当前stat，不冒称重新哈过全部6.5GB。

旧P006/P007、ownership X/E/XE、endpoint swap、N/A/G及真实后缀干预都已存在。此次有限区别是明确保持row和column两侧marginals的独立化；generic E能表达B不等于旧配置已实际测过它，但这也不是全球首次或新理论。2×2恒等式、互信息和因子分解是标准性质。实现见[前端](../../experiments/head_carrier_frontend.py)、[纯kernel](../../experiments/research/head_carrier_association.py)、[编排](../../experiments/head_carrier_observation.py)，实施范围见[冻结最小合同](../../refine-logs/head-carrier-association-20261105/MINIMAL_CONTRACT_20261105_013351.md)。

本轮保留有限计算结果，不晋升幅度检测器，不选择新q/head/layer/sign救回。下一必要问题是来源优先的语义记录和原R过去prefix能否给出完整条件下的支持/冲突/unknown，允许reuse、多对多和合法推导；随后内部量必须证明相对同前端直接文本核验的定位、可靠性或成本增量。source-first是研究方向，尚不是已验证方法，六任务各0.8目标未完成。

**代码与CI状态（发布后更新）**：冻结源码`ee69ad0`作为`5fb457e`于Nov5 02:47:16 CST推送。154个文件exact installed、60项定向测试通过20.89s；本地完整CPU300s超时仍保留为incomplete。该次提交的完整Linux CI41890599361/job126786651227及安装包checkout外测试step6均已success，[精确收据](h019_ci.json)。工程验证不构成检测效果证据。
