# 实验账本与实际结论

## 2026-11-05 当前检查点：H019完成，单独幅度不晋升

H019所有M0/M1/独立80位审计/ΔV精度阶段均已一次完成，0新模型/0标签/0风险。10原R数字q、6手工局部sourcepair、32层和48独立headpair共92160 tuple；算术/覆盖通过不代表语义或检测。EQUIV幅度在第二R大于部分SEM，关闭单独幅度/MI/chi风险；1488/1536 sourceΔV向量跨graph不同，拒绝routing-only归因。详见[结果](../docs/results/h019_head_carrier_association.md)、F063/F064、RESUME。I02新source-first语义前端尚在合同设计，无新AUROC。H018和旧reanchor分支不重跑。

代码ee69ad0 exact installed154；60targeted通过20.89s，完整CPU300s incomplete，须新push后完整exact Linux CI成功。D迁移清理约16.4GiB完成，raw/模型/失败证据保留；新增大raw落C。以下状态段落为当时历史，不代表当前活动任务。


当前补充：三批70文件10,080,059,000B已完整恢复，原16+replay16资格完成，2完整graph数值资格进行中。H018 target-only audit7626e002 PASS后预定not73主e=-.0024986267（预算.00042623517）反证必要正方向；固定分支关闭，不再补control。[具体数值](../docs/results/h018_owner_specificity.md)。F059/F060与RESUME是当前状态，下段下载中/待审计属于较早检查点。

## 2026-11-05：补齐遗漏的同题多采样与内部缓存

此前库存遗漏 `research/reanchor`。真实4source×4seed同Llama原生成16NPZ已下载C盘，共1,168,475,270B（含3份metadata）；全层头attention/raw logits与逐prefix replay hidden属于不同合同。16份hidden及00012/00013两份attributed graph正在分批恢复，不能以旧库存缺失结论代替实物检查。位置、来源、观察轴和限制见[缓存地图](../docs/REMOTE_INTERNAL_CACHE_MAP_20261105.md)，当前顺序见[计划](../refine-logs/EXPERIMENT_PLAN.md)。无新模型或目标标签，不把新采样R套用官方旧R标签。

H018 owner前端34a449f0通过。第一次6fe770d9在主存门失败（1partial/0head/4extraMLP）；明确恢复7c6d857e完成target并child-sealed，control因D磁盘满中止（恢复总2native/4head/204extraMLP）。完整pair、独立raw审计和新AUROC均未完成；已完成target/S0必须复用，禁止再启动旧双world入口。资源失败不等于方法反证。D盘已通过转存原数据压缩包及OEM安装源释放约16.4GiB，科学数据与失败收据保留。

151 exact installed新source、65targeted通过16.93s；完整CPU300s请求为incomplete_timeout。181937a完整Linux CI41858691205通过；新3368f56 push后必须核其exact CI，不能以targeted替代。以下历史段落保留当时记录，当前状态以RESUME为准。

2026-11-03当前研究：[H017结果与图表](../docs/results/h017_relation_binding.md)。M3选择性K路径负结果、M4自然ago低概率、M5营业时间错误分钟高条件概率分别保留；M5只复用旧缓存，没有新增forward、评分或AUROC。完整代码9dc6600的Linux CI1101pass/1skip/9subtests。主目标六任务0.8仍未达成；当前无活动模型job，新范围条件对照尚未准入，当前恢复见research-state/RESUME.md。

当前状态更新：2026-10-29。默认主线是 `myproject/shared.py` 中的SharedDetector，冻结R4保留作历史对照。
完整金融测试及RAGTruth三任务完整TEST的V1/V2/V3已评分、评价和数值审计。
V3金融点值已超过.8，但置信区间跨.8；RAGTruth QA/Data2txt仍低于冻结R4。
最新统一[验证范围与结果](../docs/RESULTS_CURRENT.md)；V2/V3为研究候选、未晋升。
**尚无跨数据集完整验证或顶会就绪的方法结论。** 以下历史章节保留各自运行当时的状态。
本目录统一维护数据、评价、候选算法及后续实验代码；不再维护两个旧包名。

## 2026-11-02 本地理论与评价审计（CPU；没有新检测实验）

补充默认Beta共享模型的固定片段保序证明，以及pooled AUC按同片段/跨片段正负对的精确分解。`local_evaluation.auc_pair_decomposition`在既有封存评价路径调用；普通fullgap最终JSON保留原metrics并附加分解，不改检测器分数。跨片段项是全部跨片段对的补集，不能单独作机制归因。

新增27项参数化CPU检查，包括生产DP与独立copy/refresh枚举、AUC显式对子oracle、最终JSON/seal输出。最终完整套件149 passed、1 skipped、9 subtests passed；独立次级审查复现27 passed并通过窄范围代码检查。另有三个受限语法/KL fixture通过，导入的是与checkout字节一致的已安装模块。未执行LM/GPU、新标注或真实数据集实验。

默认Observer有eager/SDPA后端差异，但fullgap collector已通过measure_responses采用matched_sdpa；两种路径分开说明，不能推断所有历史结果都有后端混淆。每次历史运行仍需封存plan/artifacts确认。

[完整方法、逐行讲解与理论边界](../docs/source-reading/README.md)；[文献与独立审查后的方案](../refine-logs/theory-grounding-20261102/round-2-refinement.md)。检测器改进、事实校准、新颖性及顶会就绪仍未成立。

## 2026-10-29 主线工程统一与责任读出检验（完成、审计闭合）

myproject/shared.py统一shared/persistent/witness/responsibility；myproject/sequence.py唯一V2/V3内核。
公共Python/CLI/原文本数据集与缓存评分都调用主线，实验消融只调用核心函数。
R4 score与阶段诊断合为Detector.score_components，删除两个实验侧旧模块，无旧路径别名。
通用诊断兼容缺error_type的RAGTruth标签；tests/test_cached_transfer真实调用诊断新增回归。
源码最终299pass/1skip/9subtests42.49s；干净/tmp安装299/1skip/9subs44.63s；74源码资源精确集合，旧模块不存在。
完整2992答/448202token重算，41888原V1/V2/V3/R4及原轴数组逐位保持；不是拷贝旧分数。
真实8B公共CLI witness核验137token与归档逐位相同；新模型骨干次数未插桩，不报告为0。

唯一新responsibility诊断由同一模型推导：P(absent)=sigmoid(sV3)*sigmoid(d)，
r=sV3+d-logsumexp(0,sV3,d)，且r<=min(sV3,d)。独立三状态枚举检查通过，无新观测/FIT/权重。
完整AUROC/AP：QA .820332/.171270，Summary .722511/.079181，Data2txt .707506/.115250，
FinQA .743079/.269941，TATQA .785727/.381096，五组相对V3的两个主配对区间均负。
同完整组件top5%预算，monthly恢复，正确TAT3 FP14→6；错8/错科目仍漏，不能以局部改善掩盖整体下降。
按协议拒绝替代，不调权混回V3，不改默认；接口保留显式诊断/复现用途。

outputs/unified_mainline_20261029保存实际命令、双epoch源码/wheel/身份、seals、原轴门禁、全部评价与292条金融错例。
评分在epochA完成；其后epochB仅修复通用诊断可选error_type字段和scope，预测内核hash不变；两版测试/包均保留。
独立UNIFIED_MAINLINE_INTEGRITY_20261029逐条重算全部旧数组、新公式、各点指标/1000主配对区间及六例，
总体WARN只保留历史RAGTruth primitive GPU producer身份不完整限制；无数值/标签泄露/指标归一化/范围虚报问题。
same-family/provisional，不是顶会就绪；audit_completion.json追加闭合，未覆盖先前pending回执。
方法收敛见proposals/token_responsibility_20261029/FINAL_PROPOSAL.md；计划评分不是论文评审分数。
当前论文主张仍须独立自然标签和冻结R4/GroundHOG-U同口径证据；候选完整词表中心化是已知几何桥端点量，
尚未实现/测效/查新，先核对旧Rt等价性，不能换名重跑。工程完成与方法未获新增益分别记录。

## 2026-10-29 当前V3逐步说明与新两轮方法评审（完成，非新增效能实验）

算法逐步说明docs/METHOD_V3.md经新鲜审阅核对，与代码无实质出入。
独立上下文主评审初评3/10，round2区分“统计读出贡献”和“事实机制主张”后终评4/10、信心4/5；
补充评审独立4/10、信心4/5，未平均分数。完整报告RESEARCH_REVIEW_20261029_V3.md。
数学统一与精确推断成立，未证明z是真伪/来源条件/原生成器状态；新数据与强基线证据不足、创新增量有限。
最低门槛是冻结方法/主比较对象后，独立自然人工跨度评价，保留冻结R4并补GroundHOG-U；
若要讲错绑/合法推导机制，另须定位明确的来源编辑/no-op/修复证据。
均匀pi,c也有相邻同态5/6的先验偏好，实际n<=64有限；这是解析说明，不是新实验或推断错误。
本轮无runtime改动、无新模型调用或AUROC实验；METHOD/PLAN/HANDOFF更新不改变既有分数。
主请求gpt-5.6-sol/ultra与补充gpt-6.1-sol/ultra均被工具接受，provider级模型身份未独立验证；
same-family/provisional模拟审查，非真实会议审稿或接收。trace .aris/traces/research-review/2026-10-29_v3_requested。

## 2026-10-29 RAGTruth完整TEST当前共享主线（评分评价完成，迭代中）

统一prepare-dataset→transfer collect-cached→score --compare-r4→evaluate；全部三任务各900答/150来源、
全部六生成器各150条，2700答/424408token。原官方文本/当前tokenizer/source mask/原units与历史primitive逐条核对，
两世界rawfloat64差值与观测精确相同，未读取旧检测score；0新增模型调用。历史原始GPU权重/source未认证，
仅当前tokenizer/config身份、原生观测制品哈希与完整原轴可核验，不冒称新8B全量提取。
当前固定Beta共享：QA AUROC/AP .845010/.217477，Summary .725466/.089549，Data2txt .699617/.112565。
冻结R4同人口结果详见outputs/ragtruth_full_shared_20261029/baseline_completion.json。
三任务均确认共享低于R4（.892891/.352736、.762063/.096576、.798318/.190392），
全部结果写封存后接官方人工跨度标签，TEST已历史暴露不称独立确认。
源码263pass/1skip/9subtests，干净/tmp安装12pass，73源码资源/wheel/installed集合精确。
一次新fixture的浮点exact -1断言失败（11pass/1fail），改测试atol1e-15；运行观测对照仍逐位相等，不放宽生产验证。
read-only代码review RAGTRUTH_FULL_CODE_REVIEW_20261029 qualifiedGO/same-family/provisional。
下一固定persistent共享状态方案在baseline指标前已预声明，原跨度/原gap不变，不扫参数、不改默认。
这一阶段完整评分评价已结束，允许开始新候选源码；历史73源快照和原score/eval均不覆盖。

## 2026-10-29 联合持久状态候选V2（完整重放完成）

原d_t/原span固定，pi,c~Beta1,1，转移c*zprev+(1-c)*pi，非事后分数平滑。
experiments/research/persistent_shared.py正权Gauss积分+HMM FB精确联合边缘化；
c=0退化回现有Beta共享，能改变span内顺序。协议在原baseline指标之前写成，
实际score --method persistent --protocol强制绑定其SHA。prelaunch.json追加完整292金融迁移范围，
与3×900 RAGtruth用完全同一模型；源/读出/评价身份分离，旧obs版本保持。
更高source支持不等于条件相容性，新候选只检验非交换连续状态读出，不承诺核心事实机制已解决。
独立Fraction多项式枚举与积分/HMM误差<=8.9e-16，c0与旧cavity<=1.8e-15；
源码274pass/1skip/9subs48.91s，/tmp24pass/9subs28.72s，74文件wheel/installed精确。
paired_source_differences改为复用固定排序/同tie的加权统计，与原QA1000bootstrap区间误差<=1.2e-16；
种子/重采样来源/统计定义完全不变，仅加快运行。候选code review qualifiedGO，samefamily/provisional。
全量RAGTruth2700+金融292评分/评价均exit0，0新增读出模型调用。
AUROC/AP为QA .848133/.229203，Summary .735070/.102514，Data2txt .713894/.124746，
FinQA .735589/.279019，TATQA .789787/.383798。五组点值都高于V1，
但QA AUROC配对区间跨0；金融未过.8。outputs/persistent_shared_20261029/completion.json记录实际结果。

## 2026-10-29 潜在专家混合观测V3（完整重放及数值审计完成）

同一持久状态模型、原轴/原span，只把风险状态发射换成(pS+p0)/2：
e=logaddexp(0,d)-log2，等权先验已预声明，无FIT/任务调权/裁判。
同2992答/448202token、0新增读出模型调用，旧V2全部38896数组逐位相同。
完整AUROC/AP为QA .846975/.214833，Summary .762337/.104562，Data2txt .755716/.145209，
FinQA .802656/.325033，TATQA .842992/.465722。除QA外四组相对V1的两个配对区间均为正。
金融点值跨过.8，但绝对95%区间[.768988,.834952]/[.792669,.891909]均跨.8；
QA/Data2txt仍不及R4。Fin53错8、TAT54科目仍漏检；TAT3正确计算FP6→14，
孤立错误表现退化，未证明关系错配检测。完整表、固定六例、route/NLL/熵诊断见
reviews/RAGTRUTH_FULL_ITERATIONS_RESULTS_20261029.md。
V3相关测试源码27/9subs30.16s、/tmp27/9subs31.26s，74文件精确；不冒称重新跑完整274套件。
独立审计RAGTRUTH_ITERATIONS_INTEGRITY_20261029重算全部指标/1000bootstrap、原轴/基线/V2控制，
重hash11250原始文件及11模型文件，数字/制品PASS、范围WARN。fresh科学评审NOT_READY；
两类评审均same-family/provisional，指定正式reviewer不可用。未晋升默认、总体研究目标未完成。
outputs/persistent_witness_v3_20261029/audit_completion.json追加审计闭合，不覆盖先前completion。

补充固定首末六答真实8B核验：740token，local/full delta及route逐位相同，
新增240calls/6834处理token、47.042s、峰16321506304bytes，输出ragtruth_native_smoke_20261029。
六例一致不能反向认证历史全量producer身份；不称2700答全量新提取。

## 2026-10-29 V3必要性反证消融（全量评价与补充审计完成）

按科学评审冻结WITNESS_CONTROLS_20261029：仅c=0、raw d下界固定-log2、原span内固定置换再映回原token。
复用persistent_shared.py，经统一transfer score --method witness --ablate-witness --compare-r4调用。
相关源码29tests/9subs35.53s，干净安装29/9subs30.49s，74文件/wheel/installed集合与hash精确。
完整2992答/448202token评分评价均exit0；封存后核对41888旧V3/原轴数组逐位一致，再接标签。
0新增LM调用，不挑种子/参数/数据集。全部AUROC/AP如下（原V3见上一节）：

|任务|c0|固定截断|原span内置换后映回|
|---|---:|---:|---:|
|QA|.847532/.208876|.842594/.208976|.844615/.213015|
|Summary|.760087/.095260|.755162/.098214|.755925/.100447|
|Data2txt|.744908/.130104|.744493/.135846|.746135/.135095|
|FinQA|.789838/.294751|.796660/.322663|.791376/.315781|
|TATQA|.818675/.408548|.837732/.459900|.822525/.444746|

V3−c0的Data2txt/FinQA/TATQA两个指标区间均正，QA/Summary只有AP区间正。
V3−置换五组AUROC区间均正、除QA外四组AP正；447572/448202分数改变>1e-10。
V3−截断金融两组两个指标区间均跨0，尚未证明混合观测稳定优于简单稳健化，也不证明等效；
Data2txt两指标区间仍正、Summary AP正，QA二者跨0。不能以全局增益证明已识别事实风险。
详细定位/全部区间见reviews/WITNESS_CONTROLS_RESULTS_20261029.md；补充诚信审计
WITNESS_CONTROLS_INTEGRITY_20261029已闭合：2992原轴/41888旧数组、32真实短span独立枚举、
448202无延续独立DP、100指标组/60分层指标/30配对区间均核对通过，最大指标误差2.22e-16。
总体WARN仅范围限制，same-family/provisional/正式reviewer不可用，非新GPU审计。
只读code review与评分并发完成、早于标签评价，不是评分前批准；准确时序追加review_chronology.json。
outputs/witness_controls_20261029保留实际命令、身份与封存制品，audit_completion.json追加闭合，不覆盖旧pending回执。

## 历史科学实验

以下是先前实际执行并报告的结果，非本次目录重构新跑。旧experiment_results
原始内容已按用户明确要求删除，报告数字仍可查阅，但不能声称原制品仍能直接重放。
数据为反复暴露的开发/探索人口，不是独立未见TEST。

|实验|实际做了什么|结果|推进了什么/没有解决什么|
|---|---|---|---|
|Rt完整检测|比较有源/屏蔽源分布，构造归一几何中点，span共享混合拟合，再原链平滑；自然DEV276+受控80|三RAGTruth均不及R4；Hallu升但不胜简单基线|关闭该核心；概率正规化没有自动带来关系支持信号|
|无标点图分段|24不同来源回答，每任务8来源，4游走尺度；精确连续DP，保留token上下文和组内差异|词内非法切点修为0；没有完整检测AUROC|提供可复用分段工具，尚无证据优于标点的最终检测效果|
|图结构严格对照|匹配接收/发送/距离边际与合法段长集合|96格83相同、9正、4负；给定多步段长，单步71/72复现边界|复杂连接/更多步数的必要性未成立；不能推论所有span无效|
|共同状态检验|偶/奇层组拟合转移：无偏移、span共享、分符号共享、逐token偏移|单步条件CE增量共享.001281、逐token.009677；逐token96/96更好|共同状态有弱增量，常数共享不足；不是幻觉AUROC，层组也非独立数据|
|R4逐阶段错例|重建276答29103token各阶段，核对误报漏报|预算→奖金的prizes原gap=-1.370，创新只+.0125；原始与估计两处都有缺陷|明确来源促进≠关系支持；不能只归咎平滑或span|
|无标签FIT敏感性|固定recipe，按来源随机六次减半参考分布|QA AUC .894151–.894555，Hallu .471169–.473886|原分布内减半很稳定，不能用FIT选样解释跨数据集巨大落差|

### Rt与R4的自然数据对照：AUROC / AP

|数据/任务|冻结R4|Rt共享+原平滑|
|---|---:|---:|
|RAGTruth QA|.894336 / .516843|.749585 / .256138|
|RAGTruth Summary|.774180 / .103156|.661829 / .056552|
|RAGTruth Data2txt|.742037 / .208888|.718826 / .152798|
|HalluRAG|.472617 / .287484|.526021 / .323978|

Hallu是原句标签广播至token，不能与RAGTruth人工细粒度错误区间混为同一标注。
Rt的Hallu成绩也低于同批简单NLL .595040，不能作为新主线提升。

## 目前有用的研究结论

1. 整源source gap不能区分“使用来源里的词”和“正确使用来源中的事实关系”。
2. R4个体项先做窗口平均、CDF、区间中心化，再乘固定小权重，局部反证可能失效。
3. 不能把所有token强制成同一个风险；需要共享估计规律，同时保留局部差异。
4. 更复杂的图、更长游走或一个新的概率模型名称，不是目前缺失的关键。

这些缩小了研究问题，但还没有形成已验证的新检测算法。下一研究计划见
[PLAN.md](PLAN.md)，当前R4与待解目标见[方法说明](../docs/METHOD.md)。

## 工程执行

|事项|状态|说明|
|---|---|---|
|统一实验源码目录|完成|旧数据/评价包移至本目录；候选算法位于research/；不保留旧名别名|
|全局Codex运行规范|已写入并在本会话加载|全局AGENTS与项目规则均指向本目录|
|旧结果删除|完成|没有归档副本、软链接、路径映射或重放兼容层|
|公共算法数值测试、wheel与新鲜审阅|完成|源码与仓库外wheel均128通过、9子测试；1可选制品缺失跳过；评审问题已修复。不是新AUROC|
|新上下文观测资格测试|完成，未通过|104例、208次前向；残差8/8主对照反向，正常改序可翻转，关闭此读出|

新运行在本表增加：问题/假设、输入与来源划分、实际运行命令、源码身份、
原始观测/封存分数/指标位置、失败解释。结果只写项目outputs，不在其中写算法。

## 2026-10-27 上下文观测资格测试（完成）

固定候选为唯一词锚点的上下文位移残差；不使用实体、错误位置或正确性标注选锚。
8类语义×4来源/回答交叉×3保持含义的顺序，共96例；另4例后置关系词压力、
2正常拒答、2已暴露自然回答的覆盖诊断。新原始状态在
`outputs/contextual_probe_20261027/`，代码在本目录，不重建被删除的旧结果树。

[冻结协议](protocols/CONTEXTUAL_PROBE_20261027.md)与
[设计审阅](reviews/NEXT_CORE_20261027.md)、[实现审阅](reviews/CONTEXTUAL_CODE_20261027.md)
均在首次模型运行前完成。实现审阅发现并修复了遗漏顺序变化下错误样本的门槛漏洞。
全局和项目AGENTS现仅含代码规范，算法目标留在计划/研究文档。
CPU完整测试132通过、9子测试通过，1可选历史制品缺失跳过。
实际模型提取已完成：208前向、17,962输入token；加载后含I/O 28.480秒，
峰值张量分配16,202,008,064字节。残差、余弦和匹配raw source gap均0/8族通过
预定全部资格条件。残差/余弦8/8原始对照反向，保持含义的来源顺序调整后8/8转正，
因此读出受顺序/前文强烈影响；居中没有修复。source gap原始方向7/8为正，仍未
胜过全部正常改序；它不是完整R4。自然prizes没有词锚点，不能声称该漏检已解决。
关闭本固定读出，不翻符号/扫层/自然扩采。[实际结果](reviews/CONTEXTUAL_RESULTS_20261027.md)。
这轮无自然AUROC，不能宣称超过R4。独立诚信审阅单列，不用负结果包装提升。

## 2026-10-27 有向依赖指纹（完成，未通过）

第二轮不反转source gap，而独立提取来源内与回答内的物理删词影响矩阵，
对齐词身份后比较促进/抑制双通道分布的Hellinger距离；MSE单列不组合。
四类全新事实、双方独立顺序、全部64交叉对照，去重为32段；再加6段缺失/拒答
覆盖检查，共38段。词锚点与可见边共同掩码在读分数之前由文本冻结，
置换null保持每列边集合与有符号强度，只改发送者身份。掩码跨构造族共用
只是机制资格控制，不是可直接部署的自然检测器。

[冻结协议](protocols/DEPENDENCY_PROBE_20261027.md)与
[设计与Bayes等价分析](reviews/RECIPROCAL_DESIGN_20261027.md)。设计代理生成文档后
发生容量错误，因此不把该轮称作已完成的正式评审；首次GPU前另做新鲜实现审阅。
代码在conditional_dependencies.py、dependency_probe.py、research/fingerprints.py。
初次CPU检查曾因测试分词器只支持字符串模板而失败，已改用显式字符串模板分词；
另修正资格函数不必要地要求对照词null字段。完整源码测试137通过、9子测试通过、
1可选制品缺失跳过。所有新输出在outputs/dependency_probe_20261027。

实际采集完成：38段、492世界、1516次分块骨干调用、32,884输入token位置；
加载后含I/O 33.283秒，峰值张量分配16,098,960,896字节。所有重复intact概率差0。
Hellinger和同掩码MSE均0/4族通过；H仅9/16个顺序条件方向正确，8个目标的
跨顺序分离间隔全负，正确例真对齐未稳定胜过保边强度的身份置换。关闭本固定读出。
[实际结果](reviews/DEPENDENCY_RESULTS_20261027.md)、
[独立诚信审阅](reviews/DEPENDENCY_AUDIT_20261027.md)。独立复算38原始矩阵、
512个读出列及全部68比较；额外独立CPU测试75条删除边最大误差4.77e-7。
审阅总体WARN：构造数据非自然检测基准；未留全部删除世界逐token logp、旧准备
代码完整前像及模型权重全哈希。执行源码由当前源码和wheel保留，不能把矩阵复算
说成8B native逐世界概率被独立重跑。原始WARN保留，不为关闭状态警告而改结果。

最终源码完整测试137通过、9子测试、1跳过；审阅修订后五个相关测试通过；
仓库外安装最新wheel完整测试同为137通过、9子测试、1跳过。R4源码/参考15文件
与HEAD仍逐字节一致。无后台GPU模型任务、无新自然AUROC/AP。下一计划已更新，
优先核对同次原生前向的有符号写入及GroundHOG-U，而非继续改造已失败读出。

## 2026-10-27 确定性有向状态分段（完整评分完成，独立复核中）

按用户最新要求改进旧node2vec式分段：不采样游走、不对称化，直接拟合实际回答
token的有向attention目的地分布。区间共享目的地计数，保留每个token的条件预测与
残差；精确连续DP扣除固定边界编码费用。此分段状态不等于事实正确性；背景使用
整条回答fit-view估计，属于离线/transductive，不称无未来信息的在线预测。

源码：attention_states.py、span_pilot.py、research/directed.py；复用旧spans.py的
连续DP。稳定myproject数值代码/参考未改。协议在
[DIRECTED_SPANS](protocols/DIRECTED_SPANS_20261027.md)。冻结24来源/48回答，
QA、Summary、Data2txt各4个FIT与4个DEV来源，每源2回答；无标签选择，无生成器过滤。
同批无标签FIT分别重拟合参考；标点、原walk1、新direct_mdl及零边界费用消融均
重算对应local概率并走完整原R4评分和平滑。完整数据集/未见TEST并未运行。

新鲜实现审阅已检查原生attention行、精确DP、浮点差值及来源bootstrap；重要问题
在首次GPU前修复。源码与仓库外wheel均142通过、9子测试、1可选制品跳过。
原生采集运行于outputs/directed_spans_20261027，命令/身份在launch.json；
采集48/48、score封存27文件、evaluate接官方标签均已完成，退出码0。9252次骨干
调用/185961输入token位置，加载后测量循环含I/O 251.532秒，峰值16.416GB；
成本含全部对照，不能当单独候选推理成本。GPU已释放。

|DEV任务|标点同FIT AUROC/AP|新direct_mdl AUROC/AP|旧walk1 AUROC/AP|零费用消融 AUROC/AP|
|---|---|---|---|---|
|QA|.782950/.152578|.761668/.211786|.758464/.135139|.754941/.131832|
|Summary|未定义|未定义|未定义|未定义|
|Data2txt|.626191/.062614|.628116/.098251|.636254/.066907|.636834/.063739|

QA1392有效token/105正例、Summary1078/0、Data2txt1521/59；每任务DEV仅4来源/8回答。
Summary无正例不能算指标，不补挑错例。新方法AP点值增加但QA AUROC下降，不能称
稳定超过标点。新方法span约46–49 token，标点约22–24；跨head预测gain反而较低，
不能据此称语义共同状态已被验证。源码/参考15文件与HEAD仍字节一致。
[全部数值、区间及解释](reviews/DIRECTED_SPANS_RESULTS_20261027.md)。
[独立诚信审阅](reviews/DIRECTED_SPANS_AUDIT_20261027.md)已完成：WARN、same-family/
provisional，无诚信FAIL；48原生制品、12参考、15最终分数、标签和配对区间全部
独立复算吻合，参考/分数最大绝对差0。WARN保留4来源、Summary无正例、已暴露DEV、
权重分片未全哈希等边界。
[科学结果审阅](reviews/DIRECTED_SPANS_RESULTS_REVIEW_20261027.md)已完成：逐阶段分数
逐值一致，细分的固定窗口/中心化会消除个体差异；QA AP提升主要出现于后端创新，
共享区间项本身未更好。该代理数值完成后曾容量中断，短重试仅整理既有证据形成报告；
保留原迹，不冒称不中断审阅。生成的逐行诊断均在outputs下，不纳入Git。
不再额外运行审阅建议的16例来源错绑演示：源影响不等于关系支持已由历史实验确认，
重复此已知结论不能推进当前算法。尚无完整新方法/未见TEST/顶会就绪结论；无后台
实验，代码未提交推送。

## 2026-10-27 主线集成与FIT消融（完成并审计）

有向attention与连续分段现位于myproject/attention.py、segmentation.py，实验直接
导入；删除迁出源文件，不留算法副本/无调用兼容层。主API按参考绑定的segmentation、
innovation、calibration测量与评分，旧默认R4全部12组原分数仍逐值一致。
predictive模式用相关head预测准则从原费用/零费用中选择，本批48/48选零费用，
不支持非平凡自适应必要性。point移除个体项的预平均，但仍居中；singleton原观测
进入其区间来源项，不能说总信号被清零。no-FIT提供固定有界对照，不是事实后验。

独立代码审阅发现参考文件缺失/NaN/未排序可能静默评分，已修复且覆盖加载拒绝测试。
源码与仓库外wheel均157通过、9子测试、1可选旧制品缺失跳过。原15文件的字节不变
声明仅适用于之前运行：本轮修改了API/Detector配置，当前保障是旧默认分数逐值一致，
原参考资源未改。新wheel与原run的wheel分别保留，不覆盖。

七组消融复用原48回答/24来源，CPU、0次新模型调用；42评分/参考文件先封存再接标签。
新输出outputs/fit_ablation_20261027；资格/安装日志outputs/integrated_fit_20261027_checks。
命令/源码/配置/实际退出码在新run的execution_record.json，明确记录是执行后转录。

|系统|QA AUROC/AP|Data2txt AUROC/AP|
|---|---|---|
|原标点/window/task FIT|.782950/.152578|.626191/.062614|
|原依赖/window/task FIT|.761668/.211786|.628116/.098251|
|标点/point/task FIT|.784904/.158310|.626411/.097562|
|原依赖/point/task FIT|.759241/.157785|.621368/.089816|
|预测选粒度/point/task FIT|.758893/.134986|.637100/.070435|
|同上/no FIT|.715670/.117374|.631918/.059360|
|同上/pooled FIT|.753188/.132843|.648404/.075680|

Summary1078token/0正例，全部AUROC/AP未定义；每任务仍仅4DEV来源，已暴露开发人口。
点分支在原标点下提高AP，在粗依赖下下降；没有噪声方差证据，不能断言下降完全
来自降噪丢失。细粒度Data2txt创新归零由52.79%降至0.657%，仍不等于稳定检测收益。
当前CDF FIT提高固定组合的排名，但单特征排序未新增信息；跨任务FIT有升有降。
这检验的是边际坐标/后端组合与参考专属性，不能证明所有FIT都必要，或学到了语义真值。

[统一设计/实际结果/审阅](/share/home/tm902089733300000/a903202310/lys/codex/research/unified_fit_20261027/RESULTS.md)。
统一cavity模型仍为待验证标准参照，尚未实现；V1的NOT_READY审阅裁决保留，不通过
给标准经验Bayes改名制造创新。当前无后台GPU，代码未提交推送，顶会研究目标未完成。

新鲜独立诚信审计已完成，见共享目录FIT_ABLATION_AUDIT.md/.json。32040独立
数值检查、408实际分词器API检查、55执行/安装包检查均通过，21参考/分阶段分数
逐值复现，18组配对区间吻合；总体WARN、same-family/provisional。范围保留
已暴露48回答、每任务4DEV来源、Summary无正例、全部选择零费用和模型权重身份
限制；自定义token AUROC/AP不等于官方case级F1复现。统一设计V2已审阅为连贯规格，
科学就绪仍NOT_READY，尚未实现新cavity模型，不以数学参照冒充新贡献。

## 2026-10-27 联合共享估计（执行中）

已实现experiments/research/pooling.py和pooling_run.py；固定full对照，Beta共享状态，逐token精确积分，不预平均/CDF/route混合。FIT仅估计同一模型alpha/beta。冻结合同与代码审阅在共享cavity_implementation_20261027。163源码测试+9子测试通过、1跳过；独立数学与集成审阅GO。旧48的9系统评分/评价已退出0；新增144回答/72来源与旧24来源无交叠，选择及原始输入hash已冻结，attention-only采集运行中。原始概率复用历史cache，48资格例full/local/route逐值一致。新旧人口分开报告，均为历史开发人口；不称独立TEST。新结果尚未闭合/审计，不晋升默认方法。

## 2026-10-27 两轮联合共享估计与分段迭代（评分/评价完成）

第一轮旧48/24来源与新144/72来源各9系统完成，新旧来源无交叠；每个人口一半回答FIT、一半DEV，非未见TEST。固定full原始对照，精确Beta共享后验保留个体观测；FIT只学alpha/beta，不用CDF或线性拼接。源码experiments/research/pooling.py及pooling_run.py。新增attention144完成2029backbone/118032token/161.398秒，峰16.369GB，历史概率48资格例full/local/route逐值一致。第一轮数值审计完成，所有分数/54指标/30来源配对区间通过；正式指定审阅型号不可用，补充审阅same-family/provisional，非正式接收结论。

新增人口QA/Summary/Data2txt AUROC：raw .665267/.691666/.654480；predictive固定 .733615/.806476/.722789；predictive FIT .739310/.807481/.736031；标点固定 .803728/.825900/.774877；标点FIT .836323/.842967/.793687；同人口R4 .809571/.874404/.781121。标点FIT是预定次级系统，不替换原主比较。FIT存在条件增益，非完全必要/完全无用；新QA同模型标点FIT的AUROC差来源区间[.014875,.050849]。旧48存在反证，Summary无正类保持null。

第二轮冻结新合同后，以双视角留一残差预测替换分段准则，其他输入/后验/FIT不变；两人口仍全跑并保留7个原对照。源码state_partition.py/loo_pooling.py，不按标点/等分/费用搜索。新loo_fixed AUROC .728413/.790754/.720360；loo_fit .734028/.791160/.725048，未解决分段，平均4.229token，未晋升。旧48均长4.252token，QA AUROC/AP下降，Data2txt局部增加。源码和仓库外安装wheel完整167测试+9子测试通过、1跳过，15旧默认/冻结评分逐值一致；新数值审计另列。所有原46源码/参考绑定保持，第二轮只添加源文件。

原始制品outputs/cavity_old48_20261027、cavity_new144_20261027、cavity_sources_20261027、loo_old48_20261027、loo_new144_20261027。两个checks目录记录命令/源码/wheel/实际退出及失败，原结果不覆盖。first new144 evaluator跨用户轮次丢失退出会话，仅保留完成日志/原子结果，不虚报观测OS0；第二轮4阶段全部实测OS0。一次事后定位诊断JSON int64发布失败已保留，仅修计数字段序列化后发布，不改评分。

[第一轮完整结果](/share/home/tm902089733300000/a903202310/lys/codex/research/cavity_implementation_20261027/RESULTS.md) · [第二轮及失败分析](/share/home/tm902089733300000/a903202310/lys/codex/research/cavity_implementation_20261027/LOO_RESULTS.md)。协议精确副本在experiments/protocols/SHARED_POOLING_20261027.md与LOO_POOLING_20261027.md。下一优先为边界不确定性与来源专家的联合后验，不再第三种路由代理；尚未实现。本轮仅RAGTruth三任务，未新增HalluRAG。研究目标未完成，默认R4未替换，未提交推送。

补充独立数值审计现已闭合：第二轮192分段/2977221合法候选区间独立DP一致；39706区间显式逐目标SSE最大差1.15e−14；12新评分最大差3.69e−14、6FIT/18起点、42原对照文件、54指标对、30来源配对比较均吻合。答内定位诊断的所有9方法/计数/54宏指标对也复核一致；不是重新定义主评价。机器/文本报告LOO_AUDIT.json/.md、正式route不可用与补充WARN明确分开。最终绑定outputs/loo_checks_20261027/final_review_bindings.json，共享COMPLETION.json。不改变默认/主比较，目标未完成。


## 2026-10-28 机制解释核对（无新实验）

按用户问题核对spans.py、Detector和两轮共享模型原始指标；node2vec衍生代码未删除，
walk1小人口QA下降/Data2txt局部增加，未测的多步完整检测不称失败。严格区分旧CDF FIT
与新Beta先验FIT、图内共享结构与总体背景校准、统计排序收益与幻觉生成机制。
不再以“幻觉更罕见/少关注来源”解释有效性；当前只支持有限的共享与条件FIT增益，
没有完成频率/难度/标注连续性等机制解耦。说明追加docs/METHOD.md。本次未改算法、
未新跑模型或声称新AUROC，后续候选仍须先冻结再比较。


## 2026-10-28 既有FN/FP真实样例重建（完成，非新方法实验）

按用户最新要求先复盘，不重起候选。新增case_audit.py复用原R4诊断与Beta pooler，
从仍存native_support_ragtruth_all及官方RAGTruth JSONL核对10988、3180、2390、3182
四例728token；没有恢复已删除旧结果树，无新GPU/总体AUROC/参数修改。完整证据
[FN_FP_REVIEW_20261028](reviews/FN_FP_REVIEW_20261028.md)，原始制品outputs/fn_fp_cases_20261028。

预算→奖金属于原始两路gap都未读出关系错误；null→否定的does却有local+11.995737、
full−2.193166，R4pair point+4.901285后被低共享量/受限创新压住。local只丢区间之前
历史，仍保留区间内前缀；prizes在首区间，两路本来语义条件相同，不把它们近同当
对local自我强化的新检验。FP2390 Earth有84.0549%LME权重，正常Story被广播高风险；
新Beta在该例阻断广播但未修复Earth原始异常，不宣称同预算FP减少或新全局收益。
另核对Elvis的屏幕呈现限定遗漏；同源正常budget来自另一生成器，不伪称单因素对照。

独立审阅/复算：20输入hash、728轴/标签、全R4阶段与Gauss积分通过；最大final差
4.44e−16、Beta7.11e−15。相关10测试通过；新wheel在/tmp运行同一CLI，四例记录与
源码全部逐值一致。命令/wheel/安装验证见outputs/fn_fp_cases_20261028_checks/execution_record.json。
本轮只定位检测器失败位置，不声称解释原生成器因果机制。正式skill评审路线不可用，
实际补充协作/数值核验状态写明；推测性新对齐方案暂置于证据复盘之后，不取代既有研究。


## 2026-10-28 配对post-token来源响应观测（启动时记录，完成结果见后）

沿真实FN/FP继续，不改分区优先级。scalar gap不能区分部分错绑，本轮测同一原token
有源/无源post-state差的local方向一致性，单点判别与来源响应幅度控制并列；不是
已关闭Rt完整词表几何归一的重跑，也不是旧跨文本词锚点残差。冻结合同protocols/
DELTA_LOCAL_20261028.md，候选delta_local.py、observer delta_observe.py、runner
 delta_local_run.py；原R4代码不变。主读出direction_dependency，标点与FIT白化及其他
控制全部事前固定。两人口原48+追加144、四旧案例去重196；无效应选样/标签拟合。
171源码测试+9子测试通过1跳过，新鲜补充代码审阅GO；正式指定型号不可用不伪称正式
通过。启动时真实原生采集session58459运行于outputs/delta_local_20261028；该轮现已完成并审计，
具体评分/AUROC与失败结论见后续完成记录。逐token预测/消费后状态严格分开；两条件同SDPA，实际旧cache数值差异会报告。


## 2026-10-28 配对post-token观测（完成，主假设失败）

196回答新采集及两人口16系统全部完成，5585backbone/213634处理token/394.616秒，峰16.377GB。主方向dependency追加QA/Summary/Data2txt AUROC .425886/.410701/.521046，AP .093381/.021703/.042079；同批R4 .809571/.874404/.781121。未翻转方向或选取个别成功案例。白化无改善，原token全保留。Elvis/does局部排名改善不代表总体检测改进；prizes仍未识别。

独立审计复算196hash、84新评分文件/12FIT、12R4字节拷贝、96指标与42来源bootstrap，最大几何差2.553e-15、完整score9.237e-14；官方GT pinned upstream SHA一致。源码/仓库外安装171 tests+9subtests通过1跳过。原生新旧present logp平均差.015178/最大.812492已披露；prepare两文件早于最终修正，真实prelaunch/collect/score 52项身份一致。正式route不可用，补充WARN/same-family/provisional。

完整[结果及失败定位](reviews/DELTA_LOCAL_RESULTS_20261028.md)、[独立审计](reviews/DELTA_LOCAL_INTEGRITY_20261028.md)，原始outputs/delta_local_20261028。两个evaluate实测exit0，6任务含旧Summary无正例null。默认R4不变，未达到研究目标。

## 2026-10-28 来源直接写入与历史继承分离（启动时记录，完成见后）

从同一FN/FP继续，更换观测，不再优化分段代理。protocols/SOURCE_WRITE_20261028.md、source_write_observe.py和source_write_run.py，复用现有Beta精确local推断；当前token反证可覆盖共享倾向。每层真实attention×V按source/history/other partition保留方向，经W_O再合并；原token预测行P+t−1，区分native BF16 activation ablation与FP32 smooth-gradient诊断，显式记录原生舍入误差。已有GroundHOG-U相关工作，不能称signed write本身为新贡献。

固定同196回答/原分区/原标签轴、16系统7比较，primary direct_pool_dependency，不训练标签、不用judge、不翻转、不选择层。prepare前代码审阅修复父制品校验、当前模型hash、跨phase源码冻结；175源码测试+9subtests通过1跳过，独立4test与BF16重复trace通过。实测prepare exit0；启动时GPU collect session12439运行于outputs/source_write_20261028；本轮现已完成且主假设未改进，详见后续完成记录，不能称已识别错绑。


## 2026-10-28 direct source write（完成，未改进）

196回答全部完成，2766backbone/160175处理token/772.002737秒，峰18.464GB。主direct_pool_dependency追加QA/Summary/Data2txt AUROC .683424/.702844/.627756、AP .234434/.086837/.063284，均未胜同full池化/R4。Earth/Story的来源写入支持方向与整源gap不同、prizes轻微转正，没有转为总体可靠检测；does和Elvis依旧有源促进，不能当修复。16系统/六任务及7比较全部封存评价exit0，独立96指标/42bootstrap/30bytecopy与196+196原生hash复算通过。

源175tests+9subtests1skip通过。首wheel多两已删模块（旧build/lib缓存），严格安装源码集合检查失败但数值代码相同；保留原失败制品，清理生成缓存后clean wheel/installed55精确集合及175tests通过。规范已补AGENTS。原实验55源码快照在outputs/source_write_20261028/installed_clean；之后第三轮扩展runner不冒充本轮源码。BF16累计舍入项显式保留、norm重放0；不宣称FP32分区精确native additivity。

[完整结果](reviews/SOURCE_WRITE_RESULTS_20261028.md)、[独立审计](reviews/SOURCE_WRITE_INTEGRITY_20261028.md)、[标准共享修正界](reviews/LOCAL_POOLING_INFLUENCE_20261028.md)。正式route unavailable，补充same-family/provisional/WARN。自然检测主假设失败，默认R4与研究目标状态不变。

## 2026-10-28 source/history相对写入（完成，未改进）

明确为看过同四旧case之后的自适应开发读出；原normalized activation experts logratio=direct_gap−history_gap，没有新增观测/信息或新GPU，不把代数重新命名当创新。复用source_write_run.py新增--reuse、同Beta1 pooler，原16系统96份cohort文件bytecopy不变，仅新增competition_gap/dependency/punct三个系统；独立review真实无标签集成、门禁破坏控制及728case积分通过，14目标测试通过。

19系统/六任务及4比较已完成；主dependency追加AUC .659389/.628987/.523033，AP .157403/.035537/.047905，未改进。does raw转正但共享后同答risk尾61.05%，没有修复；不以分数零点作为真假阈值。第三轮独立114指标/24来源bootstrap、96bytecopy、原轴及49320新score复算通过；仓库外wheel56源码严格一致、179tests+9subtests通过1skip。真实执行prepare91052/score42949/eval74617+24219全部exit0。

完整制品outputs/write_competition_20261028，[结果](reviews/WRITE_COMPETITION_RESULTS_20261028.md)、[审计](reviews/WRITE_COMPETITION_INTEGRITY_20261028.md)，protocols/WRITE_COMPETITION_20261028.md。无Hallurag新结果、非未见TEST、未升级R4/提交推送。共享模型的“保留反证”有代数性质，但三轮均未提供足够的关系兼容性观测。后续不重复方向一致性、signed source alone或同两gap差分作为“新机制”；新增核心候选须针对具体用途/极性/范围约束给出可测的新信息，再允许自然数据验证，不能要求先具备真值证明或四例全对。研究目标尚未完成。


## 2026-10-28 金融迁移与R4组件诊断（启动记录；已完成见下）

用户要求分析原R4优势与数据依赖并跑金融token检测。独立类别诊断216行已完成，见reviews/R4_COMPONENT_EVIDENCE_20261028.md；原Beta标点FIT的QA/D2条件改善也已独立重算，不再将最近三轮失败概括成所有改动无效。可支持叙事/可改步骤与带标签DEV选择限制见reviews/R4_PAPER_DESIGN_20261028.md及R4_NARRATIVE_REVIEW_20261028.md。

金融主源调研未找到已核实的英文自然人工金融token集；FRED发布原有合成/silver字符标签可用，FinQA162+TATQA130全test逐字重构，原Q/生成prompt不可恢复，采用统一source-only completion prompt，不给编辑任务/Text/mark修复。PHANTOM为answer分类、FAITH为mask预测，未偷扩token标签。适配器在datasets.py/span_markup.py，通用native_dataset.py缓存四条件导出的两路delta及route、同QA recipe frozen与64金融无标签FIT(32+32源独立)两ref和13stage，非另建项目。

初次7090采集首条前exit1：Llama模板trim尾换行，旧encode_source搜索原prompt失败，无COLLECTED/无分数；保留native、collect.log、failure_v1。最小修复inputs.py只在原prompt不在chat时按strip内容对齐、要求完整source不变；原正常路径同。独立actual tokenizer356条全匹配旧码在trimmed输入上的dictionary，真实tinyLlama测量逐值回归通过。190源tests+9subtests/1skip，reviewed installed188是在修复前的中间版本，不冒充最终；最终trim-fix wheel/190 installed验证和native_v2将独立记实。数据/prepared/labels不改。

原始制品outputs/finance_transfer_20261028，冻结protocols/FINANCE_TRANSFER_20261028.md；启动时没有金融效果数字，成功采集/全预测封存后再接标签。

## 2026-10-28 金融迁移与统一入口（完成）

实际356答采集完成：64FIT+FinQA162/TATQA130完整test，9897backbone calls、387691处理token、283.26秒、峰16,540,873,216 CUDA bytes。native_v2全部观测及scores_v2全部292预测封存，evaluation_v2/evaluation.json、artifact_diagnostics.json均完整；首native失败目录保留。主score/evaluate退出回包未完整保留，按封存及独立全量重算确认完成，不补造exit code。13预定读出、所有类型、3×2×1000来源bootstrap均验证。

|固定读出（AUROC/AP）|FinQA|TATQA|
|---|---:|---:|
|point|.593050/.165205|.649406/.260281|
|区间共享prior|.722855/.285470|.748064/.438633|
|完整冻结R4|.661671/.145231|.711049/.220941|
|金融无标签FIT完整R4|.666400/.128507|.710326/.177704|

金融FIT没有一致改善：两组件AUROC差CI跨0，AP差CI为负。global prior→final明显下降不能泛化为所有局部定位都下降：TATQA混合回答macro反而提高，且排除正常回答、不能替代主指标。事前固定相对位置基线FinQA .730878/.243068、TATQA .669264/.189823，必须保留合成构造/位置解释限制。关系、实体类型弱，unverifiable较强，但类型人口少，不作普遍机制结论。原问题缺失、source-only提示、silver标签及精确来源而非公司分离均明确披露；不将prior事后晋升成未见test确认的方法。

完整[结果](reviews/FINANCE_TRANSFER_RESULTS_20261028.md)、[独立审计](reviews/FINANCE_TRANSFER_INTEGRITY_20261028.md)、[代码审阅](reviews/FINANCE_TRANSFER_CODE_REVIEW_20261028.md)。11模型文件含4权重shards与59源/资源hash全部核验，全部356原轴/13分数独立复算最大差6.66e-16；全部16辅助global/macro读出一致。正式指定reviewer不可用，补充same-family/provisional，总体WARN；不是科学审稿通过。

主结果冻结包installed_trim_fix的190 tests+9subtests/1skip通过。解冻后仅收敛入口：`r4-experiment prepare-dataset` → `transfer collect` → `transfer score`（无标签FIT+两参考评分封存）→ `transfer evaluate`。直接复用native_dataset公共函数，删除重复独立CLI；算法/门禁函数AST保持一致。中央CLI实质CPU集成测试包含缺标注评分、special排除及原轴评价，fresh code review无重要问题。旧金融结果使用旧59源snapshot；当前源码正确拒绝其旧seal，未为入口变化重复GPU。最终安装验证单独记在本run的integrated_*制品，不冒充原实验版本。没有默认方法晋升/新HalluRAG/顶会就绪/本轮提交推送。

统一入口最终验证完成：源码191 tests+9subtests/1skip（35.44秒），清build重建wheel后/tmp完整测试同为191+9/1skip（35.17秒），installed_integrated严格59源码/资源集合相同，仅两个CLI文件较主实验快照变化。安装CLI help及git diff --check exit0，完整命令/实际exit/制品hash在outputs/finance_transfer_20261028/integration_completion.json。项目AGENTS补充统一入口规范，README/RESEARCH/交接同步；没有新增依赖或重复算法实现。

## 2026-10-28 金融错例→表格绑定→源约束联合推断（完成，小变化、未解决）

沿完整292金融test已暴露人口分析，不是新未见TEST。统一入口transfer diagnose，新纯函数table_bindings/constraints，I/O transfer_diagnostics；复用原封存观测，无新GPU/judge/金标准实体。read_native用于保留旧身份的inspection，validate仍严格拒绝不同源码/环境重放R4。

真实错误表明原始强信号也被后端压掉：TATQA50年份首token raw full-gap风险百分位99.5%→共享36.8%→final51.5%；51 monthly 98.8%→6.9%→9.4%；54 Acc 97.8%→24.6%→19.0%；FinQA59 revenue 99.2%→25.0%→15.9%。同实体其余subtoken可反而获源促进，不能只靠首token异常。全类prior捕获后final丢失正token FinQA167/TATQA151，恢复9/3；entity/relation/temporal的漏检与无依据错误不同。FP也有明确正常词/计算步骤：FinQA670 FP中243在tagfree答，TATQA287 FP全在error-bearing答内正常部分。完整结果reviews/FINANCE_BINDING_RESULTS_20261028.md。

v1诊断完成但有parser错误（跨行/后继年份标题绑定、数字行覆盖表头），fresh reviewer复现；保留diagnosis、dist_v1、closure_v1，不覆盖。V2严格首literal年份header、不跨行/后继colon；header多行percent-change回归。全292答52可检查对，12冲突(5+7)，其中6对条件和金额都在源中；仅8答受影响。所有12的year端点与silver正区间相交是小型已暴露诊断，不是普适正确率。对称两端标记52token=24TP28FP，不能偷选gold年份端点。

第三轮主模型在原冻结chain势上条件化“每个冲突两端至少一个token风险”，bitmask精确前后向，非新指标线性相加；binding_no_chain仅去lambda作消融。主binding_chain FinQA AUC/AP .662354/.145350（R4 .661671/.145231），TATQA .711975/.221236（.711049/.220941）；top5 TP仍133/101，未改善预算召回。no-chain .667062/.150730、.717033/.231949，TP148/108与旧frozen_unary相同，不能将其去平滑收益归因于新约束。

独立partition复算原模型满足全部子句概率.572–.944：至少一个subtoken事件过宽，不能定位整语义单元。monthly、entity、运算/单位都尚未被有限表格约束覆盖。下步应明确原子语义单元的支持状态、条件/行/单位对齐与可验证派生，保留个体信息及归因不确定性，检验token化/长度偏差；不是继续CDF/路由调权。这仅是下一设计，未声称已实现。

三轮实际diagnose exit0，原父13读出、GT、原轴不改；新18读出先封存后接silver GT，前后复验父seal/protocol/derived NPZ。当前62源/资源身份与仓库外wheel精确相同；源码205tests+9subtests/1skip（51.31秒），/tmp安装205+9/1skip（39.56秒）。独立96随机小链全枚举max4.97e-14；全量独立端点枚举+两状态HMM重算新向量max2.56e-13，284无冲突回答主score与R4完全相同。review报告FINANCE_BINDING_CODE_REVIEW与INTEGRITY_20261028；formal不可用、same-family/provisional/WARN。轮次wheel/源identity/命令/日志保留outputs/finance_binding_diag_20261028，不升默认、不声称顶会成功、研究目标未完成、无本轮提交推送。


## 2026-10-28 Token优先共享与主线晋升（完成）

原292答/23794token先封存再晚接silver标签，Beta(1,1) full cavity主模型FinQA AUROC/AP .728275/.257152、TATQA .768769/.355865；旧R4 .661671/.145231、.711049/.220941。个体观测未平均/居中，修正有界，但并非每步依据判定器。旧prior AP .285470/.438633仍更高；TATQA tagfree正常标注答新增33个top5 FP，正确80.8 USDm推导的million误报仍在。独立全部数值/词轴/标签审计通过，整体same-family/provisional/WARN，非正式PASS。原62源快照outputs/token_first_pooling_20261028/installed，205tests+9subtests/1skip。详见reviews/TOKEN_FIRST_RESULTS及INTEGRITY_20261028。

随后按用户授权，唯一pooling实现移入myproject，SharedDetector及API/CLI/数据预测/原生评分默认固定共享；R4显式历史对照。不需要FIT或R4 reference；测量后端仍有未使用视图，未声称减少观测成本。新63源快照outputs/shared_mainline_20261028/installed，292答23794token生产共享与旧cavity逐位一致、显式R4全部重放一致；真实8B两旧答124token与缓存逐位一致。源码211tests+9subtests/1skip最终41.56秒（旧完成回执44.05由独立completion_corrections.json订正）；/tmp安装211tests48.18秒。第一次wheel生成失败保留，清生成缓存重试成功且63源/资源精确集合。详见SHARED_MAINLINE_CODE_REVIEW与运行完成回执。无提交/推送，AUROC>0.80目标未达。

## 2026-10-28 词内层级共享（完成，仅小幅变化）

原span不改，legal_atoms词内Beta层级允许token差异；primary hierarchical_score，固定浓度1，atomic_shared仅强制词内同标签消融，不得事后替代主模型。统一transfer explore-shared复用原金融观测、无新FIT/GPU，292答23794token先封存后评价。FinQA主AUROC/AP .731481/.258965（固定共享.728275/.257152），TATQA .772781/.357992（.768769/.355865）；top5 TP314→319、205→207。atomic分别.735118/.267982、.773308/.373290，但抹平词内差异，不升主线。TATQA tagfree FP33→35，FinQA词内FP39→46；不能仅报告TP。

代码独立64精确有理多项式/全枚举核验max7.11e-15；全量独立层级核验max1.03e-13，基线与旧生产快照逐位一致，全部指标/类型/case吻合。正式reviewer不可用，same-family/provisional/WARN，silver范围限制保留。源码217tests+9subtests/1skip43.27秒，64源wheel快照outputs/hierarchical_shared_20261028/installed；安装测试另见完成回执。详见protocols/HIERARCHICAL_SHARED与reviews/HIERARCHICAL_SHARED_CODE_REVIEW/INTEGRITY_20261028。两个AUROC未达0.80，下一单项为同模型无标签先验FIT，非旧CDF FIT。

层级轮补充完成：仓库外安装217tests+9subtests/1skip44.50秒，64源/资源精确集合；completion.json已写，非仍在运行。

## 2026-10-28 共享总体先验FIT（已启动，尚无结果）

protocols/SHARED_PRIOR_FIT_20261028.md事先冻结，中央transfer fit-shared-prior复用原64训练来源无标签full gap拟合一个总体alpha/beta；不按金融组件各拟合，不改默认固定共享，不做CDF FIT，不新增GPU。目标测试7passed，完整测试及实际292答运行已启动，原始输出outputs/shared_prior_fit_20261028，完成后另记实际数值。

共享先验FIT完成：64源159span4475token、单个全局alpha=.5831885615/beta=3.6067090658；既有梯度阈值接受第一起点（SciPy success=false但gradient3.50e-8<1e-6），第三起点success=true/2.24e-10，全部原始尝试保留。FinQA AUROC/AP .751182/.277031、TATQA .782001/.381970；仍低于.8。FinQA top5 TP314→300，TATQA205不变、tagfree FP33→39，80.8推导million仍误报。源码217+9subtests/1skip42.49s，64源安装集合精确一致，/tmp实质CLI集成1passed28.97s。新观测/评分先封存再GT；code review GO/WARN。按用户最新纠正停止FIT方向，不升级默认，不继续调参，下一项针对真实错例的关系null专家。outputs/shared_prior_fit_20261028已完成回执，数值审计报告独立补入。

共享先验FIT独立审计已完成：评分Beta积分max4.494e-13、所有30主指标行/12类型及23794case token核验；reviews/SHARED_PRIOR_FIT_INTEGRITY_20261028，overall WARN/same-family/provisional，formal REVIEW_UNAVAILABLE。

## 2026-10-28 关系null来源专家（已开始，尚无全量结果）

按用户纠正停止FIT，单项新核心为协议RELATIONAL_EXPERTS_20261028。保留表格literal多重集，按列跨行/按行跨列移动numeric单元，原/空/两null来源对原回答teacher forcing，同span共享latent world及Beta1 token state，积分世界证据后求token边缘，非风险线性叠加。原逐字$/%类保持不代表隐含单位USDm等语义量纲全匹配，必须保留此限制；纯文本关系未被表格扰动破坏。

首次header回归实际失败（220pass/1fail/1skip/9subtests），没有GPU；修正前置literal年份header后关闭header区，保留body中年形员工数。v2全部221pass/1skip/9subtests43.30s；6实际旧错例GPU收集完成并封存，snapshot dist_v2。机制结果不算修复：2020第二token从-2.048到-1.061仍低；Accrued剩余token从强负数拉近0而非获得明确反证；monthly1.433→.159被削弱；正确million6.280→6.084仍高；错数8的关系对照gap反为-7.5，source swap不天然就是合理反事实。不能以绝对分数变化宣称AUROC提升。完整机制数值在mechanism_cases.json，无6例AUROC宣传。

v3只补collection结束native/protocol hash门禁、原prompt hash及K1逐位退化分支；数值模型不调参。全量292观察/结果待实际运行，不能把本记录当完成。

关系专家v3启动后发现实际TATQA98/115年份2019/2018(1)header被当金额改写。全量collector主动SIGINT exit130，outputs/relational_experts_20261028/full为不完整失败制品，无seal/score/evaluation，不引用效果。v4前置单YEAR header也冻结，保留footnote列，新增2参数回归；15纯/16实质目标集成通过28.44s。historical shared_score补入同一结果轴。新prelaunch_v4，实际后续全量full_v4及comparison_v4；仍需独立全源变换检查，不把v2六例或v3 partial冒充最终人口。

## 2026-10-28 关系null专家最终v4（完成，拒绝主候选）

full_v4完整292答/23794token/887span：8988backbone/674939处理token/250.253608秒、峰16399248384 CUDA bytes，collect和compare实际exit0。comparison_v4先封存预测再原silver标签；13旧读出逐文件保留，加shared历史/4新读出共18。主world_shared FinQA AUROC/AP .693095/.248915、TATQA .674423/.305032，均低于同次SDPA匹配控制.729517/.260282、.768258/.355036；历史固定共享.728275/.257152、.768769/.355865另列。top5 TP匹配324→315、199→181，不挑辅助NLL救援。目标未达、不晋升、不FIT调权。

独立全量积分max2.49e-14、36指标行/12类型/全部case数据吻合，11模型文件sha含权重核验。数学和原轴正确，失败主因是观测/状态含义：大量关系null近原（正常84.58%/78.78%、错误73.43%/60.99%的pair abs gap<.1），正常风险中位数-1.099→-.219/-1.888→-.303，分离恶化。不能说所有后验偏近零；错误token仍更偏PE。v4 header保守修复改变6来源，FinQA21年形数量误冻是覆盖限制；v3 partial exit130无seal无分数保留。源v4相关16tests28.44s，安装最终223+9subtests/1skip46.36s、66源严格集合；旧v2 full221另记。完整结果reviews/RELATIONAL_EXPERTS_RESULTS_20261028，代码review同名CODE_REVIEW，审计同名INTEGRITY；overall same-family/provisional/WARN，非科学PASS。

关系专家独立审计已全部闭合，报告RELATIONAL_EXPERTS_INTEGRITY_20261028：WARN/formal REVIEW_UNAVAILABLE/same-family provisional；36指标/12类型/23794case token重算通过，未证明近零世界普遍获后验偏好，错误token反而更偏PE，勿采用过度解释。源66冻结已结束，历史wheel保存。新方法短评NEXT_CORE_OBSERVATION_REVIEW_20261028只提议固定S+源内可验证关系锚+完整陈述内部交互，先测错8/有据80.8；尚未实现/执行，非效果结果或顶会评审通过，不继续FIT。

## 2026-10-28 固定来源的约束交互状态（实现/试跑前）

按用户继续设计迭代，protocols/CONSTRAINT_INTERACTION_20261028.md冻结Problem Anchor：无金token标签、无LLMjudge、冻结8B，先测内部表示能否区分量值约束，而非再FIT/来源扰动。新增候选constraint_interaction.py及中央transfer constraint-pilot的constraint_pilot.py；保留原每token，B/I/C三mask分别为span双向/self独立/causal，前缀正常causal；仅C状态保留给下一span。源数值构造可验证两减法正反式原型，程序证书伪监督、非无任何监督，零参数优化。锚不含Y历史；Y/control数值仅作排除，主query保留完整前序步骤。

方法单轮6.65/10 REVISE，formal unavailable/same-family provisional：源内算式原型到自然文字/单位有域差，分数不作跨source校准；Fin53 decrease94仍方向错；LOO仅模板内检查。按反馈改逐源门槛、不可用处理和上下文防泄漏。初始sign-only RHS互换不匹配错8的量值问题，在无GPU结果时修正为源三元组同号不同结果RHS互换；排名前缀定向偏差也在试跑前改为排名后交替极值/独立order域hash。两次代码审阅发现d退化全轮中止及annotation校验误插score，均在GPU前修正，原失败测试不冒充通过。下一实测状态单独记，不把提案或tiny模型当真实性能。

## 2026-10-28 约束交互状态V1（完成，主门槛未过）

真实8B六源/12变体/688原与控制token，521backbone/16423处理token/50.151665秒、峰16368087552 bytes；全部特征/评分封存后才原silver标签。source68、wheel/installed集合及11模型hash一致，229源tests+9subtests/1skip44.01s，仓库外6目标测试通过。主B-I+证书方向Fin53错误8−数值94在decrease/increase下.620081/.774333；TAT3公式错误80.9−80.8为.146104，但总结两个单位-.033461/-.012279反向，因此primary gate=false，拒绝晋升。certificate LOO六源三个读出均1.0只限模板内，不能当金融AUROC。

独立3376检查、全部688token投影max3.11e-15、LOOmax2.44e-15、3seed tiny全mask/cached oracle通过。事后raw-causal删除检查六对比均正(.424567/1.023761/.098227/.036179等)，独立重算max1.78e-15；明确posthoc，不改主门槛、不记未见确认。实质发现：存在可读数值约束信号，但减掉independent state会在复述阶段抹掉信息；双向与减法的必要性未获证据。完整报告CONSTRAINT_INTERACTION_INTEGRITY_20261028，WARN/formal REVIEW_UNAVAILABLE/same-family provisional。两个0.8目标仍未达。

## 2026-10-28 事实约束B与续写一致性C四状态（实现，未跑结果）

protocols/FACT_COHERENCE_20261028.md，沿同两模块/中央constraint-pilot --factorial，零新增runner。固定source，源内证书生成前步正/负×结论正/负四世界，词汇/数值/token bags相同，B由Decimal运算证书、C由前后结果一致性标定。主raw-causal四Gaussian观测模型对C边缘化，binary为同证书折叠C的删除控制；不把C1续写连贯当B0事实有效。无外部FIT/神经权重更新，但明确source证书伪监督和局部moments估计，不声称零学习。TAT3补独立前错/后对纠错和前对/后错注入（8变体），先看最弱错误结论能否高于最强正确结论，未观测前不写结果。代码/测试/审阅启动，旧68源V1快照不改。

## 2026-10-28 四状态B/C模型V2（完成，拒绝）

真实8B 72proofs/288worlds/16变体1112token，631backbone/21343处理token/45.435620秒、峰16368087552 bytes，主CLI实际exit0。source certLOO fact Fin53 .90625/TAT3 .875，但自然总结prior80.9下正确80.8分87.671736反而高于错误80.9分76.963002，主worst-case margin=-23.441869；binary=-24.219459同失败。两单位同值来自因果结构数值位于单位前，不称单位语义不变性。主fact/coherence(实际incoherence)/binary三读出全部保留，无事后换主、无扫温度/层/外部FIT。V1的原始C探索不能当V2未见验证。

独立全部1112token三分数误差0.0、所有四state证书/原轴/silver标签/11模型SHA/source68与installed/wheel通过；审计oracle最初漏prefill分块已自修，失败log保留非项目bug。source231tests+9subtests/1skip44.64s；/tmp8目标13.29s。科学gate REJECT，integrity WARN/formal unavailable/same-family provisional，报告FACT_COHERENCE_INTEGRITY_20261028。两轮总结reviews/CONSTRAINT_STATE_RESULTS_20261028.md。没有新全金融AUROC/AP，目标仍未达，默认共享不变。

## 2026-10-28 来源字段—值—年份证书（完成，拒绝主候选）

protocols/SOURCE_BINDING_MATCH_20261028.md，中央constraint-pilot --binding，新增可复用binding_witnesses.py。
固定实际S，只选两个交换年份后均被明确单元格反驳的source事实，排除缺失/等值/字段歧义；
沿用四状态Gaussian以单独检查证书含义变化。六原检测失败例中3源可用，另外3源显式unavailable，
32对/128世界，3原答+8认证措辞/顺序/历史控制共11变体655token；282calls/11289处理token/31.946108s。
TAT50 source LOO=.75，但四主margin为-155.389956、+118.596581、-215.382846、-167.483488，3/4失败。
原始2020的两token风险171.739839/123.317325为正，但不能据此宣称修复或全数据提升。
现有正则在8probe当前年份覆盖为0（前文仍有匹配），不是宣称正则整体无效。
source234tests+9subtests/1skip43.10s；/tmp11目标20.46s；69文件/wheel/installed和11模型SHA完全一致。
独立所有分数/LOO/margin误差0；report SOURCE_BINDING_INTEGRITY_20261028，WARN/same-family/provisional，
正式reviewer unavailable，首个补充6.1-sol容量失败，fallback6-sol完成；冻结已结束，原69源归档保留。
科学主gate拒绝，不更换默认，不报新增金融AUROC/AP。下一项须直接检验来源事实对齐，不再扫truth Gaussian。

## 2026-10-28 来源约束的pre-year依赖路由阶段一（完成，局部门槛通过）

新增binding_dependency.py、binding_dependency_pilot.py，中央transfer binding-dependency；
不拟合truth方向，只用年份前token的全层/全头注意力对同unit前数值建依赖分布，再用完整source
关系证书求support/conflict/unknown质量，unknown不丢弃，其他token没有完整检测读出。
代码review修复抽样bank漏来源行、重复year列覆盖支持值两个真实blocker后才GPU；
一次并发修改代码触发full suite冻结保护(1fail/238pass)保留，最终固定源码重跑239pass/1skip/9subtests46.97s。
wheel/为修复前未运行快照，正式wheel_v2/installed_v2为71文件严格一致；/tmp16目标13.67s。
实际3可用/3未覆盖原source、3原答+8旧probe+16近远干扰=27query，152calls/5401处理token/27.208627s。
16干扰probe的32语义目标路由全部>.5，min=.511749；8真假对照主margin全正，min=.035058，
far严格胜uniform(0)/nearest(-1)，逆候选边8margin全负。保留数值质量，非注意力熵判别。
独立逐raw数值复算、32路由/8margin、全部6原轴/27特征/30编码输入、71文件/11模型sha通过，
审计BINDING_DEPENDENCY_INTEGRITY_20261028 WARN/same-family/provisional；该阶段科学局部gate通过。
仅单TAT50来源上的程序控制验证，绝不报新金融AUROC/AP、顶会通过或自然数据幻觉已解决。
下一阶段固定query、连同总计一致地交换来源两year数据列；必须做旧/新attention×旧/新证书2×2，
因外部证书单独改变已机械地使分数翻转，不将其误称内部source reasoning证据。

## 2026-10-28 固定query、相干来源整列交换阶段二（完成，局部门槛通过）

protocol BINDING_SOURCE_REASSIGN_20261028，中央binding-dependency --reference；保留全部30生成输入/27收集query，
只把TAT50的2019/2018完整数据列连同Non-current/total交换，header固定、prompt token bag相同、加总保持。
原/新source各32/32路由与8/8错绑对照通过；新source minweight=.516392，minmargin=.032784。
32固定语句位置的source-relative conflict变化方向全正确，min有向变化=.056764。
旧/新attention×旧/新source证书2×2全部保留；固定旧attention只改证书也32/32反转(min=.023498)，
故不能宣称attention内部判断了source真假。内部读依赖、显式来源证书验关系是当前职责分工。
真实新增152calls/5401处理token/25.184266s，242源码tests/1skip/9subtests46.22s，/tmp19目标16.81s。
71文件新快照与旧阶段一71文件不同hash，各自wheel/installed独立保留；11模型SHA及54原新rawscore等全部核验。
独立审计BINDING_SOURCE_REASSIGN_INTEGRITY_20261028 WARN/same-family/provisional；受控gate通过，非全金融指标。
code review修复unsealed evaluation gate依赖、reference score seal固定、preGPU固定query检查及tuple/list序列化差异，
全部在新GPU前修复，正式运行exit0。没有新真实AUROC/AP、没有晋升默认、没有顶会就绪声明。

事后原文检查发现新候选自己的自然FP：正确TAT3的2018应限定后面的-34.8，却被旧pre-year候选连到46，
产生conflict1；原始TAT50的2018还被合法差额2.0的unknown候选阻止scalar，2020仍unknown。
不能以32受控成功隐藏这些自然文本局限。下一轮仅修正关系方向、复用原有双向语法接口并缓存重放，
不改原受控结果、不把解析修复当神经创新，也不把三个原样例当完整评测。

## 2026-10-28 自然原文关系方向修正（完成，0新增模型调用）

protocol BINDING_NATURAL_PARSE_20261028；抽取既有literal_links共享双向句法接口，旧conflicts数值行为不变。
唯一显式语法按正确方向建边、竞争语法/heading未知，其余保持pre-year attention；显式分支标明parser来源。
统一binding-dependency新增互斥--observations缓存重放入口；输出保留原physical producer和当前scorer身份，
不复制算法/原始数据、不改旧制品，不把原152calls再报新增；两world各27物理状态重放，新增调用均0。
原TAT3的46@2019/-34.8@2018均支持、conflict0（旧2018错误冲突1）；TAT50原55.2@2018支持，
相干source交换后该关系冲突1；53.2@2020两world均unknown，TAT51 monthly无读出，未解决这些漏检。
两world的32路由/8对照数值保持、32cross-source位置保持。每world52显式句法边、48attention边、1unknown，
因此不能将自然样例修复归因于神经模块，不能算新完整AUROC/AP或晋升主线。
完整源码244pass/1skip/9subtests45.31s，干净安装/tmp30目标16.85s、71文件/wheel/installed集合精确。
独立54raw及score/edge、所有原轴、全部source/model11SHA、seals/缓存producer/cost0复算通过；
BINDING_NATURAL_PARSE_INTEGRITY_20261028 overall WARN仅开发范围/same-family/provisional，运行与审计闭合。
各实际commands/source/cost在outputs/binding_natural_parse_20261028/completion.json。
下一项先做完整真实人口的关系覆盖和FN/FP，补合法推导/单位/科目/主体条件与token修复责任；两个0.8目标仍未达。

## 2026-10-29 关系约束接入共享模型：完整金融评测（评测完成，审计中）

候选 experiments/research/relational_shared.py，I/O experiments/binding_transfer.py，
中央 transfer binding-full。保持原token/source gap/固定span/Beta(1,1)，对每个冲突关系使用
phi=1-sum(w*all_endpoints_clean)，count+clause-mask精确推断；保留每个重复数值提及的端点，
不预先指认错误端点、不平均token分数；support/unknown分支中性，跨span质量不重分配。
这是额外关系能量假设，不是已证明的事实后验；显式解析及内部attention路由不能等同于真假判断。
协议 experiments/protocols/FULL_BINDING_20261029.md 在运行前冻结，完整FinQA162/TATQA130、23794token，
保留FRED synthetic/silver、已暴露DEV的范围限定；无新FIT/参数搜索/裁判，只复用历史R4作为对照。
输出 outputs/full_binding_20261029/run，运行命令在plan.json，日志 ../run.log。
review FULL_BINDING_REVIEW_20261029 qualified GO/WARN，独立240组穷举最大误差1.78e-15；
源码258pass/1skip/9subtests60.93s；/tmp安装29pass20.51s；wheel/安装73文件集合及SHA完全一致。
首次并发修正期间测试1fail/257pass触发源码身份保护，保留/tmp/full_binding_source_tests_20261029.log，
正式固定源码重跑全通过；没有削弱校验。全量运行exit0，新增148条原答attention、1240calls/106979处理token，
收集+评分+结束校验106.612s、CUDA峰值16411080192bytes；起始hash/加载/构建/评价不计入该秒数。
当前shared FinQA AUROC/AP .728275/.257152，新候选 .728346/.256860；
TATQA shared .768769/.355865，新候选 .768206/.356156；全部1000次来源配对bootstrap区间跨0。
只8/162与12/130回答触发9/16关系因子，变化352/574token；top5% TP/FP/FN与shared完全相同。
20条变化回答中13条的约束端点均无银标签正token；共122个端点token中19正，804非端点token也随共享变化。
TAT16的`fiscal year2019 was260.8`被pre-year fallback误连此前212.1，制造冲突；
TAT83也把前句149495@2019错连到后句2018；Fin155真实被标正的year分数仍低于负标签value157。
原六例全部0因子/0变化；2020未知、科目/频率/合法计算等仍未解决。没有靠均值广播压掉token观测，
但错误关系约束可经共享先验改变非端点分数，保留个体delta不能自动修复错误归责。
历史frozen_prior AP .285470/.438633仍高于当前shared，不能声称默认全面最优。
结果详见reviews/FULL_BINDING_RESULTS_20261029.md与outputs/full_binding_20261029/run/evaluation.json。
benchmark_target_met=false，promoted=false；本轮候选拒绝晋升。

运行后审计已闭合：reviews/FULL_BINDING_INTEGRITY_20261029.{md,json}，overall WARN仅研究范围/语义边界，
非数值或制品不一致。fresh审计逐条重放292分数/289边/25因子、148attention与原轴，最大评分误差0；
另一个fresh复算独立NumPy验证五系统全部指标、1000次source-bootstrap，最大区间误差1.61e-16。
原CSV→292原答/来源/发布编辑跨度、73文件源码/wheel/installed、全部分数与缓存hash一致。
11个大模型文件为执行器前后rehash，审计未另行rehash或重跑8B，不能声称独立GPU复现。
初次失败日志只证明源码identity mismatch，并发修改原因是执行记录而非日志自身可证明的结论；
最终固定完整测试与安装测试通过且未放宽门禁。same-family/provisional，正式reviewer REVIEW_UNAVAILABLE。
completion.json保留evaluation_complete_integrity_pending历史状态，另audit_completion.json记录最终闭合，不覆盖旧制品。


### 2026-10-29 条件预测参照候选（运行中）

`outputs/conditional_source_20261029/`：完整292金融回答，公共API新full-vocab观测，无FIT。固定TV稳定化+已归属Vovk–Wang校准器+原共享状态内核；主AP、AUROC守门，原六错例。源码311pass/1skip/9subtests，隔离wheel同样通过。两轮方法审查6.90→7.40、REVISE；新代码审查无阻断。尚无效能结果，禁止晋升或写有效机制。主线实现和公式见docs/METHOD_CONDITIONAL.md。


### 2026-10-29 条件预测参照：完整检验失败

首轮并列数值问题中断不评分；修正版outputs/conditional_source_20261029_v2完整292答/23794token。条件模型FinQA .718312/.218039、TATQA .770173/.357203，均低于V3，四配对95CI负；rank-only亦输。拒绝晋升，不触发自然RAG新提取，不扫参数救援。详见experiments/reviews/CONDITIONAL_SOURCE_RESULTS_20261029.md。新8313calls/320997处理token；旧观测与基线逐位不变；隔离wheel312pass。完整性审计正在独立核验，结果-主张审查no/high。


### 2026-10-29 潜在进入/持续候选（评分中）

固定proposal与protocol：experiments/proposals/onset_continuation_20261029/PROPOSAL.md、outputs/onset_continuation_20261029/PROTOCOL.md。与V3同pi/c/span，入口p0、持续半source半absent；不是rank/TV救援，无新GPU。全部2992答统一主线评分后才评价。源码与隔离wheel322pass/1skip/9subtests，独立36精确积分与120旧路径逐位对照，无code blocker。效能未知，不晋升。


### 2026-10-29 进入/持续：完整五任务失败，审计闭合

outputs/onset_continuation_20261029：2992答/448202token、0新GPU。FinQA .769276/.286886、TATQA .821409/.416008，均低于V3；Summary/Data2txt回归守门失败，QA的.851473/.212407不构成双指标提升。23936旧分数数组逐位相同；独立数值审计通过，范围WARN。正式result-to-claim为no/high，不进行左边界发射救援。详见ONSET_CONTINUATION_RESULTS_20261029.md。下一新观测为固定后文的来源×回答条件交互，仅11例机制诊断，尚无结果。


### 2026-10-29 回溯条件交互：冻结机制门槛失败；三轮正式复审闭合

outputs/retrospective_repair_20261029：11答/33位置/72替换，83响应、1446calls/27735token。3/5错绑复合条件通过但3/6正确yearguards违反，advance=false，无数据集AUROC/AP宣称。新观测在Fin155和TAT97第一处有强source-positive金额贡献，尚不能区分错误与正常。冻结读出关闭，无candidateprior/聚合救援。完整性77文件、83轴、33数值复核通过；补齐指定5.6ultra正式路由，三个no/high、not_ready，路由更正保留历史。新source-compatibility目标contract正式复审可用于下一研究，不是新方法成功。完整汇总docs/ITERATION_REPORT_20261029.md，allraw/protocol/hash/seal保留，goal未完成。

### 2026-10-30 — individual token state under a shared regime

One fixed `token_state` candidate, source in myproject/sequence.py and the existing
SharedDetector/API/native pipeline. Fixed 1/4 deviation channel bounds context by
log3 while retaining own witness emission; no new source evidence or training.
Protocol TOKEN_STATE_20261030.md, plan proposals/token_state_20261030/EXPERIMENT_PLAN.md.
Output outputs/token_state_20261030. Code review fixed empty controls and saturated
contrast. Initial test attempt superseded for source changes; final validation and
full five-task evaluation subsequently completed; candidate rejected. Final source/
clean-installed suites each337passed/1skip/9subtests. See the completed result entry
below and reviews/TOKEN_STATE_RESULTS_20261030.md; no promotion.

### 2026-10-30 — fixed channel rejected; structural uncertainty follow-up

TOKEN_STATE full run complete: all five AUROC/AP point pairs decrease; both finance
paired intervals wholly negative. Isolated finance AUROC .480→.585 and .749→.834
(small24/19positive strata); TAT54 first subtoken recovered, TAT3 FP14→6, Fin53
wrong8million still missed. Complete report reviews/TOKEN_STATE_RESULTS_20261030.md.
V3 remains the retained finance candidate; no default/promotion changes.

Follow-up `adaptive_token` integrates equal-prior shared vs fixed-deviation joints,
with evidence-weighted structure posterior. Frozen equal-posterior deletion control.
Protocol ADAPTIVE_TOKEN_STATE_20261030.md; one mainline implementation, no FIT/LM.
Code/numeric checks and full five-task evaluation complete under outputs/adaptive_token_20261030.
Source and clean-installed suites each345passed/1skip/9subtests; second before-label gate
verified47872prior arrays over2992answers/448202tokens. Adaptive financial results
.793303/.304330 and .836417/.437438; both AUROC/AP paired CIs versus V3 wholly
negative. All five point pairs decline; candidate rejected, no default change.
Complete report reviews/ADAPTIVE_TOKEN_RESULTS_20261030.md; integrity review complete.
TOKEN_STATE_EXPERIMENT_AUDIT_20261030: deterministic checks PASS, overall WARN for
development/single-observer/silver scope and historical RAG producer identity.
Fresh requested reviewer route accepted; same-family/provisional, not venue certification.

### 2026-10-30 — source-repair compiler and information-budgeted revision

Implemented `myproject/proofs.py` and `revision.py`, shared public/API/native path,
and integrated `transfer qualify-proofs` before detector scoring. One-bit forward
KL budget per connected original-block/proof component; complete witness DNF with
conjunction across proof constraints. Standard optimizer is not claimed as novelty.
Fresh method/code review found and corrected assertion-scope, sign/magnitude,
rounding, and alternate-reference issues; V1 archived wheel and outputs retained.
Final source and clean-installed suites each424passed/1skip/9subtests; exact79file
source/wheel/installed identity,63loadedmodulepaths outside checkout.

Two full source-only qualifications completed at
`outputs/robust_source_revision_20261030` and
`outputs/robust_source_revision_v2_20261030`: each all2992answers,0incompatible.
V2 also0supported,2996unknownclaimdiagnostics; fourgroups all exit0,0newLMcalls,
no annotation parsing. Predeclared>=5realincompatible gate failed; B3NOTLAUNCHED.
No new AUROC/AP or source-binding detection gain. Both runs post-sealed in
bundle_seal.json; exact kernel retained opt-in, no default or baseline promotion.
Diagnosis: all2700nonfinance lack recognized table scopes; finance109responses have
somefacts but101still lack supported answer grammar. Narrow front-end is inadequate.
Finite-grammar route closed; no posthoc grammar/budget sweep. Four-world duplication
and cell-intervention-without-compatibility HMM rejected before model acquisition.
Actual result and follow-up qualification decision:
`reviews/ROBUST_SOURCE_REVISION_RESULTS_20261030.md`,
`proposals/robust_source_revision_20261030/PIPELINE_SUMMARY.md`.

### 2026-10-30 — frozen dependency assertion instrument: full finance completed

Protocol ASSERTION_PARSE_20261030.md; outputs/assertion_parse_20261030/measurement.
One pinned English-EWT UD2.5 parser/ufal.udpipe1.3.1.1, no hallucination labels,
FIT, LLM/NLI factual judge or new8Bobservercalls; external UD syntax supervision
explicitly disclosed. Pure reusable adapter experiments/research/assertions.py,
integrated `transfer qualify-assertions`; raw parses and original-token roles sealed.

Prelaunch code review found native Model lifetime SIGSEGV, failure retention and
subject-span gate bugs. Corrected before S1/S2, regression-tested, no control or
natural results used for these fixes. Original setup wheel/tests retained; final
validation_v2 source/clean-installed each454pass/1skip/9subtests,81code/resource
members agree,65installedmodulepaths. Doc witness rechecked by separate agent.

All292testanswers:FinQA162,TATQA130. No parse failures;33answers have39complete
syntactic subject/year/value candidates (FinQA18/20,TATQA15/19),2250role records.
G1simplecontrols1/4pass;4safetyguards pass;3harderframes pass only by abstention.
G2coverage passes, jointgate fails. Single source-derived noun phrase plus fixed
`was` creates a plural-agreement confound, disclosed without changing the panel.
No hard-conditioning/detector score/AUROC/AP, no default/baseline promotion.
Measurement phase20.56s; complete CLI29.14s; no numerical-stack rebuild.
TAT97knownbinding stillunresolved;Fin53 lacks unit-aware expression DAG;TAT54has
no numbers and getsno record. Candidate counts are NOT equivalent to prior proof
coverage or factual-correctness precision. Result ASSERTION_PARSE_RESULTS_20261030.md.

### 2026-10-30 — evaluation correction with unchanged assertion algorithm

V1synthetic noun/verb agreement confound was an executor control-design error.
Preserved allV1outputs; prospective ASSERTION_PARSE_V2_20261030.md and prelaunch
AMENDMENT clarify four distinct source-hash-selected fields, fixed singular NP
`the value for SOURCE_SUBJECT`,44controls. Same model and exactassertions.pyhash,
no parser/template/model sweep. Eval-only code diff reviewed before V2launch.
Final source/clean-installed each455pass/1skip/9subtests,81memberidentity.

outputs/assertion_parse_v2_20261030: all292natural records SHA256-identical V1,
33candidateanswers/39completeproposals,0parsefailures. Simple12/16frames and15/20
links,4wrongfull-rolelinks; harder10/12checks butonly5/24links and3wrong;guards16/16.
G1stillfails/G2passes,nohardconditioning/noAUROC/AP/nopromotion. cashprovidedby
operatingactivities field loses provided-by qualifier or splits underacl; other
three fields simpleframesallpass. This isolates lexical field-boundary loss without
claiming general parser impossibility. Actualcost20.98smeasurement/29.30sCLI.
natural_identity.json records mandatory exact292record comparison beforeinterpretation.
Unified result ASSERTION_PARSE_RESULTS_20261030.md retainsboth rounds and limits.

### 2026-10-31 — same frozen front end across all integrated evaluation datasets

Protocol ASSERTION_ALL_DATASETS_20261031.md; outputs/assertion_all_datasets_20261031.
Removed finance292/table-control restriction, added complete prepared-input path
with public observer tokenizer and shared sealed instrument reference. Adapter/model
unchanged. Read-only inventory verified fullHalluRAG1782 +ClaimDB1999anddatatypes.
Prelaunch code review fixed native finalseal identity +exacttokenizerfile-set
checks. Root460testsetup retained/superseded; finalvalidation_v2source/installed
each462pass/1skip/9subtests,81source/wheel/installedfiles,65cleanmoduleimports.

Sevenjobs all exit0,4CPUworkers,164.13swallaftervalidation. Full6773text population:
RAGTruth QA900/0completecandidateanswers,Summary900/3,Data2txt900/0;
FinQA162/18,TATQA130/15;HalluRAG1782/0;ClaimDBpublic1000/4,private999/4.
Total518125parsertokens/14151numericrecords/50completeproposalsin44texts;
0parserfailures,1999withouttextsource,2289withoutnumericrecordsallretained.
Financial292recordbytes matchpriorV2. Same44frame panel onlyfinancial instrument
quality,notnewper-datasetaccuracy;G1failedall,detector_scoring_permittedfalseall.
No labels/FIT/judge/new8Bcalls andNOnewAUROC/AP/defaultpromotion. Existing5task
V3/R4metrics reread withfilehashes asHISTORICALonly,notnewparserperformance.
HalluRAGall891unanswerable+3unalignablekept;ClaimDBstructure-only/notnative token.
RAGHalluidentityunresolved;HaluEval/RAGognizeTRAINstagedbutnotintegrated/run,
explicitly inventoried ratherthanclaimingall downloaded datasets evaluated.
Actualresult ALL_DATASET_ASSERTION_RESULTS_20261031.md; inventory alongsideit.

## 2026-10-31 — contextual borrowing and complete manuscript

Implemented matched_local and symmetric_local in the unique mainline sequence
kernel; direct full-gap API preserves original token axes without invented local
or route fields. Frozen MATCHED_LOCAL_20261031 protocol; full2992 historical
answers/448202tokens scored/evaluated and all old V3/R4/raw/axis arrays bitwise
verified. Matched AUROC lower on all5,3RAG AUROC/AP intervals negative; within-span
RAG pair ranking improves descriptively, fulltasktop5TP declines everywhere.
No candidate promotion. Four label-free probes1024spans:0violations.

Fresh complete HalluRAG1782TEST×11generators/162pairedprefixgroups collected from
actual retrieved contexts; native69483 validtokens,62649known projectedlabels.
V3.637840/.321886,matched.621546/.303947,symmetric.613170/.295931 AUROC/AP.
Released targetavailable891 vsunavailable891 V3.766930/.456050 vs.537424/.257369.
Sentence-derived automatic labels,unknowns6834retained,notnativehumangold.
Allvectors/metrics/sourcepaired1000seed17 intervals independently audited.

Source/eval extension fullsuite561pass1skip9subtests; cleanwheel83code/resource
hashes exact. Scores used frozen v1wheel; extended evaluator separately pinned
v2wheel. No numericalstack rebuild. Outputs: outputs/matched_local_20261031/.
Review scope warnings remain: development exposure,silver/automaticannotations,
oneobserver andhistoricalRAG produceridentitygap. Publication draft inpaper/;
proof/citation reviewscomplete, full numeric andtwo manuscriptrevisionrounds
inprogress. A completepaper is not a positiveICMLreadinessjudgment.


## 2026-10-31 — fixed-persistence candidate and HalluRAG all-seven controls completed

Frozen installed snapshot `outputs/fixed_persistence_20261031/validation_final/installed` stayed exact to85source/resource hashes before/after. Integrated CLI completed all2992 originalresponses/five tasks, with38896 baselineNPZ arrays byte-equal before labels, plus HalluRAG all1782responses/allsevenmethods with5346 old3-method predictionJSONs byte-equal beforelabels. The prospective pi-uniform,c=.5 candidate loses both pooled AUROC/AP on every population, with nominal1000seed17 source-paired intervals whollynegative; no candidatepromotion. V3 remains best amongseven pooledHallu controls (.637840/.321886); rawgap .552343/.248891, sourceNLL .605651/.243419. CandidateHallu .633501/.314116. ThreeRAG within-span ordering also falls; Hallu conditionalwithinspan increase rests on24mixedspans/458pairs and sentence-derivedpseudo-labels. Native finitebounds have zero observedviolations but lowcertificatecoverage; zero samplednegative candidate secants is not a theorem orfactuality signal. Complete results, exactcommands/configuration/input/source identities, actualtimings and retained launcherrors are in `outputs/fixed_persistence_20261031/{CPU_RESULTS.json,CPU_COMPLETION_RECEIPT.json,RUN_REPORT.md,run_*.sh,run_*.log}`. All data remain development-exposed; zero newmodel/judgecalls in this CPUstudy.

Provenance clarification for the preceding entry: the immutable installed85-module runtime stayed exact before/after; the checkout matched at launch but changed concurrently through another baseline task (new experiments.research/lumina.py and modified experiments/fullgap_dataset.py). These changes were outside the clean installed run paths and did not affect this study. The authoritative report is RUN_REPORT_v2.md; RUN_REPORT.md is retained as an earlier wording version. CPU_COMPLETION_RECEIPT.json records both identities and the actual drift.


## 2026-10-31 — completed ICML evidence expansion

- Dense pinned Qwen2.5-3B all six original populations: sourcegap seven + released-core LUMINA three,4774answers/14322densebackboneforwards. Primary: `outputs/lumina_observer_20261031/DENSE_RUN_COMPLETE.json`; actual command runner `run_dense_all6.sh`. Failed original cached-equivalence witness is preserved; repaired dense witness matches all six columns byte-for-byte. Cached same-observer archive retained separately, not a thirdobserver.
- Four fixed span-resolution deletions per population through integrated CLI, calling reusable `experiments/span_resolution.py` and `experiments/research/sequence_controls.py`: `outputs/span_resolution_20261031/RUN_COMPLETE.json`; actual launcher `outputs/span_resolution_fix_20261031/run_v7_all6.sh`. No modelcalls or label fitting. V3 meanAP increases5/6; Data tinynegative unresolved. LUMINA meanAUROC/AP rises6/6. Centering preserves local order while reducing both global points everywhere. MeanlocalAUROC=.5; Hallulocal evidence is sparse/automatic. No promoted detector.
- Fresh final integrity: `experiments/reviews/ICML_DENSE_FINAL_INTEGRITY_20261031.{md,json}` verifies all48packets, originalaxes, unknown-inclusive transforms, every original vector, points and source bootstrap examples. Numericalpass; scientificscopeWARN for exposure/labels/provenance. Final manuscript audits remain separate.
- Original-V3 finite negative influence: new14-position exact witness, independent8192copy/refreshsegmentation proof. `outputs/v3_exact_native_20261031/`; audit `paper/reviews/PROOF_AUDIT_V3_NATIVE_20261031*.json`. This is a model-mechanism result, not a new hallucination score or accuracy gain.
- v5primary87source release692full/31installed tests; v7CPUdiagnostic88source release724full/147installedtests. All earlier failed gates/adversarialchecks/canceled duplicate validation logs retained. Final assetgeneration is running; current PDF remains stale until rebuilt.

## 2026-11-02 — recovered R4/V3 performance evidence and fixed CPU rank fusion

Historical evidence consolidation now contains **271 unique result records**, without new detector scoring. The six primary tasks are original RAGTruth QA/Summary/Data2txt TEST, released FRED FinQA/TATQA TEST, and released HalluRAG answerable/target-available 891 responses. R4 point estimates reach AUROC0.8 on2/6, V3 on3/6; the descriptive per-task best envelope reaches4/6 and is not a tested detector. Exact metrics and the remaining Summary0.0377/Data2txt0.0017 gaps are in `docs/results/r4_v3_task_summary.json`. Recovered complete HalluRAG R4 TEST and its strata are separate from the old132-response VAL/DEV result; different execution/backend and observation-budget caveats remain explicit.

The user permits modular hybrids and task-specific configurations, retaining unsupervised detection and ground truth only for final evaluation after freezing methods, populations, parameters, thresholds and scores. Existing untouched annotated evaluation data are valid; previously inspected results remain exploratory. The proposal does not require a single probability model or renewed relation perturbation assumptions.

Implemented the CPU-only `transfer fuse-ranks` stage in the central `r4-experiment` entry point: orchestration in `experiments/rank_fusion.py`, reusable equal-midrank kernel in `experiments/research/score_fusion.py`, and21 passing targeted synthetic tests in `tests/test_rank_fusion.py`. Command `.venv/Scripts/python.exe -m pytest tests/test_rank_fusion.py -q -p no:cacheprovider` passed in10.42s; retained log `outputs/r4-v3-strata-20261102/rank-fusion-tests.log`. The score-only runner validates exact original token/response/source/observer identities against a frozen label-free population manifest, retains unknown-label tokens and seals the fixed equal-rank output. Synthetic checks cover independent rank-oracle agreement, ties/integer precision, axis and coverage rejection, annotation-read isolation and sealing. They do not establish complementarity, natural-data AUROC/AP, full-suite success or installed-package validation.

At this record, historical cache recovery and an adapter from those caches into the runner's score-only NPZ contract remain pending; no real fusion metrics have been computed and no new model observation has run. The host has RTX5060 Laptop8GB hardware; the active Python Torch build is CPU-only. Cache-backed CPU fusion is distinct from any later CUDA/model collection. New runtime/test files are post-recovery additions, outside the old88-file source receipt and its724/147 historical tests. `tools/render_performance_review.py` renders recovered summaries; source explanation for it and the new runtime/tests remains pending in `docs/source-reading/SOURCE_INVENTORY.json`.

## 2026-11-02: recovered caches and source-grounded error diagnosis

Recovered 2,992 paired R4/V3 responses / 448,202 original tokens and all V3 HalluRAG 1,782 responses / 69,483 valid tokens. Native/score/response/offset hashes and independent populations match. Old R4 HalluRAG arrays were deleted upstream; aggregate reports survive.

The fixed equal-midrank CPU control was sealed before labels, then evaluated on all five available paired tasks with 1,000 paired source bootstraps (seed17). Fusion AUROC QA .888133, Summary .769380, Data2txt .794619, FinQA .751483, TATQA .792889. Finance regressed against V3; the control was rejected and defaults remain unchanged. Reports: `docs/results/cpu_fusion_evidence.json`; actual outputs: `outputs/rank-fusion-cpu-20261102/`.

Current priority follows the user: concrete source-grounded FN/FP mechanisms before new method design. Full-five stage diagnostics use fixed task-wide top5% budgets, active-input versus unused-channel distinctions and span-level coverage. TATQA54/51 original-unit replays identify V3 context suppression; FinQA53/59 demonstrate different primitive-evidence and partial-span issues; TATQA3 supports legitimate-derivation FP. HalluRAG primary audit preserves unknowns until evaluation and distinguishes sentence-broadcast labels from human token spans. No fitting, new weight search or new LM calls. Public evidence: `docs/results/fnfp_mechanism_summary.json`, `fnfp_qualitative_evidence.json`, `hallurag_fnfp_evidence.json`.

Separate implementation validation: fresh installed wheel has exactly92 matching modules/resources, all3 added CLI helps pass, and57 focused synthetic tests pass outside the checkout. Receipt: `outputs/rank-fusion-wheel-final-20261102/receipt.json`. Historical full-suite receipts remain unchanged; scientific improvement and the all-task .8 goal remain unachieved.

## 2026-11-02: persistent mechanism goal and H002 observation qualification

The user requested an active, resumable research goal and mechanism-driven
iteration rather than parameter changes or repeated description of known errors.
`RESEARCH_BRIEF.md` and `research-state/RESUME.md` are the current handoff;
the immutable prior index retains31 reviewed experiment records. The local
`tools/research_memory.py` binds code/data/model/protocol/population/evaluation
identities, atomically reserves exact runs and reuses completed negative results.
Its22 synthetic checks passed. This is distinct from scientific novelty admission;
renaming a hypothesis or changing an output directory does not justify a new run.
External ARIS watchdog helpers were unavailable; no such service is claimed.

Dataset/observer contract audit completed without rescoring old methods. Common
RAG config/tokenizer hashes match finance; original RAG shards/backend remain
unidentified. Finance numeric density and scorer-unit length, Data2txt Python
literals, and Hallu whitespace/generator confounding are verified descriptors,
not causal explanations of AUROC. Evidence: `research-state/dataset_contracts.json`.

Selected H002 only for qualification: full-response automatic claim/slot proposal,
shared direct/blind/leaky observations, deterministic masked-claim blind requests,
literal quote checks, retained unknown/invalid/uncovered items. No scalar detector
or ensemble is defined. Exact text is aligned only when unique; bad explicit
offsets are not repaired. Original complete18-answer inputs are frozen separately
from evaluation materials. Protocol1536-token proposal/256-token per-observer caps
were fixed before inference, with all output-limit cases marked coverage-limited.

Fresh peer reviews found and resolved bad-JSON retention, exact-text alignment,
request-admission cache identity and mapping issues. Final78 synthetic tests passed;
then a fresh wheel was built and installed outside the checkout: exactly94 matching
modules/resources,78 installed tests passed and all4 relevant CLI helps passed.
Receipt: `outputs/semantic-qualification-wheel-20261102/receipt.json`;
wheel SHA256 `8ab21dfb4e859a6e0ad3940712c797e1fb0c8b44738324d6249c209b09e040b8`.
These checks establish engineering contracts, not natural-data efficacy or novelty.

The new resource-compatible observer is Qwen2.5-3B-Instruct, intended NF4/SDPA,
not a replay of historical BF16 Llama. All9 model files match official revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`. A separate CUDA environment is being
prepared; the original CPU environment and other applications are preserved.
The user explicitly selected idle local automatic execution. At this entry no
new model observations, scalar scores or AUROC were produced. First run must
pass runtime/resource admission and exact-run reservation; score/observation
artifacts are sealed before evaluation. Detailed gates and stop/pivot rules are
in `refine-logs/EXPERIMENT_PLAN.md`.

Runtime checkpoint19:39: the isolated CUDA environment finished exact-pin and
dependency checks. An actual tiny NF4 GPU kernel passed on RTX5060/SM12.0 with
zero language-model calls. Installed94-source wheel remained unchanged. Completed
operational identity in observation spec/model descriptorv2 with hardware, driver,
CUDA, dependency-lock and thread settings; v1 remained unclaimed/unrun and no
scientific parameters changed. A reviewed, task-specific local-idle process was
actually armed (PID31576 at11:39:30UTC), polls30s with a six-hour resource deadline,
and reads no evaluation material. Three fake early-exit controls passed (STOP,
timeout, missing frozen spec retained); they are engineering checks only.

Authoritative live state: `outputs/research-goal-20261102/autostart-state.json`.
At arming it waited for host RAM, with no natural LM outputs or new AUROC. A running
stage reports call count as unknown until completion; failure preserves partial
artifacts and cannot be reported as zero calls or semantic refutation. No automatic
rerun or precision/parameter downgrade. The16-answer/8-pair constructed panel is
also sealed and independently reviewed, but Q2 remains unrun until Q1 admission.


## 2026-11-02: H002 context-preserving natural qualification completed, frontend failed

The source-only prototype was cancelled unclaimed/unrun; the context-preserving
replacement retained all18 original inputs. Frozen95-file observer wheel SHA256
697d35b288d3503bbd684ddcc138a495570f8946b7faa631811af126c5e16571 passed exact installed
identity and6 interface smoke checks;131 source synthetic checks passed separately.
The real model loaded both shards, then completed18 proposal calls,4943 output
tokens and792.877s generation lifetime. All observations sealed;115 registered
artifacts were verified before CPU gold release. First CPU dispatch selected an
old installed CLI and stopped before review; explicit frozen-reader selection
corrected it without new model calls.

Run a5d336433cd1a04ce71962fb8500d4f9f0c8b0473a74f146e66302cbacb3e9a5 failed the
proposal gate:48 raw frames,8 valid empty-slot frames,40 invalid frames,4 envelope
failures and0 direct/blind/leaky calls. Not a fence or token-limit issue. Independent
predefined coordinate checks explained0/31 paired integer slot fields; role and
context/answer confusion also occurred. This does not test source-QA efficacy.
No scalar score/AUROC or promotion. Evidence docs/results/h002_proposal_qualification.json;
immutable observations, old95 reader and CPU review: outputs/h002-context-20261102/.

H003 is the next strong-baseline qualification: whole original answers, finite
next-token labels and all6 fixed label assignments, with raw/off-label/permutation
diagnostics. Direct verification and option-bias controls are established ideas;
no novelty, calibrated factuality or token detection result is claimed. New H003
checkout/backend work is a separate release and must not change archived95 bytes.


## 2026-11-03 mechanism continuation

H003–H007 and their independent CPU reviews completed. H004 typed-source
coverage is3natural/0constructed, H005 focus results are mixed, external supervised
NLI H006 retains table/arithmetic failures, and H007 fixed response-burden
interaction is−.057753: none promoted. Full reports are docs/results/h003*–h007*;
the gold-density unit oracle and suffix-removal exact reuse are diagnostics only.

H008 source-only full-name+EOS closed QA passed16/16, source/event flips8/8each,
32actualforwards4.1325s; strict toy grammar already permits deterministic answers.
H009 reused those32 observations with0newcalls: all4fixed readouts/nuisance arms
remain16/16, so additive invariance is verified without an accuracy increment.
H010 external frozen NER emitted719S/136R spans on18cases; semantic coverage unknown.

H011 single frozen masked-question bridge is now run954488248f0417aebefd288d07c8976b502860d1c397579b04af2fc5d7126283.
All48originals/96slots/48Q96completions/192logicalinstances retained; exacttemplate,
fullsurname+EOS, no new model or tuning. References stay outside observation;
CPU reader/evaluation pending. See research-state/RESUME.md for actual liveness,
immutable installed117 release and no-repeat rules.

H011 independent CPU review completed5c162fe8ade0a235a221a9c0ad309bb1f98a3a8cb79ba9a06e3887192f7ba3fb. Fixedbridge failed: b0/4/12=9/8/10 of16, whole48AUC.63715; no promotion or prompt rescue. H012 now only prepares natural span/candidate artifacts; no new QA or semantic coverage claim. See docs/results/h011_masked_reconstruction.md.

H015 two-input shared-prefix engineering passed16actualforwards. H016 common-prefix scope×readout paired diagnosis completed312forwards/86930input/0generated;compoundind/joint andatomicind/joint rolecorrect9/16,16/16,15/16,16/16. Eight exposed toy sources only, no natural promotion. See docs/results/h016_paired_scope.md and REMOTE_EXPERIMENT_HANDOFF.md. Local complete CPU suite hit180s handoff bound (incomplete);70 installed targeted checks passed; exact pushed HEAD LinuxCI remains required.


## H017 relation ownership: actual engineering runs (2026-11-03)

- M0 `80913833f6fd73a6a5ed210f5d8607c377257755857798afd79d3bebffb5ca7f`: source-only occurrence plumbing completed2700answers/450sources;5806Data2txtfields;QA/Summary semanticunknown retained. No detector result.
- M1 canary `5f7cf0ad2b984ce024a1be8060b9ca60294f51bd6fd7fbe545b32c36156a8e88`: three real source-distinct QA archives faithfully replayed; all5 wrong-contract controls rejected. Maxgroupmasserror0.00176084;0labels/LMforwards. Seals in outputs/h017-relation-20261103/m1-canary.
- Remaining897 `48f5ce11c3cd40e5685a6da81e5e7cd404c5c4c294ff32a341a29d27a85769e4` completed/sealed/finished from immutable130-file target; original3 reused. Together900uniqueQA/150sources, allqualified; fivebadcontrols each rejected900cases. M1_900_AGGREGATE.json retains both true run identities.
- Source/character/token alignment completed3+897 in two sealed runs (`7d065e2e…`, `8d6662cb…`):900original_axes_matched,124817answer tokens. Semantic ownership remains unknown; boundary overlaps retained. No labels/modelcalls/scores.
- Original observer assets independently verified9files/291BF16tensors. Exact252-token prompt16465 passed two-world early-stop prefix qualification (`1fdf3766…`,2model/0head) and native full32-layer/head/cache/no-op (`7f222ddf…`,4model/3head). Bulk/cache maxlogiterror5.34058e-5; same-value Q/K/V no-op bitwiseequal. Six engineering model calls total, no detection result.
- Scientific role/value Q/K/V assay `24dd6e0bd68475e1561dc4379c2e4e2655ad3999cd935c348af8441aea3bb8ce` completed48model/44head,847.83s. K-role main gate failed4/4; V-value directional control passed4/4. No high-p wrongR in the quartet. Instruments passed; finish=refuted for selectiveK mediation only. All units retained; no layer/head/prompt rescue or detector promotion.
- Natural temporal observation `31185253c6389623bea031a07db70d77043ee67315e9fe85ca426725dc826d32` completed1native full441-token original input+1head76rows,67.24s. All32layers/32heads/77receivers retained and AV/archival15 qualified. SourceBC becameRago, but observer prefersBC(.982174) overago(.000114724). FP32 producer metadata reproduced; strictauxiliaryF64 reference remainsinconclusive6/76, failure retained. Observation only, no new risk/AUROC.
- Reused oldBF16 readout chain: ago rawfullgaprank1165/point1492 ->R4prior116867/final114453; V3witness1165 ->final20219. Fixedcrossprecision budget failed44/76 despite matching axes, so no joint-causal bridge with M4FP32. This prepares a new relation-specific readout question, not contextclipping/max-weight tuning.

Use the shared transfer CLI: `inventory-relation-source`, `qualify-relation-cache`,
`qualify-relation-alignment`, `qualify-streamed-observer`, `qualify-streamed-head`,
then the separately claimed `observe-relation-paths` and `observe-natural-relation`. Each requires the exact
asset manifest, completed head qualification and signed scientific admission; its
`--help` lists path arguments. A completed engineering gate is reused, not rerun
when downstream analysis changes. See [actual report](../docs/results/h017_relation_binding.md),
[protocol](../docs/research/h017_address_payload_protocol.md),
[derivation](../docs/research/h017_path_derivation.md) and
[offline formula view](../docs/source-reading/H017_MATH.html).
