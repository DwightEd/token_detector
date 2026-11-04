# 最小闭合合同：同q/GQA head–carrier 关联擦除

**状态**：有限机制观测 conditional GO；standalone detector NO-GO。0新native forward、0GT、0新risk。2026-11-05 01:33:51修订，保留012503历史；本文是本次唯一实施范围，优先于旧formal的SEM-only退化段落。未来I02/新source世界均不进入本次实现。实际运行须独立code review和新memory claim。

## 固定输入与前门

- 数据仅既有QA source14375的fresh完整图00012/00013，P535，T196/147，nodes730/681；使用该次A/V，不混原incremental或replay的浮点状态。下载manifest和graph2资格SHA由spec硬绑定。
- source-only有限许可：action/object/duration/unit/condition/scope、原文span/SHA和carrier位置。候选为cook/simmer/grill/onion事实；重复simmer15–20只有完整条件等价才算EQUIV，不按数字相同强行通过。
- 实验pair为同token ID的两个**完整关系闭合carrier**，都在source且对原q可见。分SEM（不同动作/条件）与EQUIV（同一完整事实的明确重复）。不把layer0同ID、共享O或norm sham当语义负控。
- 两类至少各一对、F为全部合格pair且F≤8，才进入本次数值机制试验。否则输出完整资格/unknown理由、**0机制数值计算退出**；不补造事实、选有利pair或换数据。
- q取全部原chosen token_text含`[0-9]`且同句过去prefix已有`[A-Za-z]`的原决策。仅在**已消费的R prefix**中，以换行/`.!?`定句界；小数点也保守作界，不用右侧未来字符修补，fractional续token可能被排除，理由完整保留。空行首编号排除；这是检查target y_t，不是receiver token。它是已知整个原R后的**retrospective词法选域**，不是事前可部署selector，不能把y_t或未来内容当成q已知语义。原文本/ID/offset join不明则unknown。入选及所有原T排除理由先冻结，不看正误/effects/logit结果选q。
- 原q=P+t−1；同一q的全部32层、每KV组4个Qheads内全部6对，共48对固定h1<h2。不得跨KV组或跨q。泛型kernel可为2×K；**本实物合同K=2**。

## 唯一干预和读出

对单个head-pair/source-pair的原四cell A，保持所有其它cell不变。

\[
r=A1,\ c=A^T1,\ M=1^TA1,\quad B=rc^T/M.
\]

M=0原样返回，不除、不加epsilon。A的每个source位置都必须因果可见；不能让B填入future masked cell。原A可为0，此时χ不可定义但erasure仍有效，记录censored。

对2×2 A=[[a,b],[c,d]]：κ=(ad−bc)/M，A−B=κ[[1,−1],[−1,1]]。共享V时，erased−native pre-O两head为−κ(va−vb)、+κ(va−vb)。如O另行合格，residual差为κ(O2−O1)(va−vb)。没有W_O时只报pre-O，绝不伪造4096维消息。

**每个48×32 pair实验独立**；不相加重叠delta、不把多个实验拼成一份CF world。group-wide 4×K是另版方案，本次不做。

主读出为原四cell/B、row/col守恒误差、M/κ、I(head;carrier)、χ/undefined原因、pre-O delta及范数，全部q/layer/head/pair保留。没有W_O的M1中，范数明确为**各128维head delta及两head的2×128 concat范数**；不是4096维residual，也不能将两条head delta在128维相加（那恒为零）。用F64从所存F32计算是离线数学算术，不冒充原BF16 forward。

P=A/M，B/M为固定marginals的最大熵独立coupling；KL(P||B/M)=MI。0log0=0，不平滑。这些是standard性质，不主张新理论。若V相同，或在有O的工程测试中O1=O2，residual差为0；共同V平移不改差。这些只是数值负控。

纯kernel数值门在新实物计算前固定：native A为有限非负F32，V有限F32，转F64后计算；B有限非负，row/col绝对残差≤`64*K*u64*M`（u64=2^-53），M0无op精确返回；无零质量epsilon。2×2闭式delta与直接AV差比较用预设操作界`256*K*u64*sum_j((Aij+Bij)*abs(Vj))`逐坐标，零scale要求精确0。任意rank1输入的B/directAV只要求该数值门，不强求B位同一；闭式sameV/sharedO零与directAV舍入比较分开报告。这些是保守F64操作资格，不是语义或CUDA误差定理；失败不clip、不改门，先保存raw再停。

原token对齐的投影**本版只保留接口，不是必做score**：若W_O/finalnorm/unembedding参数尚未独立资格，明确unknown，不加载参数或计算投影。若后续另行冻结其参数/资源合同，只能作为标准centred-logit线性诊断，不能当CF logp/完整suffix效应或risk。本次无full-vocabulary head调用。

## 竞争预测与止损

| 解释 | 本次能检验的区分 | 结论边界 |
|---|---|---|
| coarse source/head mass | row/col均保持，纯边际模型无变化 | 非零消息只排除纯边际解释 |
| relation-context specialization | 在SEM/EQUIV全部固定pair、原数字q上比较全部层头的依赖及范数；不是只展示patch能改数字 | 若无EQUIV或同等norm解释，语义特异性不成立/unknown |
| 纯扰动幅度 | 若有O投影合同，用固定坐标循环移位1位的等范数delta作为方向sham | 只控幅度/方向，不能称语义负控 |
| source位置/RoPE/距离 | 同q避免query时刻差；记录source绝对位置/距离 | 未作保义位置控制即不能宣称因果绑定；positional机制也可能是合法实现 |
| 词面 | carrier ID相同、完整关系许可、EQUIV控制 | 不保证排尽上下文中的数字/动作词、语言难度 |

不从χ符号或MI大小推真伪：head没有甲/乙语义标签；任意head顺序不能决定truth。合法多对多和缺失证据都不制造排他cycle。当前不产生原R检测分数、AUROC、posterior或方法晋升。

## 最小预算、顺序与后续

1. **M0** 只source/原轴metadata资格并freeze F/Q、所有pair/q与计数。前端不合格则0机制计算退出。
2. **M1** 纯kernel/tiny synthetic、独立review后一次CPU实物：0native/0MLP/0full-head/0GT。tuple数`32*48*F*Q`，Q≤343且总数≤1,000,000；超预算停，不裁头/层/q。全部单独输出标量/索引，非全部大delta矩阵。禁止逐tuple复制2×128 F64 vectors（1M份已约1.91GiB）；以κ和原V文件/index/SHA、每layer/sourcepair/KV组复用的ΔV保留完整可重构表示，逐tuple完整保存标量/索引/各head及concat范数，不丢tuple。
3. CPU2线程、private host≤512MiB、staging≤16MiB、输出≤2GiB；C06全新目录、事前≥5GiB自由空间、600s单次上限。主图为独立`.npy`，A/V用只读mmap/逐head或选定row读取，不加载整A、不`np.load`整图字典。原NPZ只读取明确小成员token IDs/text/special mask等元数据，不打开其巨大attention/hidden成员；前端字符串/ID用已封存资格join。任何失败保留且无自动重试。尚无运行结果或实测时间承诺。
4. 数值负控包括random legal matrices、双边守恒、sameV/sharedO零、实际GQA/directAV delta、置换协变、M0/零cell、future拒绝和不叠加重叠pairs；不是测试照抄公式。

**具体最小后续干预，当前不授权**：若M1不是数值/范数伪象，在固定原R00012上最多新做两个source版本：完整事实保义换序、同action slots内交换完整duration/unit关系。先零模型核closure/条件/token编辑和原R/q不变，按fact provenance追踪carrier。总新native上限2，只用于区分固定slot与语义上下文预测，不重复仅看native似然差；做不到合法编辑则不加模型。这个位置/语义门没有通过前，只称计算机制pilot。

## prior与准入意见

旧P006、ownership X/E/XE和V donor重组已经存在；本次不重做其source null/suffix。限定旧源码审查未发现明确row+col双边保持的head×carrier独立化；genericE能表达B不等于旧实验已做该null。χ的加性边际消去/双线性恒等已有[2026 pairing-cycles论文](https://arxiv.org/html/2610.23248v1)；单pass图/归因已有[Unpack](https://arxiv.org/html/2605.23393v3)，不宣称首次。

**Conditional GO**：可实现/审查本纯kernel与source资格合同，并由root另claim一次已存图机制试验。**NO-GO**：standalone hallucination detector、原token风险方向、自然AUROC、把I02当已实现、追加native模型或旧H018符号救援。
