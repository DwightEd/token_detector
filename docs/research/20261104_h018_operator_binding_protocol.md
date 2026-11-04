# H018：原回答的来源写入如何进入 Norm＋MLP

本页保留冻结协议，并补充第一门的实际完成状态。**两条原回答的2次native、8次head已完成，407原token通过独立审计；尚无检测改善结论。** 第一门只覆盖来源13603，检验局部计算代理及支持型反例，不读取目标标注，不计算自然AUROC，不宣称六任务达到0.8。实际数值与限制见[H018结果](../results/h018_local_source_use.md)及[离线进展页](../source-reading/H018_PROGRESS.html)。

完整冻结约束见[第一门合同](../../refine-logs/operator-binding-20261104/ROOT_PHASE1_CONTRACT_20261104_221000.md)与[独立科学审查](../../refine-logs/operator-binding-20261104/REVIEW_ROOT_PHASE1_20261104_221642.md)。已完成的Data2txt900结果见[H017文本对照](../results/h017_nli_detection.md)，该结果不随本次提案改变。

## 1. 为什么需要这个有限门

既有概率差、来源写入和NLI头概率，没有区分“来源信息被取到”与“接收端如何使用它”。本次新增量只是在**固定原生层输入**下，来源attention写入进入本层post-attention RMSNorm＋MLP后的局部差分。它不预设attention只复制、MLP专管真值，也不把负作用定义成幻觉。

P006已经测过来源写入及其终端移除`H−source_write`；来源删除概率差也是既有对照。本次保留同次forward的P006读出、固定随机幅度对照及native NLL，隔离新增Norm＋MLP差分的作用，不把旧量改名为创新，不做分数加权、选符号或调阈值。

研究对象是冻结Llama观察器如何处理**已有回答**，不是让另一个模型重新回答问题。它也不是原GPT-4/Mistral生成错误的因果重放。

## 2. 原S/R、合法否定与 weak OR

两例共享完全相同的原prompt和S：`Tinker's Burgers`，source_id=`13603`。来源明确给出`WiFi='no'`、`street=True`、`lot=False`、`valet=False`、`garage=False`；`Music=None`保持unknown，缺失字段不能当False，属性也不能继承hours的星期条件。

| 原回答 | 生成器 | 原R token数T | 焦点及许可边界 |
| --- | --- | ---: | --- |
| 5682 | GPT-4 | 174 | `However, they do not have WiFi.`为来源许可的否定；局部`they`人工回指restaurant，不等于已建立通用回指前端 |
| 5684 | Mistral | 233 | `on the street or lot`保留歧义；`valet parking is not available`是来源许可的否定共事实 |

弱析取`Avail(street) ∨ Avail(lot)`在True/False赋值下成立；“两项都可供选择”的菜单读法才有不获许可的lot分支。**不把弱OR称为逻辑矛盾，不把5684整句直接赋为错误。** 两原R还改变了前文、位置和生成器，不能用任意跨R差异证明操作符因果。

原5682另有`there is no valet, garage, or lot parking`，它与5684共享字面`or lot`及同一来源lot=False，却有不同否定范围，可约束单纯词面解释。支持型反例、True共事实和unknown都保留，不能为了漂亮符号而删除。

## 3. 原概率轴与 F055 坐标

零模型主线分词门已完成：两例的P均为**803**，source mask均含**707**个token；5682的N=P+T=**977**，5684为**1036**。全部原R ID、offset、分段及历史集合槽6/8一致，没有重生成、联合重分词或删除难词。

原target采用0基编号，真实预测位置为：

$$
q_t=P+t-1,\qquad q_t\text{预测原 }y_t.
$$

读入R[t]后的receiver是P+t，预测下一token。因此5682在预测not65的q=867只读到`However, they do`；预测WiFi67的q=869只读到`…not have`。不能以后见WiFi要求这些位置必然选WiFi；允许提前规划，但不把它当实际读取未来词。完整WiFi否定首次闭合在读完WiFi后的q=870，属于离线读后诊断，不能回写为WiFi67的事前风险。

| 位置 | 当前实际绝对坐标 | 用途 |
| --- | ---: | --- |
| 来源valet=False / garage=False | 234 / 240 | 完整字段值闭合 |
| 来源street=True / lot=False | 245 / 250 | 完整字段值闭合 |
| 来源WiFi='no' | 277 | 包含字符串闭合边界；不以较早key状态冒充已读value |
| 5682预测not65 / WiFi67 | 867 / 869 | 真正原概率轴 |
| 5682读完WiFi67 | 870 | 完整否定关系的读后诊断 |
| 5682读完no-list中的lot102 / parking103 | 905 / 906 | 合法否定列表诊断 |
| 5684预测or65 / lot66 | 867 / 868 | 右属性未消费时不贴后见错误 |
| 5684读完lot66 / 完整valet否定74 | 869 / 877 | OR歧义与合法否定诊断 |

这些是**当前主线分词器的操作坐标**，不是历史Data2txt prompt已独立保存的证明。旧native里的`full_token_id`仍只是R；不能用R-only文件冒认完整prompt轴。来源闭合只保证信息可用，不证明已编码、使用或核验关系。

第一门的风险仍覆盖全部T个原预测行，统一使用q_t；字段坐标只用于身份核对和受限诊断。方法scope固定完整source_mask，不按人工WiFi/lot字段切来源，也不挑头、层或token。

## 4. 计算定义与有限理论

### 4.1 原生来源写入

令a为同次原生eager softmax attention，v为真实V，S为完整来源key集合。在每层、每个原预测receiver保留真实GQA对应关系，先分区投影再相加：

$$
m^S_{\ell,t}
=W^O_\ell\operatorname{concat}_{h}
\left(\sum_{j\in S}a_{\ell,h}(q_t,j)v_{\ell,h}(j)\right).
$$

不对来源attention重新归一化。source/history/other构成完整互斥分区；history包含R receiver的self key，不能全部称作earlier R。当前资格只接受无bias的原Llama，避免把同一个bias分配三遍。

真实native A、V与pre-O AV同时保留；不在SDPA返回None时静默换算法，也不把raw pre-RoPE QK直接当attention。

### 4.2 新量：固定背景中的局部非线性差分

设x为本层真实post-attention/pre-MLP残差，F为原模型的post-attention RMSNorm接MLP：

$$
F_\ell(x)=\operatorname{MLP}_\ell(\operatorname{RMSNorm}^{post}_\ell(x)),
\qquad
\delta_{\ell,t}=F_\ell(x_{\ell,t})-F_\ell(x_{\ell,t}-m^S_{\ell,t}).
$$

两个分支均用相同T行形状，且reference须对同次native norm/MLP输出通过固定数值门。它没有改来源、原历史、attention skip或其他层输入，因此是**held-native-input局部代理**。

在实数算术中，带正epsilon的RMSNorm与SiLU MLP光滑，微积分基本定理给出：

$$
\delta_{\ell,t}
=\int_0^1 J_{F_\ell}(x_{\ell,t}-\alpha m^S_{\ell,t})\,
m^S_{\ell,t}\,d\alpha.
$$

这只说明局部差分测量了F沿固定来源写入方向的响应；不证明它是真值、绑定专属量或有效幻觉特征。
程序实际只执行F的前向相减，不显式计算Jacobian或做数值积分；重放的是被研究模型的原模块，不另请问答模型解释向量。

### 4.3 全层固定累积与原token读出

32层按0→31顺序用FP64累加，再一次转换FP32：

$$
D_t=\operatorname{FP32}\left(\sum_{\ell=0}^{31}
\operatorname{FP64}(\delta_{\ell,t})\right),
\qquad H'_t=\operatorname{FP32}(H_t-D_t).
$$

H取真实native finalnorm的**输入**，不是已经归一化的last_hidden_state。完整N行finalnorm重放仅替换原预测行，其余行保持H及原布局；最后用相同T行、完整128256词表head评分，不分块执行head。

令g为原finalnorm＋完整head，定义：

$$
\ell_y(z)=z_y-\operatorname{LSE}(z),\qquad
e_t=\ell_{y_t}(g(H'_t))-\ell_{y_t}(g(H_t)),\qquad
r_t=\sigma(e_t).
$$

实数域中也有：

$$
e_t=-\int_0^1
\left\langle\nabla_H\ell_{y_t}(g(H_t-\beta D_t)),D_t\right\rangle d\beta.
$$

它只刻画固定原token的概率方向；r不是校准的幻觉概率。**32个held-input差分相加不是链式传播后的反事实forward**，没有补上后续层Jacobian，不能把它包装成完整来源移除或“MLP真值”。上述两式的δ、e指对应实数函数的恒等式，不是FP32实测值逐位相等；实际计算另按冻结数值资格检查，不能用恒等式冒充浮点误差证明。

同时保存分子与分母：

$$
e_t=(z^{cf}_{y_t}-z^{nat}_{y_t})
-\bigl(\operatorname{LSE}(z^{cf})-\operatorname{LSE}(z^{nat})\bigr).
$$

生产代码固定用FP32 `target_logit - torch.logsumexp(full_logits)`。不同减法次序的舍入单独保存；不把全词表分母变化称作独有的抑制机制。

## 5. 对照、资格与费用上界

| 对照/检查 | 固定做法 |
| --- | --- |
| 同T reference | F_T(x)对native RMSNorm输出/MLP输入及native MLP write；不能拿norm输出与未归一化x比较 |
| 独立m=0 | 真正执行counter路径；局部norm/MLP逐bit相同、delta为0；完整N finalnorm逐bit复现native |
| norm-sham | `matched_noise(m_source, seed=17+layer)`，逐receiver匹配整个O写入向量的L2范数；目标0则sham0，不用epsilon填补失败 |
| P006 direct-write | 来源写入按原FP32层序累积W_S，比较H−W_S；这是旧prior的同次对照，不是新量 |
| native NLL | 保存原token原生置信度，不拟合去相关或与候选融合 |
| 物理身份 | 原生A/V/GQA/pre-O AV、分区O、future attention=0、两次native残差相加逐bit；所有层与头完整保留 |

冻结上界为**2次native完整forward、8次完整head、256次额外局部MLP、256次对应post-attention RMSNorm、8次额外完整N finalnorm**；另计192次分区W_O请求和64次AV重建。native自身32×2层MLP与2次finalnorm属于完整forward。m=0已逐bit同一输入输出，复用native head，不假报额外head调用。

native AV/分区O使用`abs≤0.002+0.0001·abs(native)`；reference RMSNorm为`1e−6+1e−5·abs(native)`，reference MLP为`2e−5+2e−4·abs(native)`。D与独立CPU同序FP64累计逐bit相同；sham幅度门与所有finite/shape门均固定。e分解误差界为`32·float32_eps·(abs(z_cf)+abs(z_nat)+abs(LSE_cf)+abs(LSE_nat)+1)`，这是操作性资格界，不是真假阈值或浮点定理。任何失败均先保存已返回raw再停止，不调宽、不删token、不自动重跑。

原参数BF16，计算采用官方Llama的**native eager FP32**、FP32 RoPE，TF32/AMP/grad关闭；它不是历史SDPA/BF16逐bit复现。运行环境与权重按已核的Torch2.8.0+cu128、Transformers4.57.1、原SM120资格绑定，换设备须另行资格。GPU预算**6GiB**，host/GPU均保留**512MiB**；逐tensor mmap、单层物化，不量化、不截来源/R、不并发。按层顺序落盘，单次CPU传输最多8MiB、总临时缓冲约束32MiB；最多16行的logits落盘只改变存储，不改变完整head算术。

## 6. 调用链与运行入口

当前实现链为：

```text
experiments.run: transfer observe-operator-binding
  → operator_binding_observation.run → prepare / observe_case
  → 已核主线tokenizer轴与完整原S/P/R
  → StreamedObserver.hidden（官方原模型，eager FP32）
  → operator_capture.capture_local_effects
      → source_write_observe.partition_writes
      → operator_binding.local_effect / OrderedAccumulator
  → 完整N finalnorm / 完整T×词表head
  → operator_binding.full_vocab_endpoints / score_endpoints
  → myproject.measure.token_scores（原轴，不读GT）
```

主库的`extract/Observations/score`与V3状态内核保持独立；H018 orchestration在`experiments/`，候选计算在`experiments/research/`。局部4F在native MLP hook中、父layer卸权重前完成；递归hook有guard，callback完成逐层保存后才进行CPU累计参考工作，所有hook在finally清理。

以下仅展示参数结构。先核数据/源码/资源、登记独立run，采用全新输出目录；命令不授予自动重试或第二科学门：

```bash
# 两阶段均填写同一组已核路径；protocol及其绑定的科学/资源收据需一起转移。
token-experiment transfer observe-operator-binding \
  --prepare-only \
  --protocol /path/to/h018/protocol.json \
  --axes /path/to/h018/tokenizer-axes-v2 \
  --prepared /path/to/test_Data2txt \
  --frontend-contract /path/to/h018/FRONTEND_OBSERVATION_CONTRACT.json \
  --model /path/to/Meta-Llama-3.1-8B-Instruct \
  --asset-manifest /path/to/model-acquisition/manifest-v1.json \
  --asset-verification /path/to/model-acquisition/independent-verification-v1.json \
  --asset-state /path/to/model-acquisition/mirror-state-v1.json \
  --head-qualification /path/to/h017/m3-head \
  --capture-root /path/to/supervised_local_transport_20261008 \
  --device cuda:0 --output /path/to/new-h018-plan

# 准备封存、fresh review及实际资源通过，再单独claim观测阶段。
token-experiment transfer observe-operator-binding \
  --plan /path/to/new-h018-plan \
  --protocol /path/to/h018/protocol.json \
  --axes /path/to/h018/tokenizer-axes-v2 \
  --prepared /path/to/test_Data2txt \
  --frontend-contract /path/to/h018/FRONTEND_OBSERVATION_CONTRACT.json \
  --model /path/to/Meta-Llama-3.1-8B-Instruct \
  --asset-manifest /path/to/model-acquisition/manifest-v1.json \
  --asset-verification /path/to/model-acquisition/independent-verification-v1.json \
  --asset-state /path/to/model-acquisition/mirror-state-v1.json \
  --head-qualification /path/to/h017/m3-head \
  --capture-root /path/to/supervised_local_transport_20261008 \
  --device cuda:0 --output /path/to/new-h018-observations
```

## 7. 换机数据与 M0 复现要求

下表远端路径相对`/share/home/tm902089733300000/a903202310/lys/`。这是已知存储位置，不代表H018本地产物已上传；完整交接见[远端数据说明](../../REMOTE_EXPERIMENT_HANDOFF.md)。

| 材料 | 已知远端相对位置/当前用途 |
| --- | --- |
| prepared | `research/token_detector/outputs/ragtruth_full_shared_20261029/test_Data2txt/`，使用原`inputs.jsonl`、manifest；第一门不打开annotations |
| 5682 primitive R | `research/graph/outputs/native_support_ragtruth_all/source_first_v1/responses/000177/response.json` |
| 5684 primitive R | 同一根下`responses/000777/response.json` |
| 历史native | `research/token_detector/outputs/ragtruth_full_shared_20261029/native_Data2txt/`；当前M0实际用的是另行恢复/重封装的`native_Data2txt/observations/6.npz`与`8.npz`，须按下列SHA转移，不能猜测远端同名文件等价 |
| 原观察模型 | `models/Meta-Llama-3.1-8B-Instruct/`，完整四BF16 shard、index、config及tokenizer元数据；本地manifest固定原权重身份 |
| 原capture配置 | `research/graph/outputs/supervised_local_transport_20261008/`，仅复用配置/原资格身份，不把QA block15状态冒充这两Data2txt状态 |

必须另行转移完整`outputs/h018-operator-binding-20261104/tokenizer-axes-v2/`及frontend合同、模型收据、原head资格和协议引用的科学/资源收据。**`outputs/`被Git忽略，源码push不包含数据、模型、axes或本地运行证明。** 下列哈希公开身份，不是可下载附件：

| 必需身份材料 | SHA256 |
| --- | --- |
| prepared `inputs.jsonl` | `050c79f2711aa292845f27ed4f3ae8b68f0d263bc9d29678486f7e8fb6477521` |
| prepared `manifest.json` | `d1f0ba0658222556e2542e145da84538d14a9fa5b230dd9bb8fa34b830944b80` |
| 5682 primitive R | `bdf8ab1d987061c6063c94658a898e817f8f922381ce363f09e76e6b2f133f56` |
| 5684 primitive R | `ad2068e4d503983238f6a2eef4725fe98361e46c39f0c0af1f56f3b3cdde260c` |
| native `6.npz` / `8.npz` | `e745bab437e84812570478c870f8fa43a2c3e7b6b8b72c5f382402c7a9c23539` / `934a0fe4d0ca2221620592148b33ff3785dd7c8f6d2a760da0409427527de2d4` |
| frontend合同JSON | `3d165b58ee4415b007e4edfdc270bce2d7cf5d72f756c01e646d817e08d5c232` |
| M0 `axes-5682.npz` | `9c0b4e33b682c625a3101a12148de00c46f7540b97217affb30de9614e001f31` |
| M0 `axes-5684.npz` | `6a3d9d4212a21b1a596252dcbec12a73e102ddc7070810fd0ac3a4f3714668ae` |
| M0 `case-5682.json` | `c8f156f246a6f39ed04686f29879f6046acfd4b7b84aef7b958ef0c6279e58ac` |
| M0 `case-5684.json` | `8ae52886610cafbef70ae60b741435f61fd670f8b14184a4353f84f226f3e036` |
| M0 `plan.json` | `4e607242e40aecd7cd2a308e2f70c0791165d32de7d07d1a515dfc58414bbe12` |
| M0 `summary.json` | `744dba08037f9eac35d76878d5665bd669aa4ff736d4104ccc39d434bc454c3b` |
| M0 `seal.json` | `8d3002e70eaeb9265db9a63ee689da8f2a608bdb5848be7faf74e8fbd7963fb1` |

M0 helper是当前两条原R的有限坐标准备，不是已部署的全900前端。若新机缺axes，需原prepared、这两份primitive/native身份材料、相同tokenizer/config元数据，调用主线`encode_source/encode_response/answer_axis`及`relation_alignment`的offset/映射逻辑，独立编码prompt和R，重建source mask、prechoice行与跨边界字段锚，再单独冻结/qualify新的坐标计划。不能直接用旧R-only native推断P，也不能用NLI tokenizer坐标替代Llama坐标。

现有frontend/M0记录含原机绝对路径；CLI不会自动解析远端历史挂载。换机须显式建立新执行路径映射/本地契约并重新资格，保留原SHA与原seal，不直接修改已封存文件来“适配路径”。

## 8. 目前完成到哪里

F055的M0为0模型：修复局部answer_index默认0与历史集合槽6/8的检查错误，原失败保留；成功v2完成4次字符串编码，连同首次失败累计6次，未加载模型权重payload或读取标注。只证明原R及当前坐标身份。

最终冻结源为`3a53a195807f3d016df42b96a6177066f9f3bb76`，149个package成员的checkout/wheel/外置target SHA完全相同；wheel SHA为`0323dffa8d57bd9df47c0b546dd6ff6fd8340d24e802e8a0eb986d62baf0c4a3`。新增kernel/capture/orchestration的外置联合检查为**37项通过，pytest 21.86秒**。完整CPU套件达到300秒上限，明确记录为`incomplete_timeout`，不能称全套通过；当前checkpoint CI待push后核对，历史CI不代替。本地证明`INSTALLED_V3_TARGETED_VALIDATION.json`、`INSTALLED_V3_FULL_VALIDATION.json`及外置`snapshot-proof.json`未随Git发布。

首次metadata计划缺config参数，随后观测进程在CUDA尚未初始化时reset计数器失败，两次均发生于0 model/0 head/0 GT；失败保留，参数及初始化修复后另claim。实际run `841f447909662ec142e56058162e243ce7e9810baa9a96012815aaf0b2ee4426`已exit0，内部153.33秒、launcher185.22秒，严格执行2native/8head/256extraMLP预算。原raw与seal位于本地`model-observation-v3/`，不随Git发布。

独立审计run `0fcd8affd06146c171105f84840d30752a3facd43e64cc0fecd5906f6edf8fd9`已完成：407原token、64层、8完整词表分支、原轴及主risk join通过，AV最大绝对差约5.09e−6，GPU FP32 logp对CPUF64 oracle最大差约2.04e−6。oracle只审数值，不重写分数，不证明语义。七固定点中not65符号数值未定，合法not73的主量与sham同向相近，lot66仍weak-OR unknown；没有错误关系对照，没有新AUROC或检测改善证据。

第一门只许可报告这个局部代理的计算身份与固定行为。第二stage的owner/布尔交换尚在独立proposal审查，未准入、未运行；完整900效果评价也未准入。保留全部结果，不换号、换层、改scope或用NLI/格式改写救援。六任务各0.8目标仍未完成。
