# H018：来源写入经过 Norm＋MLP 后，原词概率怎样变化

**两条原回答的有限实验已完成，全部407个原token通过独立数值审计；尚无新检测效果或AUROC。** 新增读出可以忠实计算，但合法否定处的变化与随机等幅对照相近，尚未建立关系特异性。没有错误关系对照，不能声称区分正确与错误，更不能宣称六任务各达到0.8。

[离线进展页](../source-reading/H018_PROGRESS.html) · [完整协议与代码调用链](../research/20261104_h018_operator_binding_protocol.md) · [结构化结果与SHA索引](h018_local_source_use_summary.json) · [旧完整900检测结果](h017_nli_detection.md)

## 实际研究对象与时刻

来源13603明确记录`WiFi='no'`、`street=True`、`lot=False`、`valet=False`、`garage=False`。研究保留原S、原prompt和完整原R，不重写回答、不追加问答。5682由GPT-4生成，T=174；5684由Mistral生成，T=233。当前Llama tokenizer下两例P=803、source mask为707个token，N分别977与1036。这是当前操作轴，不补证历史Data2txt完整prompt身份。

原target采用0基编号：

$$
q_t=P+t-1\quad\text{预测原 }y_t;\qquad P+t\quad\text{已经读入 }y_t.
$$

5682预测not65时只见`However, they do`，WiFi尚未出现；不能后见地要求此刻已验证WiFi属性。预测WiFi67仍未读入该词，完整否定首次闭合在q=870。5684预测not73时已有`valet parking is`，属性已明确。两原R的前文、位置与生成器不同，不能将跨R差异直接归因于操作符。

`street or lot`保留**weak-OR unknown**：street=True使弱析取成立；只有把表达理解成“两项都能选择”才涉及不获许可的lot选项。不把它当已确认错误，也不将整句赋假。`no valet, garage, or lot parking`的否定列表、合法valet否定、Outdoor/TakeOut/Groups真共事实均保留。

## 本轮新增什么，怎样读出

令$m^S_{\ell,t}$为真实native attention通过真实V、GQA及$W^O$形成的**整个source mask写入**；不按人工字段切来源、不重新归一化、不选头层。令$x_{\ell,t}$为真实post-attention残差，$F_\ell$为本层原post-attention RMSNorm接MLP：

$$
\delta_{\ell,t}=F_\ell(x_{\ell,t})-F_\ell(x_{\ell,t}-m^S_{\ell,t}),\qquad
D_t=\operatorname{FP32}\!\left(\sum_{\ell=0}^{31}\operatorname{FP64}(\delta_{\ell,t})\right).
$$

从native finalnorm输入$H$的原prechoice行减去$D_t$，保留完整N布局；重放原finalnorm和完整128256词表head：

$$
\ell_y=z_y-\operatorname{LSE}(z),\qquad
e_t=\ell_{y_t}(g(H_t-D_t))-\ell_{y_t}(g(H_t)),\qquad r_t=\sigma(e_t).
$$

正e表示该局部代理的移除提高了原token概率；负e表示降低。**32层held-native-input差分的求和不是向后续层传播的反事实世界**，没有后续层Jacobian；它不等于移除全部source信息，也不等于“MLP判断真假”。本轮新信息仅是这项Norm＋MLP局部差分。

同次forward保留P006旧对照：按层序FP32累计source write后从H直接减去；另有seed=`17+layer`、逐receiver等完整向量L2幅度的sham，以及native NLL。没有融合、调权、拟合符号或改阈值。

## 七个预先固定点：全部报告

下表均为封存原值的显示舍入。P006和sham列也是各自移除后的logp增量e；risk仅是原sigmoid读出，**不是校准的幻觉概率，0.5没有被设为分类阈值**。

| 原例/target | 原token及语境 | native NLL | 主e | 主risk | P006 e | sham e |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 5682/65 | not，谓词WiFi未出现 | 0.00007820 | −0.00011826 | 0.499970 | −0.00003624 | +0.00001335 |
| 5682/67 | WiFi | 2.896446 | −0.851917 | 0.299031 | −2.106402 | +0.066797 |
| 5682/102 | lot，no…or lot | 0.328817 | −0.075052 | 0.481246 | −3.007704 | −0.196289 |
| 5684/66 | lot，street or lot；unknown | 5.887210 | −0.184692 | 0.453958 | −1.908296 | −0.159100 |
| 5684/73 | not，valet parking is… | 0.008911 | −0.005198 | 0.498701 | −0.007582 | −0.005989 |
| 5684/87 | available，outdoor seating | 0.032368 | −0.001354 | 0.499661 | −0.007097 | −0.006443 |
| 5684/105 | available，takeout/groups | 0.130722 | −0.022947 | 0.494263 | +0.003479 | +0.004822 |

5682/not65的$|e|$小于预先固定的两端点oracle预算和0.00045002，必须标记**numerically_unresolved**；不能仅凭打印出的负号断言可靠抑制。其native条件概率约0.999922，但这只是当前前缀下预测not的概率，不能当WiFi关系已核验。

5682/WiFi67的主e与P006均为负，首先显示来源依赖的词选择；没有单独证明极性绑定。合法的`valet ... not73`原概率约0.991128，主e为小负值，但sham同为相近负值，缺少关系特异性。Outdoor共事实也不能排除一般向量移除效应。

TakeOut/Groups的available105主e为负，P006及sham为正，说明新量与旧直接写入读出并非同一个数值；一个支持点的差异不能证明独有语义核验或总体检测优势。弱OR中的lot66仍为unknown，其原概率约0.002775，不能当作高置信错误或FN。

**当前只通过计算门。** 固定支持点未给出清晰的正向反例，但没有错误关系对照，且词面、一般向量扰动、来源归属等混杂未排除。原参数、source scope、全部层头及分数保持冻结；下一stage的owner/布尔交换还在独立proposal审查，未准入、未运行，不预填结果。

## 两条真实成立的读出约束

这是自行推导的实数归一化概率恒等式，不是引用来的真假理论。设$p_n$、$p_c$分别为原路径与该局部代理路径对同一原token的概率：

$$
e=\log\frac{p_c}{p_n}\le-\log p_n=\operatorname{NLL}_{native},\qquad
\sigma(e)=\frac{p_c}{p_c+p_n}\le\frac{1}{1+p_n}.
$$

因为$p_c\le1$，高native概率会压低正e的可用幅度；当$p_n$接近1，sigmoid正向上界接近0.5。这解释一种**读出尺度限制**，不证明检测无效，也不产生AUROC上界：该上界随token变化，精确单调sigmoid本身保持排序。它不是幻觉后验或校准概率。

代码采用FP32的`z−torch.logsumexp`，上述实数恒等式不充当浮点严格定理，不据此clamp、改分数或放宽数值门。not65的不确定性仍按已冻结oracle预算保留。

## 审计、费用与失败保留

原生产者run `841f447909662ec142e56058162e243ce7e9810baa9a96012815aaf0b2ee4426`已exit0、封存，登记为`supported_scope`。独立审计run `0fcd8affd06146c171105f84840d30752a3facd43e64cc0fecd5906f6edf8fd9`也已exit0，407原token、64层记录、8个完整词表分支全部通过：

- ordered FP64 D、一次FP32转换及P006 FP32累计逐bit一致；full-N m0 finalnorm输入/输出逐bit一致，其他路径只改变原prechoice行。
- 全部物理头native A×V、分区、残差、reference重放及sham幅度门通过；future attention非零坐标为0。AV最大绝对差为$5.09264\times10^{-6}$。
- 完整词表FP32 logp对CPU FP64 oracle最大绝对差为$2.04294\times10^{-6}$，最大误差/冻结预算比小于0.00942；oracle不产生方法分数。原预测分数、token及字符轴完整join。

| 项目 | 实际成本/结果 |
| --- | --- |
| 完整native / 完整head | 2 / 8 |
| 额外局部MLP / 对应Norm / full-N finalnorm | 256 / 256 / 8 |
| 分区W_O / AV重建 | 192 / 64 |
| 生产者内部 / launcher总时长 | 153.33秒 / 185.22秒 |
| 独立CPU审计内部时长 | 56.06秒，0新模型 |
| PyTorch peak allocated / reserved | 3,327,624,704 / 4,460,642,304字节，约3.3276 / 4.4606 GB |
| 单次host传输缓冲 / 本轮host传输payload | 8MiB / 9,690,132,176字节 |
| 保护线与预算 | host/GPU均512MiB reserve；GPU 6GiB预算 |
| 目标标注读取 / 新AUROC | 0 / 无 |

原Llama机制研究累计62次接口（2次早停、60次完整native）和61次head；旧DeBERTa NLI的18163次forward单独记账。额外MLP/head不是零成本，也不改称完整model call。

两次真正进入模型前的工程故障保留：metadata计划漏`capture_identity`的config参数；随后观测在CUDA尚未初始化时reset计数器报`Invalid device argument`。两次均0 model/0 head/0 GT，修复只补参数和显式设备初始化，另claim后才执行上述唯一2native预算，没有偷偷重跑已完成原模型。

当前源冻结为`3a53a195807f3d016df42b96a6177066f9f3bb76`，149个package成员在checkout/wheel/外置target逐SHA相同，wheel SHA `0323dffa8d57bd9df47c0b546dd6ff6fd8340d24e802e8a0eb986d62baf0c4a3`。外置新增链路**37项通过，pytest 21.86秒**；完整CPU套件达到300秒上限，状态为**incomplete_timeout**，不能称全部测试通过。本次checkpoint的确切GitHub CI仍待push后核对，历史CI不能代替。

本次采用原BF16权重、官方Llama的native eager FP32计算；不是历史BF16/SDPA逐位重放，更不是GPT-4/Mistral原生成器因果证据。独立审计检验保存中间态、操作性误差门与原轴身份，不重跑模型数学，不证明语义真值。

## 公开摘要与本地大文件的边界

[本页结构化摘要](h018_local_source_use_summary.json)包含7个固定点的原精度值、成本、失败、版本与证据SHA，可直接在GitHub阅读。[协议](../research/20261104_h018_operator_binding_protocol.md)列出换机所需的prepared/primitive/native/model远端位置和M0身份材料。

`outputs/h018-operator-binding-20261104/model-observation-v3/`、`phase1-independent-audit-v1/`及其原始大文件均被Git忽略；公开SHA是本地证据身份，**不是GitHub附件或下载链接**。源码push不包含约GB级raw、模型或完整私有挂载路径，不提供不存在的公开artifact链接。无需下载raw即可读本页与摘要；复算则须按协议另行转移已封存资料。
