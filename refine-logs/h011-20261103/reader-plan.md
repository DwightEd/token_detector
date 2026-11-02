# H011 CPU reader：观察前固定计划

状态：仅接口/验证/分析计划；**等待 root 封存 observer wheel 117 后的代码写入信号**。此文件不读取实际 H011/H008 observations、H007 design/construction、模型资产或评价材料。此阶段不新增 production `.py`、不改 `run.py`、不改已冻结 kernel。

## 科学范围与固定 observable

本轮检验实际 R 经已知 PERSON 槽位去值化之后，能否桥接 H008 的 source-QA 能力，以及保留的 0/4/12 正确 filler 对该 observable 的影响。它不是自然槽位前端，不建立一般 `(S,R)` 的唯一编辑定位，不把 H007 A/B/C 与 H011 重建解释为同一仪器，也不以本轮选择 prompt、权重或最优 burden。

固定问题以 `experiments/research/masked_reconstruction.py::QUESTION_TEMPLATE` 的 exact bytes 为准；collector protocol 必须存其全文并严格复核。两姓名全部 literal occurrences 擦除，原 R 的 event / 标点 / 全部合法 suffix 与 exact S 保留。`[OTHER]` 不同 source person 仅属于这个二人二事件双射 toy 合同。

R direction 的两种原回答必须 collapse 为相同 query/messages；原槽值和 candidate index 仅供冻结后映射读出，不得进入 QA prefix。所有 48 originals / 96 slot occurrences 保留；预计 48 unique queries / 96 unique full-name+EOS completions / 192 logical requests。实际 forward、token、duration、memory 必须使用已封存 raw，不能将预计数冒称实效。

## 拟新增接口与最小共享扩展

收到 root 写码信号后才新增：

- `experiments/masked_reconstruction_evaluation.py`
- `tests/test_masked_reconstruction_evaluation.py`

拟接口：`evaluate(inputs, design, construction, observations, model_path, h008_review, output, protocol)`。CLI 由 root 接入；路径均为显式外部参数。

允许最小扩展现有 `reconstruction_evaluation._inventory/_snapshot/_validate(..., profile=...)`，默认 profile 仍为原 H008，原 32-completion 行为不变。H011 传固定 `masked_reconstruction_qualification` profile：其协议 loader、prepare、plan-builder、completion-builder、helper identities 和 expected completion count 一起固定，不能仅把目录计数从 32 改为 96 后放宽其余检查。尽量复用 completion metadata / raw validation / freeze / inventory / seal / result revocation，不复制 tokenizer/backend 或原始 grammar/构造真值实现。

## 顺序与路径隔离

1. 仅做路径/stat/完整嵌套 inventory admission：inference、construction、H011 observations、model、H008 CPU review、CPU output 必须是互不嵌套的独立树；协议在所有树外；禁止 symlink/别名/路径穿越/annotation 名。`design == inputs.parent / plan.json`。输出必须不存在。此处不读/hash H007 design/construction 或旧 CPU reference payload。
2. 对 H011 96 个 request 子树先枚举并检查所有成员名与所有 seal key，**再 hash 任何 archive payload**。固定 root metadata + 96 child 的 exact members；缺项/多项/错 key/别名均 pre-reference 拒绝。保留第一次 root seal 原始字节 SHA，不接受 reseal 抹除历史。
3. 只用 inference inputs、CPU fast tokenizer 与固定 kernel 重建 H011 全部 query、occurrence、candidate、prefix/candidate/EOS token IDs、causal shift、execution key 与完整 plan；模型资产只验身份、只加载 tokenizer，不加载 model，不调用 generator。逐 raw 检查每 token logit/logsumexp/logprob、joint sum、shape、finite、资源门、次数和全部成本；重建 complete 并 exact 比较归档。
4. 重建通过后写 CPU `plan/protocol/source_identity/freeze` 并 exact inventory/hash 再读回验证。计划明确 `reference_design_accessed=False / construction_truth_accessed=False / h008_control_accessed=False / model_calls=0`。冻结前不得打开或 hash construction 树、design、H007 inference seal（其中包含 design hash），也不得沿旧 CPU packet 读取其已释放 reference/readout。
5. 只有 `_check_cpu_freeze` 成功后，才 admission + seal snapshot H007 inference bundle、construction bundle 和完整 H008 CPU control packet；所有相关 inventory 通过后再逐 payload hash/read。随后解析 design/construction 与旧 CPU 数据。不执行任何模型调用、不做拟合或分支选择。
6. 写出所有结果、读回等值、冻结每个已写文件 hash；finalization 前后反复检查 observation 初始 root/child seals、输入/协议、source identities、model/tokenizer、CPU freeze、reference/control snapshots 和结果 hashes。最后写 complete、最后 seal 后**再次全部 unchanged**。失败只撤销本轮新建的 complete/seal 成功标志，其余失败证据保留，不自动重试。

## H008 control 的完整身份链

root 已提供以下外部预 pin，必须作为严格 protocol 字段：

- H008 CPU seal SHA256：`c62f8906d7cfb6986c4da99d29bb152d2f6ea7e85c5e02305356fb0d17dbd32b`
- H008 observation seal SHA256：`7558352ae7c3f5d46b4ab5b7bce72940262382806aaa4e9eaab5d81557e85a33`

新 CPU freeze 后，先验证旧 CPU exact inventory、初始 seal 字节、payload hashes 与原 CPU freeze；保留全部 16 queries、32 candidate observations、48 case readouts、原 summary/costs/source identity 和释放凭据。禁止只摘正确 query 或最优结果。

当前旧 CPU plan 本身没有 model_identity；因此须沿其 `observation_root` 指针检查旧 observation 独立路径、完整 nested inventory、预 pin 初始 root seal、child seals 与 model_identity，建立实际 control 身份链。不能由“同一个模型名”替代这一验证，也不能把新 reader 的源代码 hash 冒充旧 reader hash。只验证/复用完整旧封存结果，不重新运行或重算旧模型分数。

H008/H011 底层九份模型资产 SHA、revision、runtime、NF4/FP16、SDPA、completion framing、max input budget、资源门须一致；纯执行 archive-wheel / CPU target / readout-prefix 的允许差异必须逐字段列白名单并输出完整 diff，任何额外差异拒绝。H008 为 32 unique model completions，H011 为 96；明确保留实际成本，不能称同预算比较。旧 source-only query prefix 与新 masked-R prefix 的不同是预定 observable 差异，必须明示。

## Reference 复用与精确 join

使用既有 source-defined `_prepare(samples)` 生成一个纯结构视图：相同 48 originals、8 S、16 source/event query keys、96 exact slots。该过程只编译 grammar，不导出 source truth binding 当答案。完整 references 仍在 CPU freeze 后由已有 `_references(source_view, design, truth)` 验证：48 factor cells、全部 source/response/target/hash/filler/construction rule 和支持/矛盾配对均不复制新实现。

旧 `_references` 的 `counts==3` 对 source-only 16Q 正确，不能直接套到 masked 48Q。H011 通过精确 `(source_sha256, event)` 映射到已验证 source-defined reference，再按冻结 design 的 burden 归属 masked query。每个 burden 必须 16Q，每 masked query 恰有两 R directions 的 occurrence、一个 supported-reference contribution，故 `counts==1`；两 directions 的 query/messages 与 ordered candidates 必须 exact collapse。每 source/burden 两事件恰好各一次 reference 且为两个不同 source people；无多对多模糊 join、无按观测选 reference。

## 预定读出与全部结果保留

每 burden 0/4/12 **分别**保留全部 16Q 的 full-name+EOS joint log probabilities、canonical raw log-odds、reference margin、exact greedy/tie/wrong、两个候选的 conditional mass、完整 pool log mass / underflow、candidate token counts。各 burden 保留 8 sourceflip + 8 eventflip 配对及原 H008 同定义的 oriented interaction、正确反向与 ties；正 interaction 不等于两侧正确，三个 burden 不挑优。

保留所有 48 cases / 96 original slots 与 max-slot risk 的 secondary 描述性结果（含每 burden），复用共享 metrics；不得将两候选归一化质量或 max-slot risk 称校准 truth/joint factuality probability。附全旧 H008 16Q control 的封存读出与 costs，以及逐 source/event 可核对的对齐；不新增模型、温度、长度修正、阈值或权重。

需明示：8 sources / 4 scenarios 下强相关；48 original answers 不是 48 独立 source reconstructions；完整二候选池不等于自然开放答案空间；raw sequence likelihood 保留名字长度/频率偏好；toy parser 本可直接求真；自然 aliases/coreference/slot frontend 未资格。R direction collapse 后 readout 差来自原候选 value alignment，不能称模型已识别唯一修复位置。

输出 exact inventory 拟为 CPU freeze 四文件，加 `reference-release.json / h008-control.json / summary.json / queries.json / cases.json / candidate-observations.json / complete.json / seal.json`；字段定稿以实现时固定 protocol 为准，不根据实际结果改 arm。

## Synthetic CPU 测试范围

仅自包含 48-factor toy、fake tokenizer/backend、fake sealed H008 control；不读取实际观察、truth 或模型。

- 有效 96-request / 48Q 流程；所有 construction/design/旧 control 首次 read/hash 均在 CPU freeze 后；tokenizer-only guard 禁 model calls。
- 完整 archive plan/readout/raw/token/cost/ID/helper tamper，即使重封 root/child seal仍在 reference 前拒绝；bad nested name/seal-key 在任意 payload hash 前拒绝。
- 三 burden 每16Q、每query supported reference count1，复用 source-view count3、全部48 factor/hash/slot join；错 factor、label join、source/event mapping、缺项/多项/unknown不删例。
- Rswap collapse 与所有 96 occurrence 保留；主 8+8 flips 每 burden 的正交方向、exact ties；负 interaction / 正 interaction但错误排名的 toy 反例。
- 错 H008 CPU/observation预 pin、错模型/runtime/额外差异、旧控制删query/改cost、路径别名嵌套到reference、reseal during release均拒绝。
- 结尾成功标志 seal 后的变更会撤销本轮 complete/seal，保留 raw evidence；有限 pool/underflow 与长度偏好限制字段始终保留。

此计划不会触发任何实际 observation/evaluation。收到 root observer117封存信号后才开始生产 reader 代码。
