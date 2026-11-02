# 当前恢复：H006 诊断已完成，H007 新关系×回答负担控制正在准备

用户要求持续推进；六任务各 0.8 尚未达成。H006 全部预测和 CPU 评价已封存，结果 docs/results/h006_external_nli.{md,json}，F020。QA/Summary/Data/Fin/TAT 字符 AUROC .728655/.820705/.732662/.660139/.580005；Hallu 原生句子 .736111。它是使用外部监督训练的诊断基线，不是已采用无监督主方法或完整任务成绩。04表格交换仍反向，合法80.8推导被误判冲突而直接引用可识别；不能重跑旧面板或挑任务赢家。

当前 H007 为既定 NEXT_MECHANISM 第2比较的新控制：2关系×2词汇×2来源方向×2回答方向×3负担=48答，matched whole/首个自动单元=96逻辑case。来源始终含同样12个正确无关事实，回答增加0/4/12正确干扰句；目标句首部固定，比较关系分离和交互效应。资格代码和生成器正由各agent实现，root准备CPU reader/CLI/统一109文件wheel；尚未宣称冻结、模型调用或结果。重启前必须查最新spec/claim/PID，不能重复启动。

辅助7B公共量化参考仅在下载：outputs/larger-reference-readiness-20261103/receipt-v2.json，parent33404、02:40:10北京启动；不是模型已可运行或已有效。旧v1工程失败回执保留；v2严格计算剩余下载量+20GiB+512MiB空间，.part原子rename，无覆盖/无自动重试。当前主实验不依赖下载完成。

以下为此前恢复记录，当前状态由上面及最新运行工件覆盖：

# 当前恢复：H005 不晋升，H006 已封存，主线程整理结果并准备 H007

用户要求持续迭代，不因旧 goal 工具的 blocked 状态停止或重问许可。目标仍是六项主任务各 AUROC≥0.8；无监督约束、先冻结评分再评价、禁止 gold 训练/调参/选臂保持。已暴露面板永远是探索数据，不能重命名或重分组为新确认集。

H005 已完成且不晋升。固定自然18+构造16=34答，匹配整答34+精确单元104=138逻辑case、828逻辑请求，实际714前向、114精确复用、286.044077秒、752910输入token；无拒绝、unknown或空回答。观察 run `bedd217f55b341b2558de6f28820fd93dbb5f2b9ee02131c79bd0ad58f8f43be`，CPU run `747970d9d6aabf87643ad58c2f9d951c3d2c6019fddbd7911fa3c43579c6691c`。原观察104文件目标和单独105文件CPU目标保持，不因文档或评价器变更重跑。

完整结果见 [H005 报告](../docs/results/h005_focused_verdict.md) 和 JSON；事实 F016/F017。字符轴 whole→unit AUROC：QA .658217→.366792、Summary .708687→.438602、Data2txt .602205→.712755、FinQA .486554→.660139、TATQA .746642→.746642；HalluRAG 原生18句/5答 .611111→.625。构造16答两模式 AUROC 都为 .793651，null/False 配对仍反向。保留所有组与两个模式，不选任务赢家包络、不拟合阈值或权重。这些不是历史 BPE 指标，也不是六任务0.8证据。

事后字符单元上界审计仅覆盖13条已释放自然案例，5条原生句子案例不在此几何中。QA/Summary/Data2txt/FinQA/TATQA oracle 上界为 .995408/.999673/.971149/.842860/.856997。gold 单元正密度只用于上界审计，从未用于预测、分割、调参或方法选择。它说明当前排序还有余量，不能单凭它定位语义绑定、算术或标签偏差的具体原因。见 outputs/h005-20261103/unit-ceiling-audit.json；禁止把 oracle 当检测器或据此重跑旧范围读出。

H006 已在主线程实际完成观察及 CPU 封存，最终解释仍由 root 更新，不能停在旧 pending 文本。观察 run `77b26cefda05c920159b46866ee94a8aa49f901fb30072a9c0c5e9d19606b849`，CPU run `b484904456032737a9de6975d9a485caf9ef455ea14f4e23349b1c24c4782130`。34答/104单元全覆盖、0 unknown；144独立pair、18 batch前向、2.3771251秒。工件 outputs/h006-20261103/nli_v1 与 posthoc_review_v1，事实 F019 只记录封存和覆盖，不替代 root 的最终诊断。历史 worker45072、parent7692、启动18:38:45UTC；autostart-state 的旧 PID 与阶段不是活跃进程证明，禁止另启同一任务。

H006 是冻结外部 MNLI/FEVER-NLI/ANLI 监督的 NLI 诊断，并非已采用的无监督主方法或新颖机制。固定 DeBERTa revision `6f5cf0a2b59cabb106aca4c287eed12e357e90eb`、7文件哈希，标准 source-premise/exact-unit-hypothesis，512总token含3special，source窗预算509−hyp长度、overlap=min(64,budget//4)，max-entailment风险；没有原回答/QA上下文，不能当作与 Qwen 严格匹配的架构因果对照。root 的107文件 release wheel `c11e4c3b912d906febf3ee39137130e18aac2b5fb9c68bb933f3d95539851c4d`、commit `884d812` 已保留。此前两个13/512 token工程前向通过（F018），不能与真实诊断成绩混称。

主线程下一步整理 H006 最终结果，并推进此前 NEXT_MECHANISM 第2项已经计划的全新负担控制 H007。新构造材料来自 relation×lexicalization×source×response×burden 的全因子，不是重新扫描 H005 参数或提示词。root/其 agent 正在生成48原答与集成专用 whole/首单元 collector；此处不声明已冻结、已调用模型或已有结果。恢复前核查 root 最新 claim、真实PID/starttime/命令及状态，不重复启动任何等待或已认领工作。STOP 仅按用户明确停止请求创建。

H004 也已完成且不晋升：仅3条自然 Data2txt 来源实际类型转换，构造控制覆盖0，B=A、D=C；不能把 null 控制说成 typed 干预成功或失败。三自然错例仍高支持，规则未修复 null 顺序，构造三类 A/B11/16、C/D10/16。见 [H004 报告](../docs/results/h004_typed_factorial.md)、F015。H002自由提议前端失败/sourceQA零调用不等于核心重建失败；H003自然/构造结果 F013/F014 继续复用。

完整输入库存仍是 outputs/full-input-inventory-20261103/inventory.json：主组900/900/900/162/130/891=3883，Hallu pooled1782、unavailable891另保留；全任务模型运行未由本报告启动。H005/H006都是小型暴露面板，不是完整六任务验证。禁止 blind weighting、旧 prefix reset/repair 或有限证书换名复验。SummaC 等已建立粒度核验先例，效能、原生成因果与论文新颖性必须分别举证。

已知本地源快照：H005 observer `eabfffc`、CPU `98007cf`、H006 `884d812`，均未 push。本轮仅整理既有文档和状态，不修改任何活跃代码、环境或模型产物。状态文件写权限交回 root 后，由 root 追加 H006 最终结论与 H007 的真实运行状态。
