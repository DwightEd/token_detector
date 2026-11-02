# H011 CPU reader 独立静态复核

**PASS：本次源码快照未发现未闭合的实质阻碍。** 未运行新旧测试、tokenizer、模型或GPU；未读取实际H011/H008观察、construction truth或原任务gold。本结论是源码/协议复核，不是结果复算或效能评价。

复核文件：

- `experiments/masked_reconstruction_evaluation.py`，378行，SHA256 `e34ad7645fbff00336427c549e8770e6c64fc708fdf374e3b8d4be1ea14462cf`。
- `experiments/reconstruction_evaluation.py`，443行，SHA256 `f1007eee9ac124d0160e7892f3bcbae9e395cf1237743f5d842c8c9b23762b04`。共享扩展为显式profile/helpers参数，默认仍是原source-only observer；未改变token framing、raw验证、概率/风险或成本算式。作者另报原H008默认27项synthetic回归通过；审查者没有重跑。

## 身份与释放顺序

H011 root及96个request子树的全部成员、别名与seal key先被admit，然后才hash观察payload。初始root seal字节SHA被保留，后续_snapshot仍传同一masked profile，不能重新封存抹去历史。CPU pure `completion_metadata`重建所有96 execution的prefix/candidate/EOS、causal shift及键；每个raw逐token验证、joint和、shape/finite/资源记录/次数与成本重建，complete需exact相等。没有调用GPU backend或模型forward。

来源定义视图与masked视图必须对应同一48完整输入、source key和文本SHA；96槽不被筛掉。全部新观察重建通过后，才写CPU plan/protocol/source/freeze，且明确尚未访问design/construction/H008 control。freeze后先admit第三个已知control namespace与seal key，再检查H007 inference/construction双bundle的完整inventory，避免先hash一个命名空间、后才发现另一个已知树含越界项。

H008 CPU packet的外部预pin seal先核对，保留其全16queries、32raw候选、48cases、summary/costs/source/freeze/release。再由其plan中的独立observation_root建立旧observer的完整32request seal链，外部old-observer seal也必须一致。旧CPU的输入/协议/来源身份、成本和48readouts连接到旧observation complete；旧release须绑定同一H007 design/truth及seals。这里复用旧封存结果，没有重跑旧模型或拿当前代码SHA冒充旧reader身份。

旧/新model identity递归diff必须恰为三项：observer wheel、CPU module target、readout prefix，且每一项都等于外部protocol pin。其他字段——九份资产、revision、runtime、量化、预算、completion framing和资源政策——均不能变化。固定notes/hypothesis_id额外字段允许，其余协议默认值和类型严格保留。

## 全部读出与最终封存

source-only16Q的参考构造复用既有 `_references`，仍检查每query三个supported burden的一致性。masked48Q按(source SHA,event)精确连接：每query两R方向、一个burden、恰一个supported contribution；每burden16Q，每source/burden两事件对应两不同source人。Rswap相同query/messages的资格由固定kernel的完整plan重建保证，不靠模型分数或标签挑选query。

0/4/12三个burden分别保留16Q的raw joint logP/odds、greedy/tie/wrong、pool mass/underflow、姓名token数，以及完整8sourceflip+8eventflip。正interaction与两边正确反向分开；48cases/96slots及max-slot secondary指标保留，不把它们当48个独立重建。旧H008 control全体仍写出，不筛成功例，不挑最优burden，不拟合长度/温度/权重。

每一CPU输出先过精确stage inventory、readback equality再记录最初hash；最终seal仅写已记录hash。complete写入在revocation try内部。seal后再次检查初始H011观察root/children、输入/协议、源码与tokenizer/model guards、CPU freeze、H007 reference、H008 CPU与旧observer初始seals、结果hash；任何最后变化撤销complete/seal，失败材料保留，无自动重试。

保留的边界：模型资产后续guard对大权重使用size/mtime而非每次重hash；identity允许差异只按外部pin审计，不证明两个prefix科学等价；已知toy槽位、正确event锚点、双射与完整二候选池都不是自然前端或唯一编辑定位保证。H00832与H01196候选也不是同预算比较。静态PASS不替代root专用synthetic测试、安装验证及之后的封存结果评价。
