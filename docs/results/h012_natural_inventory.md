# H012：自然擦值接口库存完成，QA 与语义资格仍未知

18 个自然原答、719 个来源 NER occurrences 和 136 个原答 NER occurrences 全部保留。130 个目标可按固定规则编译成擦值 artifact，6 个因 mask spans 重叠保留为 unknown。**这不是 130 个合格 QA，更不是自然检测覆盖率**：QA 准入为 0，所有语义状态及未提案区域仍未知，没有任何 risk=0 的默认填充。

H011 在已知 toy slots 上的直接 masked-QA 桥已失败；本库存没有修复或验证它。H012 不产生可执行 question/system/messages，不运行 QA，也不据候选池存在、大小或原值在池中来判断真假。

| 完整库存 | 数量 |
|---|---:|
| 原答 | 18 |
| 来源 occurrences | 719 |
| 原答目标 occurrences | 136 |
| source pools | 148 |
| compiled artifacts | 130 |
| unknown artifacts：overlapping_mask_spans | 6 |
| 若未来评分的潜在 completion pairs | 680 |
| 本轮 QA / 模型 / tokenizer / NER 调用 | 0 / 0 / 0 / 0 |

680 只是按固定池计算的潜在配对库存，未获评分许可，也不是已完成 forward。全部 136 个目标的 native NER label 计数为 DATE 30、PERSON 19、GPE 16、ORG 16、CARDINAL 15、MONEY 13、ORDINAL 8、NORP 5、WORK_OF_ART 4、TIME 3、QUANTITY 2，以及 LANGUAGE/EVENT/LOC/PRODUCT/PERCENT 各 1。

编译器复用已封存 H010 库存，不重跑 spaCy。来源候选的内容与顺序只由来源决定，并保留重复 source occurrence provenance；目标 native label 选择对应池，包括空池。原答中目标字面的全部重复出现及同 label 的其他 occurrences 按冻结规则遮蔽。offset、Unicode、重叠和 placeholder 冲突不作自由修补。目标原值是否在来源池中仅为最后的 readout 诊断，不能反向筛选候选或判定矛盾。

span 与 label 来自冻结的 spaCy 3.8.18/en_core_web_sm 3.8.0 外部监督前端，包含 OntoNotes 等训练监督，不是完全无监督的实体提案器。目标任务真值未用于本库存。NER emitted occurrences 不等于正确实体、完整断言或语义覆盖；别名、指代、原 Q 锚、候选域完整性、合法算术、非断言和事实性全部为 unknown_not_assessed。字面擦除也不证明其他上下文已无语义线索。

完整 [JSON 结果](h012_natural_inventory.json) 保留原始 sealed summary/counts、全部 artifact（18 原答、148 pools、136 proposals）、protocol、plan、producer identity、complete、成员哈希及 registry result。读取前后核对原始 seal 和全部已登记 artifact 哈希，没有改写库存。

固定设计见 [H012 DESIGN](../../refine-logs/h012-20261103/DESIGN.md)。运行 ID 为 1daab8c5696195e0e46eb394441d5f84528bc9adf62ac5442da925b85bf1c4e9，状态为 sealed_unscored_cloze_inventory；registry outcome 为 inconclusive。原始 seal SHA256 为 93fb4087aa429ad7162fd7b3abcc146d75e0e630e3e6a6e574fd42b44b7c4cc7，124-file 核心 checkpoint 为 d9ce98f1a724c3fb1816e9f9ce41e6dad9caa138。执行身份以 JSON 中封存的源文件哈希为准。未晋升方法，也没有自然效能结论。
