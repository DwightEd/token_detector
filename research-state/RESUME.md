# 当前恢复：H011擦值QA桥失败，推进H012自然工程库存

用户持续要求继续迭代不要停。六任务同AUROC≥.8尚未达成；旧goal工具blocked不覆盖继续授权。目标gold只在方法/预测/评价冻结后用于评价与事后诊断，禁止标签训练/参数/模板/任务赢家/经验权重选择。每结论更新facts/decisions，本入口不堆回旧日志。

## 本轮实际结论

H011/F026已全部完成：同3B固定模板、48旧答/96槽/48Q96完整姓名+EOS，96forward12.3813288秒，32816input/516forcedtarget/0generated，0unknown。b0/4/12正确9/16、8/16、10/16；sourceflip3/8、2/8、4/8；eventflip1/8、0/8、2/8；全48答AUC.6371527778/AP.6798580339。最短也失败，不能归因suffix。封存raw选择显示每8来源有7/8、8/8、6/8两个角色选同名，双反向均0；不是新分数或attention因果证明，toy锚在逻辑上仍充分，不能说擦除必然信息丢失。

观察954488248f0417aebefd288d07c8976b502860d1c397579b04af2fc5d7126283，CPU5c162fe8ade0a235a221a9c0ad309bb1f98a3a8cb79ba9a06e3887192f7ba3fb；全sealed/registry完成，worker36792/parent60036已退出，勿重跑。docs/results/h011_masked_reconstruction.{md,json}、图、refine-logs/h011-20261103角色审计与两轮独立review。Observer117wheel60cc608fbc408cde8f09845c71a4c978e6306c5c728585b11da1ad8b1617e5df/codeb6a7995；CPU118wheelb5ece3c5b43f149d33d909e7725cd63d84eaee9898a78ef48605e874594af461/code5690007，均已push。19kernel、11sharedcollector、3完整integration、27旧默认CPUregression、6newCPU及6outsideinstalled已过；tests不是效果。

H008/F023仍16/16来源定义QA、两flip各8/8，32forward4.1325464秒；stricttoyparser可直接解域，未证自然覆盖。H009/F024复用32raw、0新模型，4臂均16/16，结构不变性成立但无准确率增益，不能加强bias造优势。H007/F021焦点抗负担失败I−.057753；删suffix288requests严格等96旧b0，最简也仅5/8，不能再包装改进。H004/H005/H006负/混合结果复用旧facts与报告。

## 当前下一问题，不能越走越偏

H012由lead设计、nli_tests可写纯kernel：outputs/h012-natural-masked-20261103/DESIGN.md。只reuseH010已封存18答/36字段719S/136R NER occurrence，做全部136自然cloze artifact与S独立同nativeNERlabel候选、重复sourceprovenance/exactliteral擦除/overlap等unknown；不生成QA messages、不跑模型、不声称语义覆盖。semantic anchor/candidate completeness/alias/arithmetic/nonassertion全部unknown_not_assessed；无person不同或双射假设；无/单候选不判矛盾；Rmembership只readout诊断不得反选候选；未提案tokens继续unknown不填0risk。H011否定directcloze迁移必须写入前提，不能换7B/prompt救分。

root提出下一待prior/混杂审查入口（未登记/未实现/未运行）：对原复合句只取目标对应单个谓词conjunct并保留event锚，H011所有system/template/numeric合同不变，只maskedstatement原子化，16Q32complete并保留全部48/96slot映射；严格cacheaudit先查是否等旧请求。H005/H007sentence级仍含两谓词，可能是新结构差异，但须lead审查P026/P025/旧scope是否重复，不能先说因果/新颖性/自然有效。另一入口S定义问题再R匹配也尚无自动可靠实现。不要盲跑自然3883或重访已知现象。

FRED原Q恢复/F027首完整exact路线封存负结果：162Fin+130TAT、252distinctS，官方pinnedRagBench三列49666rows rawdoc0匹配0。官方三脚本未找到可证明normalize变换，不fuzzy/读R/labels消歧。outputs/fred-original-question-recovery-20261103/attempt-v2为16membersealed38.2MB三列projection；README/completion/review已完成，Git只小证据refine-logs/fred-question-recovery-20261103。此结论不证明所有Q恢复不可能。HTTP/footer可接收未选列字节/stats，仅三列materialized，不能夸大字节级隔离。

## 资源和同步

无GPU job armed。7B公共9assets完整；旧7GiB门waiter0forward已软停。显式v2较小6GiBallocator/6.5GiBfree在确认H011退出后通过2synthetic(short+512)；peakreserved5953814528bytes，helper23348/parent33324退出。runtime-readiness-v2.json，F028/独立review只证明工程、非4096/检测效果/模型选择。禁止重下weights、关闭用户应用、改旧env或自动重试；尚无7B科学实验。

全任务3883输入、五组2992答/448202历史token轴见F022；Hallu891仅小样本可恢复原serializedordinal/full_index未知，V3full_gap不是R4，不换字符/新tokenizer冒充原轴。源码batch只新增2files/296lines/12defs，累计23/2717/140、145文件待讲；H011新文件也未讲，不能称全项目逐行完成。

Git核心最新5690007已push；本轮H011结果、FRED/7B小证据/state待统一commitpush。桌面D:/Desktop/myproject_研究资料_20261102此前dccbb2d，exportagent完成H008–H011/batch严格扩展，正追加FRED sealedprojection和7B小receipt；root commit后运行export_checkpoint.py并验证offline MathJax/links。实际commit ZIP应等HEAD且cache不写不删。旧冻结targets/wheels/design/rawrecords均不可修改。
