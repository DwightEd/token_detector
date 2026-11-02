# 当前恢复：H009无准确率增益，推进自然问答合同

用户要求继续迭代不要停。六任务同AUROC≥.8尚未达成；旧goal工具blocked不覆盖持续授权。真值只在方法/评价/预测冻结后用于评价和事后诊断，不用于目标训练、参数、模板、任务赢家或经验权重选择。每个结论更新facts、decisions与本短入口，已知现象不换名重跑。

## 已知结果与本轮决定

- H007/F021：480实际forward，焦点抗负担假设失败；预定I=−.057753。删后缀288请求严格等于96个既有b0请求，0新推理，最简仍仅5/8。不要再跑。
- H008/F023：同3B换为S-only问题＋完整姓名/EOS重建；8来源、16唯一问题、32forward4.1325464秒，16/16正确、sourceflip/eventflip各8/8；48旧构造回答AUC/AP1不是48独立QA。stricttoyparser已能直接解题，未证自然coverage、原生成因果、lengthbias消除或新颖性。观察c841a9359221a06af5bc04f1efe2c0b2d86b6691218ec09cdc97bc3deecfebf2，CPU b9b8f31decfd72962a48790e389db8c6370d3232a4e6348529d5d38af60da6f4，均sealed。
- H009/F024：已按观察前STRUCTURED_FOLLOWUP跑完整4臂，0新模型。baseline/固定a[-2,-5]与b[-1,-9] ×独立/双射均16/16、8/8、AUC1；不变性8/8通过，但没有准确率增益证据。不得放大偏置制造优势。运行4235ad12a1e91bdda792ac8519adfe3dab88b8fe0b960378bdd13b2340ec77c9，outputs/h009-assignment-20261103/assignment_review_v1与docs/results/h009_cached_bijection.{md,json}。
- H010/F025：新独立CPU NLP环境，冻结外部监督spaCy3.8.18/en_core_web_sm3.8.0；18样例/36字段，719S/136R exactspans，0工程unknown，5offsetprobes通过。**没有QA/score/语义覆盖评价**。outputs/h010-natural-span-20261103/completion-receipt.json、FEASIBILITY.md及pilot-summary.json。禁止span数冒充覆盖或实体同现冒充事实。

## 下一判别问题

先读lead生成的outputs/h010-natural-span-20261103/NATURAL_CONTRACT_NEXT.md；若尚未生成检查agent，勿重复委派。主文献与可识别性审查正在进行；S,R能发现关系不受支持不等于能唯一定位被编辑person/event；不能由此推出全局AUC=.5或不可能≥.8。原问题Q可作语义锚，但真实且离题不自动等于hallucination；金融问题缺失的历史审计先复用。

候选H011尚未登记/实现：在H007已知exactslots先隔离NER误差，R中目标姓名全部擦成[TARGET]、另PERSON擦成[OTHER]，保留事件及burden。S独立列候选，固定模板，预计48Q/96complete候选；测试S定义Q资格能否迁移到R擦值Q，非权重/模板搜索。必须先完成prior/信息流/可识别性审查、冻结collector和协议再执行，不能把此候选当已授权运行中的job。自然136NER proposals候选完整性、别名、跨句、类型错、nonassertion/arithmetic仍未解决。

## 身份、资源、同步

H008 observer112 wheel9c8cf5e0425da2f97ca3704d25f8d0734d5c7343953a70bcfcd5b2707a171f03(code9f836fa)，reader113 wheel4d74f0aa17a3a6b77ff7f20dd93c7801cb9d10be168c39d103dd4d8c902301db(code77eda09)。H009115 wheel114fb8cff396ae635c9395ae076a4211bfc4a7dfbc2646d7716a9ec133deaafa(code89ca8e4)，6outsideinstalled checks通过。已sealedoutputs/targets/wheels/原design不可修改。

7B9个公共资产已完成校验；outputs/larger-reference-readiness-20261103/receipt-v2.json。独立有界helper19736仅2synthetic工程forward，目前waiting_postwarm_resources，读runtime-readiness.json；不是新检测实验。勿重复下载/关闭用户应用/自动重试。若资源门不可达先审查实际门槛，不能把等待说成模型已可用。GPU环境/CPU原env保持；本轮只有isolatedNLP新env。

完整自然3883输入与五组2992答/448202历史token轴见F022；Hallu891仅小样本确认可恢复原序列化ordinal，full_index未知，V3full_gap不是R4，不换字符/新tokenizer冒充原轴。H004/H005/H006负/混合结果直接引用已有facts，不能重新选择赢家。

源码讲解新batch：docs/source-reading/batches/20261103-h008-core-v1；仅新增2files/296lines/12defs，累计23/2717/140；整体168文件中仍145待讲，不得说整个项目已逐行讲完。Git核心HEAD89ca8e4，本轮结果/state/archives待统一push；桌面此前dccbb2d，导出agent正在扩展H008/H009/H010+batch，root提交push后最终导出。
