# 当前恢复：M3科学关系干预已claim并运行；勿重复启动

最新用户指令D038：开展调研、实验、验证、迭代。主任务仍是已有回答的无监督幻觉检测；真值仅在全部评分冻结后评价。优先读F046/F047，复用已完成结果，不恢复独立角色QA。

**工程已封存并finish：** M0保留2700答/450来源、Data2txt5806字段，QA/Summary语义许可仍unknown。M1由canary3=`5f7cf0ad…`及remaining897=`48f5ce11…`两个真实run合计900唯一QA答/150来源；全部qualified，两量违例及precision-unknown均0，五种坏实现各900例拒绝。报告`outputs/h017-relation-20261103/M1_900_AGGREGATE.json`；不得重跑两个人口或称为一次run。

**M2现已全900：** canary3=`7d065e2e7d08739ac4c2a8ad2097b09e426f10ed7c357561c4f53a7ad1926268`；remaining897=`8d6662cb8bb5060ee975424b0fa12f4c1535a78047e6de39efd0a50a92033911`，均sealed/finished supported_scope。两集合无交集，合900答/150来源/每来源6答，原prompt+R IDs、source mask、T+1位置和来源字符映射匹配。124817个原回答token、125717个query rows；occurrence映射9384contained+12boundary_overlap、按来源去重1566。**900例语义仍unknown/not_assessed，边界相交保留，历史tokenizer body SHA未得到证明。** 两个真实源码身份分别132/134文件。报告`M2_900_AGGREGATE.json`及同名md；验证`M2_900_VERIFICATION.json`。本次只聚合旧结果，0新forward/labels/scores。remaining使用不可变target `D:/projects/research/outputs/h017-qualified-20261103-175903/target`，已完成，无待启动job。

**原权重prefix已完成：** run`1fdf37662387671a919a7eb942457bae29214391790fa92dce73af7442dbeb64`，16465/source12236，完整252-token prompt，q251，执行0..15。native/blocked两世界3site节点/Q32/K8/V8坐标均在预定预算内；2次early-stop，46.401851s，0norm/head。目录`m3-prefix`，F045；直接复用。

**完整输出/cache/no-op现已完成：** run`7f222ddfcf344a70236c452b9444815c1cfffdc8928483de8fe4107f6896cff5`，同一完整252-token原prompt。4次全32层model调用、4次final norm、3次未分块128256词表head，101.082339s。bulk与独立251-token pristine-cache单query相比，hidden最大差4.72069e-5、全logit5.34058e-5、原回答首token logp差1.28746e-5；均通过固定工程容差。Q clone及同值source postRoPE K/V no-op输出逐位相同；32层缓存clone全部独立，pristine未变，hooks/weights cleanup通过。GPU分配峰值3358987776B、reserved峰值4225761280B、process working-set峰值2476904448B；commit统计另计，不等同实驻权重拷贝。证据`m3-head/result.json`、seal、`FULL_HEAD_MEMORY_VERIFICATION.json`与F046。此为工程资格，不是关系错绑机制或检测效果。

**H017实际新增原模型调用合计6次（early-stop2+full4），LM-head3次。** M0/M1/M2均0新LM。不要将随机小模型单元测试算作原模型观测，也不要重跑上述通过的工程门。

**实际正在运行：** M3科学run`24dd6e0bd68475e1561dc4379c2e4e2655ad3999cd935c348af8441aea3bb8ce`已claim并launch，spec=`outputs/h017-relation-20261103/M3_INTERVENTION_SPEC.json`，输出`.../m3-intervention`，日志`.../m3-intervention-launch.log`，exec session33754。CIM核对实际worker PID11384、parent16148、UTC启动`2026-11-03T10:24:36.8571820Z`；另有GPU-venv launcher PID16148。命令SHA/双进程身份见`LIVE_PROCESS_IDENTITY.json`，勿把PowerShell或launcher当worker。运行来自不可变136文件target `D:/projects/research/outputs/h017-qualified-20261103-182300/target`，wheel SHA`e36f287782187f577a905a3fb9172a27c4105ebfda983dd7ad972ae2d66056e4`；plan中136源码SHA全部匹配快照。

计划48次model/44次head是**预算，不是完成数**。此checkpoint已保存receipt确认的下界为model9/head5，可能另有in-flight调用；实时以`outputs/h017-relation-20261103/M3_LIVE_PROGRESS.json`和新receipt/failure为准。既有工程6次model/3head已完成，不与未finish科学counter混淆。当前未评审科学结果、不finish此run。先核PID/启动时间/命令及现有输出，绝不重复launch；等待root请求后才审结局。

协议/代码独立review与局部推导已完成。四个构造许可sourceworld固定R、数字/位置、block15全Q32/KV8，完整32层最终head；不是roleQA。address/payload为受控路径假设，负K输出介导不证明K无角色信息。来源owner仍不得按最近实体补齐；attention集中/稳定不等于正确关系。尚无自然机制结论、检测分数或AUROC，六任务0.8目标未完成。

模型已独立验证9文件16,069,722,485B并disarmed，根`C:/ResearchData/token-detector-remote-20261103/DATA04_models/Meta-Llama-3.1-8B-Instruct`，不要重复下载/移动。四原hash BF16权重以FP32算术逐模块stream；当前cu128与历史cu126不是逐位同后端证明。F040合同保持：缓存Q/K为preRoPE，原阻断执行0..15，只归档15；只有回答T+1 Q/nodes和完整输入KV；Llama观察器teacher-force已有回答，非其他原生成器原因。

当前源码新增尚需完整installed CPU suite及对应commit CI后才能宣称push验证。旧525fd28的992通过只属于旧源码。当前结论以F044至F047、真实run registry和seals为准；旧goal工具blocked不代表研究目标完成。
