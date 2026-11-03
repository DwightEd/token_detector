# 当前恢复：H017实际运行，先核进程与封存，禁止重复

最新用户指令D038：开展调研、实验、验证、迭代。目标仍是已有回答的无监督幻觉检测；标签只最终评价，不恢复H015/H016角色问答，不限于V3。

**已完成：** M0 run80913833…，2700答/450来源全部保留；Data2txt5806显式字段。QA/Summary语义unknown是已知前端边界，不当新机制结果。M1 run5f7cf0ad…，3个不同source canary全部qualified，5种错误实现均拒绝，groupmass最大绝对误差0.00176084；0模型/标签/检测分数。原证据已finish，勿重跑3例。

**实际在跑：** remaining897 run48f5ce11…，输出`outputs/h017-relation-20261103/m1-remaining`，实际CPU PID40216、launcher54888；先核PID/start/命令，封存前只是running。冻结外部target`D:/projects/research/outputs/h017-installed-20261103-172234/target`，130文件与wheel一致，后续checkout更改不影响它。使用CPU .venv、2线程。完成后聚合已封存3例与897例，不伪造单一run或重算旧例。

**模型准备在跑，无GPU科学任务：** `outputs/h017-relation-20261103/model-acquisition/mirror-state-v1.json`，实际mirror PID35512；9文件16.07GB，4个原hash-attestedBF16safetensors，另5元数据。实际模型根为C数据总目录下的**DATA04_models/Meta-Llama-3.1-8B-Instruct**，不要错写04_models，不移动活跃文件。现有8GB GPU不能全模型常驻；原BF16存储/FP32算术须逐层流式资格。标准BF16loader或NF4不是本合同替代；新streamer须独立review/claim后才能加载原权重。

**下一步：** 完成900覆盖资格；用主线encode_source/encode_response验证分开编码的prompt/R与原cache token/mask/位置一致，才映射source occurrence。再对固定值/位置的角色交叉与Q/K/V分离路径测试最终原tokenlogit；不是新角色QA准确率。M0 prose未知不以最近实体补齐。重放误差界和negative controls在真实数值前固定，不按标签挑head/阈值。当前无新检测AUROC，六任务0.8目标未达到。

读`research-state/RELATION_BINDING_DIRECTION.md`、`relation_misbinding_admission.json`、`refine-logs/EXPERIMENT_PLAN.md`。Q/K为preRoPE；source阻断执行0..15而仅归档15；只有回答T+1 Q/nodes，完整输入KV，无earlystop最终logp；这些是观察器状态，不是原GPT/Mistral/Llama2生成原因。读取错误旧假设按F040纠正。

31GB缓存恢复F041已完成，勿重复下载；00_manifests含证明，原D缓存保留。当前新增代码尚未push/全CI，34项新合成合同测试通过、130文件安装快照通过；旧提交525fd28的992通过不能充作当前代码CI。新科学运行均check/claim，完成后finish并写结论与下一判别问题。历史负实验F001–F039直接复用。旧goal工具blocked不表示当前研究完成。
