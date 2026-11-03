# 当前恢复：关系错绑的内部计算设计；数据收尾并行

**最新指令D036优先：先思考关系错绑如何检验，不把改进限定于V3。** 读 [RELATION_BINDING_DIRECTION.md](RELATION_BINDING_DIRECTION.md)、[编码前科学准入](relation_misbinding_admission.json)、[当前计划](../refine-logs/EXPERIMENT_PLAN.md)。H017仅为已审查设计，未claim、未产生新模型运行/分数；active_hypothesis仍为空。目标是原回答幻觉检测相对直接文本核验的定位/检测/成本价值；H015/H016问答支线不恢复。

四量分开：观察器条件概率、数值关注、事实主体/条件归属、持续性。先查条件接收位置到prevalue的事实所属主体，区分条件失败、检索切换、值/读出失败及post-token回声；不假定正确关系原向量距离更近。M0来源许可/unknown与M1RoPE/GQA/因果/精度资格先于解释。M2观测进入M3只需不同预测与对照spec，真实角色交叉与M3统一补采；最终机制门在联合结果后。旧错例只作研究动机，不按目标标签挑头/参数。

**数据已收尾通过，勿重复下载：F041。** 所选51,812文件/30,817,905,041B（约30.82GB）均恢复；15前缀SHA/引用检查、最终legacy联合目录重验、QA900×13=11700数组头通过。收尾器 `outputs/remote-complete-download-20261103/acquisition-finisher-v2-state.json` 已disarmed，final为同目录 `acquisition-finisher-v2-final.json`。C根 `C:/ResearchData/token-detector-remote-20261103/`，00含索引/收据，另有56KB observer-config匹配原SHA。整远端全量目录快照/模型重算闭合不在该完成声明中；原D缓存保留。

**合同纠错必须保留：** Q/K是RoPE前，不能raw点积当attention。source阻断作用于实际执行0–15层，只保存15层；nodes/query仅回答T+1行，缺完整来源Q/hidden和early-stop最终logp。Llama观察器teacher-force已有R，不是GPT/Mistral/Llama2原生成器内部原因。Summary/Data的lag8量是局部回答历史attention，非同范围来源QKV。原公开交接中的“仅15层阻断”以此纠正，不能沿用。

实验代码在experiments，候选diagnostic在experiments/research，复用主线原轴/基线/封存/评价。memory负责身份去重和证据，不自动判科学创新；每个新模块编码前补科学准入/审查，真实入口与SHA齐后才check/claim，标签最终评价。三旧日期integrity脚本的当前副本已删，H002完整副本不动；干净wheel126文件/导入/CLI通过；本机完整CPU900秒上限为incomplete，源码清理的完整验证需查当前提交LinuxCI及outputs/relation-mechanism-design-20261103/github-ci-success.json，不能沿用旧69aead3的成功。

历史下载范围和旧执行快照另存 [归档](history/RESUME_20261103_161917_with_acquisition_history.md)。历史F001–F039与负实验见facts/prior_experiments；F040为合同纠错，F041为当前文件恢复完成，六任务0.8目标仍未完成。主项目D:/projects/research/token_detector；模型实验本轮未claim，旧goal工具blocked不表示任务完成。
