# 当前恢复：M3负结果已完成；M4自然时间错例已准入、待代码审查/claim

最新目标仍是已有回答的无监督幻觉检测；真值只在评分冻结后评价，不回到独立角色QA或盲目加权。先读F048、当前报告docs/results/h017_relation_binding.md，再读F044–F047所需合同。六任务各0.8尚未达到，当前没有新检测分数/AUROC。

**没有活动模型job。** 科学run24dd6e0bd68475e1561dc4379c2e4e2655ad3999cd935c348af8441aea3bb8ce已完成/封存/finish(refuted)，48次完整原模型、44次head，847.827918s。原worker11384和launcher16148不是待续跑job；最终状态outputs/h017-relation-20261103/M3_FINAL_PROGRESS.json。旧LIVE_PROGRESS仅为历史运行下界，不得据此重启。

**科学结论F048：** 实验仪器与独立复核全部通过，属于科学负结果。block15数字K-role四组方向效应−0.03625/−0.00617/−0.01222/−0.20856，均小于范数匹配随机控制；attention地址3/4向预期移动，却没有相应最终logit变化。V-value四组方向效应+0.23024/+0.17615/+0.98068/+0.37843，Q/K/attention逐位不变。完整事实换序margin差3.32657。错误固定R的p200仅9.27e−6与2.565e−4；**没有复现高置信错绑**。不能据此设计K/attention风险、宣称K无关系信息，或把V通道效应直接升级为检测模块；postvalue、原生成器原因均未观测。

run输出m3-intervention，独立审计m3-interpretation-audit-v1，源136文件冻结target D:/projects/research/outputs/h017-qualified-20261103-182300/target，wheel e36f287782187f577a905a3fb9172a27c4105ebfda983dd7ad972ae2d66056e4。原始单位/pristine32层KV与完整logits全部保留；不重采48/44，不挑头/层/prompt救正常toy。

**下一判别问题（F049/D045已更新）：** 只对既有冻结panel中两个RAGTruth数字ID QA来源做source-only人工语义许可，保留完整主体/谓词/时间范围/否定/value及原span、unknown，随后独立审查原回答诊断合同。task=QA还有FRED/HalluRAG条目，不能混入当前QA900缓存；不读取evaluation_only/标注。source-only manual license不是自动前端覆盖或新检测效果。当前只在准备，没有新自然评分/模型claim。

**已完成工程直接复用：** M0共2700答/450来源，Data2txt5806字段，散文语义unknown。M1真实3+897共900答/150来源全部qualified，五种错误实现各拒绝900；M2真实3+897全900原轴匹配，124817回答tokens，125717query rows；9384contained+12boundary_overlap保留，900语义unknown。精确旧run身份见F044/F047与M1/M2_900_AGGREGATE.json。原16465完整252prompt的native/blocked prefix2calls通过；full-head/cache/no-op4calls/3head通过，F045/F046。不重跑这些资格。H017合计54originalmodel=2early-stop+52full，47head；0目标标注/新risk。

模型9资产16.07GB已独立验证，根C:/ResearchData/token-detector-remote-20261103/DATA04_models/Meta-Llama-3.1-8B-Instruct；31GB数据已恢复，勿重下或移动。原BF16存储/FP32算术，cu128与旧cu126不是逐位同后端。旧缓存只有回答T+1 Q/nodes、第15层原始preRoPE Q/K及全输入KV；阻断执行0..15；是观察器而非其他原生成器状态。F040保持。

**验证/发布：** 95项相关CPU通过；当前136文件安装wheel的完整CPU套件触及900s上限，状态incomplete，不能说全套通过。日志与cpu-suite-result.json在182300快照目录。push之后必须查看该确切代码提交完整Linux CI并修复失败；旧525fd28的992通过不替代当前CI。离线14处公式在docs/source-reading/H017_MATH.html。

M4 source-only许可已完成并独立核查（25tuple/86span、非穷尽/非自动前端）。root在其冻结后才检查原Q/R，发现15716是over300yearsBC→around300yearsago的时间参照错误，非亲属关系换绑。M4签名准入901c677…及独立审查8c540148…允许1原完整441token native模型+1未分块head76rows，保留全部32层/32头/77原query；新代码正在写/审，未claim/未采样。源300KV不能依赖未来BC，但可凭先验预测；Rquery/BC后状态可能已持有条件，不能从低BCattention断言未理解。文献Echo/LLM2Vec/BindingIDs/ReDeEP/2026SHROOM已限制广义创新声明。无repair/单位替换/新QA/自动风险读出，先全坐标资格再解释新观测。
