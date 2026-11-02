# H007 原设计与复核归档

此处保留观察前设计、理论、生成器及独立复核的原字节副本。实际结果是负结果，见 [H007 结果](../../docs/results/h007_response_burden.md)，不把下面的先验假设当作已验证结论。

原始冻结材料与运行包在 outputs/h007-burden-20261103。生成器的 generate(root) 可向新的空目录复现 48 条构造材料，默认不会覆盖现存材料；其标签由生成规则确定。代码库主命令 transfer qualify-designed-focus 负责观察，transfer evaluate-designed-focus 在完整预测封存后做 CPU 评价。不要为重现设计而重复调用模型，已有相同请求应直接复用。
