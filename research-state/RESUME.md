# 恢复入口：H003两组已完成，下一项只检验来源类型表达

项目 `D:/projects/research/token_detector`。六任务分别AUROC≥0.8未达成；无监督，gold只用于封存后的评价与事后诊断，不训练、扫权重/阈值或挑任务赢家。用户本轮授权的资源处理和研究已执行。`get_goal`仍返回旧`blocked`控制状态，API不能自行恢复；不要声称自动续跑已开启，也不要另建目标覆盖旧目标。

## 已有证据，不重复消耗模型

- H002：18次自由提议，8个合法frame均无slot；来源direct/blind/leaky各0调用。位置与角色接口失败，核心来源重建未测试。F012及[结果](../docs/results/h002_proposal_qualification.json)为准，不补输出、不重跑。
- H003自然：原18完整答、108前向已封存及CPU复核，前向41.48秒。18例无一在全部标签分配下语义argmax一致；Data2txt等存在高支持分放过错误。3个独立助手盲复核保持歧义，不是新增benchmark gold。见F013与[自然结果](../docs/results/h003_natural_verdict_qualification.md)。
- H003控制：16答/8对、96前向12.58秒，完整封存及复核。三类argmax11/16，6/7支持差方向正确；六个同词袋对五正一反。null被当False；合法80.8计算判为证据不足，直接写出结果后支持。见F014与[控制结果](../docs/results/h003_constructed_controls.md)。不是自然AUROC或原始生成因果证明。

下一条问题：保留全部字段路径/类型的来源表达，能否改善null语义及自然结构化来源错误，还是仅增加规则提醒有效？[H004交叉对照](TYPED_SOURCE_NEXT.md)已设计，未实现/运行。只按输入语法处理；表达×规则四条件，不按gold挑选。算术执行和盲重建保持后续独立假设，不同时堆模块。

## 运行、版本与恢复边界

所有H002/H003 worker已完成；历史autostart-state的observations_sealed_not_evaluated只描述观察阶段，不代表CPU评价没做或进程活着。不要恢复31576、28236、62168或24272。模型资源已释放，不再关闭应用。

| 已完成运行 | 完整ID | 精确源码/只读目标 |
|---|---|---|
| H003自然观察 | `fa101d666cce73beefcb12cd5979b04fb51e2771a61f48e78f9232caecff968c` | 97文件，Git `91ae38f`，outputs/h003-verdict-20261102/installed |
| 自然CPU复核 | `29e82c69db9558ec05c7b868f10cfa58f06cc497864a0ecfebcefa6217a72b47` | 98文件，Git `4fed6e1`，同目录evaluator_installed |
| 构造观察 | `1dfd32b1cdf5ad6f389ee9b26eaaa9c4a1c96b497ae67d851d351996eca5688b` | 98文件，outputs/h003-controls-20261102/installed |

原CUDA环境94包、H00295包、H00397/98目标分别保留。不能拿当前helper误判旧缓存后要求重跑模型；恢复对应只读reader。自然观察与CPU源码逐字节发布在不同commit，控制人口改动另行提交。详见state.json与verdict_qualification_validation.json。

CPU登记v1因少两个身份字段在claim/gold前停止；无效spec已归档outputs，v2成功。旧H002第一次评价也只是在错误CLI入口停止，已修复。两者都不是科学负结果或重复模型运行。

## 继续操作

1. `python tools/research_memory.py context`；只读F013/F014、当前设计及有关历史。31条旧索引保持，不重读130KB原索引。
2. 新条件先冻结代码、输入、模型/后端、分数与完整人口，check/claim；已完成观察复用。三个H003登记均已再次核验为reuse。
3. H004先实现受限语法的类型/绑定保留与拒绝行为，再冻结实际交叉条件。它不是论文主方法，未知/超长/解析拒绝不能当正常。任何收益仍需完整六任务和未接触确认人口。
4. SOURCE_INVENTORY目前128个Python文件；原21文件/2421行/128定义的解释覆盖保持。新文件与改动仍待逐行解释，检查通过不代表讲解完成。

用户允许关闭Edge/Termius/夸克，已完成；不关闭Code/WPS或其他应用，不反复关闭重开的程序。两次不同零模型待机worker各做过一次自身驻留页回收，不循环清内存。等权、local重置/repair、有限证书覆盖不足及格式/数据合同差异不换名复验。目标未达成，不标complete。
