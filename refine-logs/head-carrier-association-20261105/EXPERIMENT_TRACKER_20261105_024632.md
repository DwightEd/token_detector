# 实验追踪：H019已完成，H020未准入

| 阶段 | Run ID | 状态/成本 | 结论 |
|---|---|---|---|
| H018 target-only独立审计 | 7626e002 | DONE，0模型，233token/32层/4V | 算术通过，必要主方向反证，关闭不补control |
| H019 M0 | 02e68af9 | DONE/sealed，10q/6pair/343target状态 | 有限CPU资格通过，手工局部license |
| H019 M1 | 709af130 | DONE/sealed，92160tuple，8.714s/31.4MB，0模型 | pre-O算术通过，不稳定SEM/EQUIV语义幅度，不晋升risk |
| H019独立全tuple审计 | 9678946a | DONE/sealed，80位Decimal/165.607s | 全部原A/V连接、覆盖及数值通过，不等于语义 |
| H019 sourceΔV精度 | d1de688b | DONE/sealed，64小表/1536vectors/.284s | 1488不同，不能routing-only归因；不重跑 |
| H019安装验证 | sourceee69ad0 | exact154，60targeted20.89s；full300s INCOMPLETE | 新push完整exact LinuxCI待核，旧feCI不覆盖 |
| H020/I02 | UNCLAIMED | 正式合同/独立审查中，0LM/0risk | 最多4原S+2Scontrol计划，不是已运行或已有效 |

详见docs/results/h019_head_carrier_association.md、F063/F064与RESUME。六任务AUROC目标及顶会方法均未完成。保持原R/原token轴、unknown和所有科学失败；旧现象直接引用。
