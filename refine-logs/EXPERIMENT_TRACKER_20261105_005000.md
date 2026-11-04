# 当前执行账本：缓存补齐，未宣称检测提升

2026-11-05，Asia/Shanghai。[对应计划](EXPERIMENT_PLAN_20261105_005000.md)。

| 项目 | 实际状态 | 核心证据/限制 |
|---|---|---|
| P1原两条Data2txt | DONE，独立审计PASS | 407原targets；原两例不再跑；关系特异性/新AUROC未建立 |
| P2 owner前端 | DONE：34a449f0 | 2新S，原R233不变，source token袋/恰2bool位置，0新模型 |
| P2第一模型尝试 | engineering_failure：6fe770d9 | 1partial native/0head/4extraMLP，主存门拒绝；保留 |
| P2明确恢复 | engineering_failure：7c6d857e | target child sealed，control disk-full中止；总2native/4head/204extraMLP。target独立raw audit待做，不能报完整对照 |
| 新生成多采样原缓存 | 下载DONE，资格中 | 4source×4seed同Llama；16NPZ+3metadata，1,168,475,270B。此前旧盘点遗漏reanchor |
| replay hidden16份 | 下载中 | 实际hidden[33,N−1,4096]FP16；逐前缀重放，与原生成分开 |
| 两份完整attributed graph | 实体已定位，准备下载 | A/V/residual/post_attention/MLP写出及原executed源码；非16例全graph |
| D盘清理 | DONE，两项迁移 | 12.22GB原RAGTruth压缩包迁C并SHA核对；5.43GB OEM安装源转存C并核文件清单；D空闲约16.4GiB |
| 源码验证 | 65targeted PASS；full INCOMPLETE | 151精确installed；本机300s timeout保留。新push另查完整Linux CI |

0新目标GT/新AUROC。新16生成R不得套旧官方标签；所有source/seed/head/layer保留。没有活跃GPUjob、自动重试或旧双worldlauncher。新方法、效能、新颖性和顶会完整性均须另行验证。
