# 细粒度绑定研究：已完成证据与下一判别入口

当前F059–F064与RESUME为准。目标仍是无监督检测原回答，六任务各接近/超过.8与完整可验证方法；未达成。GT仅冻结评分后最终评价，不能选头/层/阈值或方法。下面是实跑后的新检查点，不授权任何模型或检测评分。

| 主张/阶段 | 现有证据 | 决定与下一门 |
|---|---|---|
| 保存数组真实可用 | reanchor70files10.08GB，原16/replay16/graph2各自资格完成 | 复用，不重下/重资格；原incremental、replay和BF16 graph状态分开 |
| H018局部source使用方向 | 完整target独立审计后必要正方向反证 | 固定分支关闭，不补control，不翻号/选点 |
| H019双边marginal关联擦除 | 10原q×6人工局部sourcepair×32层×48独立headpair=92160 tuple，独立80位审计全通过 | 仅pre-O计算身份；SEM/EQUIV幅度不稳定，单独幅度/MI/chi风险关闭，不重复M0/M1/audit |
| source深层状态跨graph精度 | 64封存ΔV表1536向量仅48相等，maxabs.078125 | 不将跨R幅度差只归query routing，不重跑精度；不隔离kernel/capture/历史身份 |
| I02来源优先语义许可 | 旧H002是R-first；真正I02未跑；H010依存库存不自动许可关系语义 | H020 S+question先记录主体/动作/数值/单位/条件/极性/推导义务及unknown，引用由精确quote+occurrence确定；字面存在不是语义正确 |
| H020最小新观测 | 拟固定4原S+2source-only控制，现有Qwen2.5-3B，最多6请求 | 正式科学合同+独立审查、代码/安装资格及真实导入后资源门通过才claim；未准入LM，无R/GT输入，不重跑旧18次proposal |
| 原回答检测与完整评价 | 暂无新risk；新采样R不能套旧R标注 | 前端冻结后原R接口、unknown全轴、同record直接文本强基线和真实成本；内部量须有增量价值后才完整评分/最终评价/消融 |

旧reanchor O1–O5/NAG/source-null/Vswap/endpoint/pathcos结果直接引用，不重演。6个语义proposal请求不是6次backbone forward，需逐次记录prefill/decoder实际调用及generated tokens；不改forward签名。H020仍设计，不能预填资源通过或检测AUROC。

代码ee69ad0 exact installed154/60targeted通过；full CPU300s incomplete，新push完整LinuxCI待核。D清理约16.4GiB完成、当前16.43GiB空闲；科学raw/模型/失败保留，新大raw落C。无活动GPU/pytest/download/armedwatchdog。
