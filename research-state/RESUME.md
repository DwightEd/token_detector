# 当前恢复：完整900条检测比较已完成，禁止重跑旧任务

先读F054、D050与docs/results/h017_nli_detection.md；精确结果为r4AUROC=0.798317534,AP=0.190392333; witnessAUROC=0.755715886,AP=0.145209394; nliAUROC=0.744306164,AP=0.097168289。这是外部监督DeBERTa文本对照，不是新无监督内部机制。全部900回答/150来源/6生成器/8240句/178343原token保留，unknown0；固定全局8918报警，配对source bootstrap2000/seed17。六任务各0.8的主目标未完成，不能汇报任务最好值包络。

完整评分seal a12c6a29a005d559dd61bac49f1dad7c280c3e112f87ee3040321c1d08ce194e。旧score-v1在801+partial802时因host余量536465408B触发512MiB门限，GPU仍空闲5.9GiB。原claim8902记engineering_failure；原16160成功raw全部保留。显式caba兼容plan0model完成；c6b4520f续跑只新增2003model，合计18163成功/18164接口（旧guard1），旧canary只复用历史资格。score-v2已封存并经独立全900无标签审计；之后另claim主线evaluate_dataset，最后固定R4/V3同轴诊断。所有进程均退出，没有缺失推理或后台监控，不得再启动NLI_*旧launcher。

生产者安装target D:/projects/research/outputs/h017-qualified-20261104-043603/target（140成员，wheel dcc9a73bdbd45cf4dc7e04dcefc1623c0a3c0edb3debba2fc247657c0fcc3d02）；原始prefix producer为040046/17f0db73…，核心predict/HF/数值依赖经AST/字节兼容证明。完整审计和恢复sidecar定义真实调用成本，不能把canonical replay计数当本次新增。标签仅最终评价打开；不能用这些标签调参、选择权重或删人群。

原Llama机制观测仍60model（2earlystop+58full）/53head，与DeBERTa分开。M5五世界F053已经完成，三项有限比较通过，但位置/周日评论混杂仍未排除；禁止追加世界/头/层、重复高概率或角色QA。它不是新自然检测分数；旧H015/H016、M3反例、scope-only/等权负结果不重开。

本机完整CPU测试在045716外置安装wheel（代码与043603相同，新增nli依赖extra）请求完整套件后达到明确300s上限，记录incomplete_timeout而非通过；日志及proof在outputs/h017-relation-20261103/RELEASE_045716_*。确切新源码LinuxCI状态见NLI_PUBLICATION_FINAL.json和GitHub实际提交，旧9dc6600的1101pass不能替代。研究输出不随Git发布，证据文件的路径及SHA已在公共summary里固定。

下一问题：当前没有准入新的方法、NLI救援变体或模型实验：新结果表明明确支持的否定关系也可被判矛盾，而营业时间错误已获高非蕴含风险，尚未证明可利用的新增内部信息。 下一动作限定为零模型的准入证明：从来源显式许可的字段建立主体—谓词—极性/条件闭合单位，先以5682/13603（原R的`not`=target65、`WiFi`=67）与同源5684（`or`=65、`lot`=66）为入口，补齐对应来源字段及闭合位置，复用已完成库存合同标明哪些物理观测已可匹配、哪些维度缺失，不再清点或复验已知缺失；不得拿QA层15或回答历史局部attention代替缺失的来源观测。 只有能事先提出可区分条件绑定与词法/位置/回答回声的预测、必要负对照，以及相对直接文本核验在原R定位/检测/成本上的具体价值，才重新申请有限实验；若只能换contra/max/格式/分句、复现H004/H005或M5手工世界、M3头层与H015/H016角色QA，则保持NOT_ADMITTED并停止追加调用。
