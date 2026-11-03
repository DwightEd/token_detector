# 当前恢复：本地900条文本对照中断已封存，只补2003次推理

2026-11-04 04:35更新：下述M5结论不变。全900文本对照已完成801条预测和第802条部分chunk；原score claim `8902c781...` 已finish engineering_failure。原因是可用主存536465408B低于固定512MiB保护线，显存仍空闲6354370560B；拒绝发生在forward之前，不能说GPU OOM。`local-nli-datatxt900-score-v1/seal.json` SHA `55d00ac080e61c465139d63570bd35e6fc5c070ea43645a5c40be4b939cbbe3a` 封存16160次成功raw，含历史两次canary；剩余2003次。labels尚未打开、没有部分AUROC。

显式恢复方案已独立pre-code批准：保留原predict及数值helper AST、HF和其他数值依赖字节不变，仅新wrapper/CLI校验完整canonical请求前缀并复用所有成功raw（包括partial802）。旧canary是历史资格，不宣称验证新驻留模型。兼容plan-v2需验证旧140文件/runtime并seal；score-v2新claim只允许2003次新forward，仍FP32/eager/batch1、同门限/原轴/窗口/风险规则。`<score-v2>.recovery.json`须区分combined18163模型forward与旧1次资源拒绝接口（接口总18164）。不得启动旧SCORE launcher，不得重复已成功推理；当前完整任务必须先900预测seal及独立无标签审计，再另claim主线evaluate_dataset和固定诊断。网络影响GitHub/新下载，本地数据与模型完整，不能以网络为由停止本地评价。

恢复代码29相关tests pass；当前源码尚未完整CPU/CI验证。新installed snapshot和registry状态以 `outputs/h017-relation-20261103/NLI_RECOVERY_*` 为准。无自动重试，无降低资源门限或改用unknown填充。

主目标仍为原回答无监督检测、六任务各AUROC≥0.8，真值只在全部评分冻结后评价；仍无新检测器/新AUROC。先读F053与docs/results/h017_scope_summary.json。M5模型和独立审计进程均已退出，不能重启旧launch/PID9680/20388。

M5 f47第一claim已finish engineering_failure：native+S_minus两组2full/2head完成封存后progress.json不可覆盖报错。修复只涉及I/O；7cf第二claim显式复用这2组，仅跑S_plus/Tue/Sat3full/3head并finish supported_scope。总5/5，0重复；139最终compatibletarget D:/projects/research/outputs/h017-qualified-20261104-003533/target，wheel4ee0f8257652a5b7cb56226ce02f381cbfd5e2ea9ea4f2b4ab25dfec2a10a5bc。数值observe_world/_profiles AST与所有137其余依赖被独立验证不变。

原1079token、全部185原R目标、所有32层×32头×13接收点完整保留。m=z00−z30: native5.988808,Mon6.096447,Sun外9.410328,Tue6.110023,Sat7.027855；三主差3.313881/3.300304/2.382473均超过冻结界约.0185。全部925FP32endpoint精确重放；FP64AV误差max7.47868e−6、0违反、future0、profiles全等。见m5-postrun-audit-v1/audit.json和m5-postrun-input-supplement-v1。只排除严格全局区间频次等同性；位置启发/周日brunch评论张力未排除，30不是整个Mon–Sat的正确替代，不能称新自然AUROC/因果电路/原GPT4原因。不要追加世界/头/层救援。

累计H017原Llama60model=2earlystop+58full，53head。新DeBERTa调用须另计，不能并作原Llama。M5实际峰值GPU约3GiB、postrun空闲RAM约1.2GiB；不是资源失败。

下一实际任务是LOCAL DeBERTa Data2txt900外部监督比较器，已独立precode批准（NLI_DATATXT900_PRE_CODE_INDEPENDENT_REVIEW.md b12a497...）。本地7资产380MB哈希全部匹配，模型D:/projects/research/models/DeBERTa-v3-base-mnli-fever-anli，无需网络。MiniCheck下载失败暂停，不能让它阻塞此本地对照。全900/150source/6gen、178343原targets、8240既有句scope已核查，未读labels；annotations已仅hash定位D:/projects/research/outputs/fusion-cache-20261102/prepared/test_Data2txt/annotations.jsonl，与manifest一致。

HFEntailment新增显式dataset-fp32-batch1-v1由root负责，legacyFP16/34例不变；collector experiments/nli_dataset.py及其tests由lit agent写，root centralCLI已接。新模式FP32/eager/safetensors CUDA device_map，6GiB cap/512MiB floors。streaming全量规划先seal，最长frame固定canary两次sameFP32接口核验/复用一条，K+1调用，无dedup、unknown.5，只将预算/空文本unknown，资源或数值失败不能伪填unknown完成。score全部900seal后才调用原evaluate_dataset AUROC/AP/固定预算/覆盖。它是外部监督对照，不是新无监督方法。

ROOT改动/newtests暂未完整CPU或当前CI验证；旧9dc6600 CI1101pass不替代新源码。等collector/freshreview后冻结新installed包，模型job前registryclaim，模型后完整CPU/准确commitCI。当前不要执行第二份模型任务。
