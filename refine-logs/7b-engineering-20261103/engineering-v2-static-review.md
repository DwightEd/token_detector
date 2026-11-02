# 7B engineering v2 静态复核

结论：**在 root 先确认 H011 worker 已退出、旧7B helper 已停止，再显式启动一次的条件下，未发现阻止这次工程资格尝试的实质问题。** 本审查没有运行新 helper、CUDA、模型、tokenizer或数据。它不保证实际装载成功；资源不足/OOM应保留本次deferred receipt，不自动重试。

复核源码 `qualify_runtime_with_reserved_context_v2.py`，426行，SHA256 `8d559369a472fa7c5049b3c6a4a6ddfd4e3a9943976d28071f45195cd9c0e935`。旧源码 SHA256 `8fb7e97786a176554b2fa23ee0fc8e86be88d6af31d2578e9febb0111d8730c4`。用 preparation 中7组文本替换重建后与新文件全文一致；其中 `three-hour` 出现两次，其余每组出现一次。AST/静态compile通过；没有执行任一 helper 函数。

本次只读取旧receipt的工程字段：`cancelled_this_helper_no_retry`，attempts=successes=0，weights_loaded=false。检查时新 `runtime-readiness-v2.json` 与新STOP marker均不存在；这不是未来运行时状态的保证。旧观察仍保留，未覆盖旧receipt。

## 已确认的合同

- 后warm实际free GPU门为6.5GiB，PyTorch allocator cap为6GiB，差为 **536870912 bytes=0.5GiB**。两者都比旧门/旧cap降低0.5GiB，保留同样差值；没有增加模型预算。RAM仍先3GiB、warm后至少约2.3GiB，权重装载前再查资源。运行时固定 torch2.8.0+cu128 / Transformers4.57.1 / bitsandbytes0.48.0 / accelerate1.10.1，要求SM120与BF16。
- 九份资产的文件名、字节数、SHA与固定revision均未改变。完整资产先核对一次，等待资源后、权重前再核对一次。保存的NF4/BF16配置与私有/公开load标志需一致；`from_pretrained`使用本地资产、`trust_remote_code=False`、BF16、SDPA与单一CUDA:0 device map，没有quantization_config覆盖、CPU/disk offload、`.to()`重放量化或重新保存资产。装载后还检查saved4bit、BF16、eval与全部参数device。
- 恰两次synthetic forward：一个短聊天前缀，一个由非特殊token直接构建的**精确512 token**序列。没有截断、数据集、GT、准确率或预设类别正确性。每次只保留最后一行完整词表logits，禁用KVcache，检查 `[1,1,V]` 与全部finite；A/B/C仅作工程取值检查，不作语义预测。attempt在forward前增加，任何异常不重试。
- 新receipt以 `open('x')` 独占创建；重复启动遇已有receipt即拒绝。新STOP只影响本进程。45分钟共享deadline覆盖等待、asset核对、imports、装载和两次forward；进程内1秒watcher记录终止状态后调用本进程 `os._exit`。没有外部服务/任务安装，没有关闭或终止他人进程。下载JSON最多3次的短暂重读只处理并发更新造成的JSON撕裂，不是模型重试。

## 启动条件与实际风险

**helper内置的是旧H007的退出见证，不是H011 worker身份门。** 因此“等H011退出后”须由root在启动前落实；不能把GPU暂时空闲（例如H011尚在load前）当作H011已退出的证明。建议核对本轮36792/60036对应进程确已退出，再按这份独占receipt单次启动；本审查没有主动查询或控制这些进程。

0.5GiB是资源测量时相对PyTorch allocator上限的余量，不是为整个CUDA运行时或其他应用预留的系统级锁。CUDA库临时分配、并发进程和装载峰值仍可能耗尽余量；allocator cap也可能使7B在到达两次forward前被拒绝。这应被视为工程defer，不得放宽cap、改offload/quantization或自动再跑。进程内deadline依赖正常的receipt文件I/O；它不是独立OS级监督器，极端I/O卡死不能由静态审查证明有硬实时终止保证。

两处旧诊断字面量仍写 `6.5 GiB ... cap exceeded` 和 `STOP_7B_READINESS`，而实际判断分别使用 **6GiB ALLOCATOR_CAP** 与 **STOP_7B_READINESS_V2** 常量。它们只影响消息可读性，不改变实际门、receipt阈值或停止路径；本次只读审查没有改动源码。

本结论只支持上述受控的一次工程资格尝试，不涉及社区checkpoint的事实能力、数据污染、自然任务AUROC、主方法选择或训练。
