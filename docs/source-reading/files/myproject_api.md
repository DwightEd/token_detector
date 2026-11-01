# myproject/api.py 逐行讲解

基线：`1210120748b3a05bd271bc79debcbed4fdc0c964`；SHA-256：`35a052e8f28c461a0eb7978dbd6c38ed798c0144d8d14d8c3731d3182f1616fb`；164 行。

源码与注释逐行并排。一个跨行表达式共用同一操作说明；这不代表各行独立执行。完整调用路径、输入输出轴和理论边界见 [METHOD.md](../METHOD.md)。未执行的 GPU 路径见 [验证边界](../VALIDATION.md)。

| 函数（含嵌套） | 原始行号 | 作用与契约入口 |
|---|---|---|
| `HallucinationDetector.__init__` | 18–34 | 构造器接收已加载模型/tokenizer，可注入读出器；task 与显式 method 必须与注入配置一致。成功创建轻量组合对象。 |
| `HallucinationDetector.from_pretrained` | 36–91 | classmethod from_pretrained 从本地模型路径构造完整检测器，支持参考路径、设备、注入读出和显式方法。由 CLI/用户调用，返回 cls 实例。 |
| `HallucinationDetector.measure` | 93–119 | measure 接收 prompt/response/source 和全局回答序号，返回 (对齐 measurement pack, 原始编码)，全程不写文件。 |
| `HallucinationDetector.fit` | 121–133 | fit 输入未标注例子的可迭代映射，仅给 R4 估计参考分布，返回 self 便于链式调用。缺字段 KeyError 向上传播。 |
| `HallucinationDetector.measure_responses` | 135–146 | measure_responses 对给定回答变体返回 [N_j,2] 原生 logp 矩阵及各自编码；两个世界使用匹配 SDPA，只是观测接口，不选修复、不判事实。 |
| `HallucinationDetector.save` | 148–150 | save 委托实际读出器保存：共享配置或 R4 references；不保存大模型、tokenizer、历史测量。 |
| `HallucinationDetector.predict` | 152–164 | predict 是用户入口，返回有效 token 的 ID、原文片段、半开字符 offset 和 float 分数。 |

| 行 | 原始代码 | 操作、目的与边界 |
|---|---|---|
| 1 | <code>"""Direct text interface: shared source readout by default, explicit legacy R4."""</code> | 文本公共入口：默认 shared，历史 R4 必须显式选择或提供 references。 |
| 2 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 3 | <code>import warnings</code> | warnings 用来临时抑制依赖加载提示，离开上下文恢复过滤器。 |
| 4 | <code>from collections.abc import Iterable, Mapping</code> | Iterable/Mapping 为 fit 和多回答接口提供类型说明，不自动执行运行时字段验证。 |
| 5 | <code>from pathlib import Path</code> | Path 统一模型/参考/配置的文件路径。 |
| 6 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 7 | <code>import numpy as np</code> | NumPy 用于多训练回答测量的拼接，不在 API 中实现数值推断。 |
| 8 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 9 | <code>from .detector import RECIPES, Detector</code> | 复用合法任务及历史 R4 读出类。 |
| 10 | <code>from .inputs import encode_response, encode_source</code> | 来源/回答的 token 化由 inputs 唯一实现。 |
| 11 | <code>from .measure import answer_rows, token_scores</code> | 复用原始 token 轴对齐及最终字符呈现，避免重分词。 |
| 12 | <code>from .shared import METHODS, SHARED_METHODS, SharedDetector</code> | 引入显式方法集合及默认共享读出；方法名校验先于加载大模型。 |
| 13 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 14 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 15 | <code>class HallucinationDetector:</code> | HallucinationDetector 组合冻结模型、tokenizer、观测器和读出器；调用者只传已有回答，不执行生成。 |
| 16 | <code>    """Measure existing responses and return per-token source-risk scores."""</code> | HallucinationDetector 组合冻结模型、tokenizer、观测器和读出器；调用者只传已有回答，不执行生成。 |
| 17 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 18 | <code>    def __init__(self, model, tokenizer, task: str = "QA",</code> | 构造器接收已加载模型/tokenizer，可注入读出器；task 与显式 method 必须与注入配置一致。成功创建轻量组合对象。 |
| 19 | <code>                 readout: Detector &#124; SharedDetector &#124; None = None, *, method: str &#124; None = None):</code> | 构造器接收已加载模型/tokenizer，可注入读出器；task 与显式 method 必须与注入配置一致。成功创建轻量组合对象。 |
| 20 | <code>        if task not in RECIPES:</code> | 拒绝未知任务，允许 QA/Summary/Data2txt。 |
| 21 | <code>            raise ValueError(f"task must be one of {tuple(RECIPES)}")</code> | 拒绝未知任务，允许 QA/Summary/Data2txt。 |
| 22 | <code>        if readout is not None and readout.task != task:</code> | 注入读出器不能用不同任务的参考配置。 |
| 23 | <code>            raise ValueError("Reference task differs from the requested task")</code> | 注入读出器不能用不同任务的参考配置。 |
| 24 | <code>        if readout is not None and readout.segmentation == "precomputed":</code> | 文本接口没有预计算分段边界，因此拒绝 precomputed，测量包 API 才能使用。 |
| 25 | <code>            raise ValueError("Precomputed segmentation requires measurement packs, not text inputs")</code> | 文本接口没有预计算分段边界，因此拒绝 precomputed，测量包 API 才能使用。 |
| 26 | <code>        if method is not None and method not in METHODS:</code> | 显式 method 若不在注册的简单元组中立即拒绝。 |
| 27 | <code>            raise ValueError(f'method must be one of {METHODS}')</code> | 显式 method 若不在注册的简单元组中立即拒绝。 |
| 28 | <code>        if readout is not None and method is not None and readout.method != method:</code> | 同时指定 method 和 readout 时禁止矛盾，避免悄悄覆盖用户选择。 |
| 29 | <code>            raise ValueError('Readout method differs from the requested method')</code> | 同时指定 method 和 readout 时禁止矛盾，避免悄悄覆盖用户选择。 |
| 30 | <code>        self.model = model</code> | 持有调用方模型引用，不复制或重新加载。 |
| 31 | <code>        self.tokenizer = tokenizer</code> | 持有对应 tokenizer；它必须与模型词表匹配，这里不额外验证。 |
| 32 | <code>        self.readout = readout if readout is not None else (</code> | 优先使用显式 readout；只有缺省时才实例化默认读出。 |
| 33 | <code>            Detector(task) if method == 'r4' else SharedDetector(task, method=method or 'shared'))</code> | method='r4' 选择 Detector；否则 method 或缺省 shared 选择 SharedDetector，不自动拟合。 |
| 34 | <code>        self._observer = None</code> | 观测器延迟创建，只有实际测量时才计算 Gram；当前对象复用时会缓存该引用。 |
| 35 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 36 | <code>    @classmethod</code> | classmethod from_pretrained 从本地模型路径构造完整检测器，支持参考路径、设备、注入读出和显式方法。由 CLI/用户调用，返回 cls 实例。 |
| 37 | <code>    def from_pretrained(cls, model_path: str &#124; Path, task: str = "QA",</code> | classmethod from_pretrained 从本地模型路径构造完整检测器，支持参考路径、设备、注入读出和显式方法。由 CLI/用户调用，返回 cls 实例。 |
| 38 | <code>                        references: str &#124; Path &#124; None = None, device: str = "cuda:0",</code> | classmethod from_pretrained 从本地模型路径构造完整检测器，支持参考路径、设备、注入读出和显式方法。由 CLI/用户调用，返回 cls 实例。 |
| 39 | <code>                        readout: Detector &#124; SharedDetector &#124; None = None,</code> | classmethod from_pretrained 从本地模型路径构造完整检测器，支持参考路径、设备、注入读出和显式方法。由 CLI/用户调用，返回 cls 实例。 |
| 40 | <code>                        method: str &#124; None = None):</code> | classmethod from_pretrained 从本地模型路径构造完整检测器，支持参考路径、设备、注入读出和显式方法。由 CLI/用户调用，返回 cls 实例。 |
| 41 | <code>        """Load a local model; shared needs no FIT, explicit references select legacy R4."""</code> | 不提供 references 的默认共享方法无需 FIT；提供 references 默认选择 R4。 |
| 42 | <code>        if task not in RECIPES:</code> | 在耗时模型加载前校验任务。 |
| 43 | <code>            raise ValueError(f"task must be one of {tuple(RECIPES)}")</code> | 在耗时模型加载前校验任务。 |
| 44 | <code>        if readout is not None and references is not None:</code> | readout 与 references 二选一，避免参考来源歧义。 |
| 45 | <code>            raise ValueError("Use either readout or references")</code> | readout 与 references 二选一，避免参考来源歧义。 |
| 46 | <code>        if method is not None and method not in METHODS:</code> | 加载前校验显式方法名称。 |
| 47 | <code>            raise ValueError(f'method must be one of {METHODS}')</code> | 加载前校验显式方法名称。 |
| 48 | <code>        if method in SHARED_METHODS and references is not None:</code> | 共享家族不使用 R4 分位数参考文件；不允许混用造成伪校准。 |
| 49 | <code>            raise ValueError('Shared detection does not use R4 reference distributions')</code> | 共享家族不使用 R4 分位数参考文件；不允许混用造成伪校准。 |
| 50 | <code>        if readout is None:</code> | 仅当未注入 readout 时按方法/参考路径自行选择。 |
| 51 | <code>            selected = method or ('r4' if references is not None else 'shared')</code> | 优先显式 method；否则 references 存在选 r4，不存在选 shared。 |
| 52 | <code>            if selected in SHARED_METHODS:</code> | 共享方法只创建 task/method 配置，不读训练参考。 |
| 53 | <code>                readout = SharedDetector(task, method=selected)</code> | 共享方法只创建 task/method 配置，不读训练参考。 |
| 54 | <code>            else:</code> | 非共享的已验证方法只能是 r4。 |
| 55 | <code>                reference_path = (Path(references) if references is not None</code> | 用户给路径就用该路径，否则取包内 references/<task>.npz；包资源位置基于本文件。 |
| 56 | <code>                                  else Path(__file__).with_name("references") / f"{task}.npz")</code> | 用户给路径就用该路径，否则取包内 references/<task>.npz；包资源位置基于本文件。 |
| 57 | <code>                readout = Detector.load(reference_path)</code> | 从冻结参考文件恢复 R4，文件缺失/损坏向上传播，不静默造新参考。 |
| 58 | <code>        if method is not None and readout.method != method:</code> | 注入或加载后的实际方法仍须符合显式 method。 |
| 59 | <code>            raise ValueError('Readout method differs from the requested method')</code> | 注入或加载后的实际方法仍须符合显式 method。 |
| 60 | <code>        if readout.task != task:</code> | 再次检查读出任务，覆盖加载文件内配置不符的情况。 |
| 61 | <code>            raise ValueError("Reference task differs from the requested task")</code> | 再次检查读出任务，覆盖加载文件内配置不符的情况。 |
| 62 | <code>        if readout.segmentation == "precomputed":</code> | 加载得到 precomputed 也不能用于文本接口。 |
| 63 | <code>            raise ValueError("Precomputed segmentation requires measurement packs, not text inputs")</code> | 加载得到 precomputed 也不能用于文本接口。 |
| 64 | <code>        from huggingface_hub.utils.tqdm import progress_bar_states</code> | 访问固定版本 Hub 的进度条状态字典；它是依赖具体接口，升级时需核验。 |
| 65 | <code>        from transformers.utils import logging</code> | 使用 Transformers 日志 API 临时压低加载日志。 |
| 66 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 67 | <code>        verbosity = logging.get_verbosity()</code> | 备份加载前 verbosity。 |
| 68 | <code>        progress_enabled = logging.is_progress_bar_enabled()</code> | 备份 Transformers 进度条是否启用。 |
| 69 | <code>        hub_progress = dict(progress_bar_states)</code> | 复制 Hub 进度条状态，之后原地恢复同一字典。 |
| 70 | <code>        with warnings.catch_warnings():</code> | 临时 warnings 上下文，离开时恢复 Python 过滤器。 |
| 71 | <code>            warnings.filterwarnings("ignore", module=r"(transformers&#124;huggingface_hub)(\..*)?")</code> | 只抑制这两个依赖模块匹配的 warning，不吞异常。 |
| 72 | <code>            try:</code> | 用 try/finally 保证即使加载失败也恢复日志状态。 |
| 73 | <code>                logging.set_verbosity_error()</code> | 加载期间只显示 error 并关闭 Transformers 进度条。 |
| 74 | <code>                logging.disable_progress_bar()</code> | 加载期间只显示 error 并关闭 Transformers 进度条。 |
| 75 | <code>                from transformers import AutoTokenizer</code> | 延迟导入 AutoTokenizer，文本入口之外不必先初始化 tokenizer。 |
| 76 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 77 | <code>                from .model import load_model</code> | 模型加载委托 model.load_model，以共享 BF16/冻结约定。 |
| 78 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 79 | <code>                tokenizer = AutoTokenizer.from_pretrained(</code> | 只从本地读取 fast tokenizer；offset_mapping 是后续字符对齐的必要接口。 |
| 80 | <code>                    model_path, use_fast=True, local_files_only=True,</code> | 只从本地读取 fast tokenizer；offset_mapping 是后续字符对齐的必要接口。 |
| 81 | <code>                )</code> | 只从本地读取 fast tokenizer；offset_mapping 是后续字符对齐的必要接口。 |
| 82 | <code>                model = load_model(model_path, device)</code> | 加载冻结模型并放到用户设备；没有自动找远端实例。 |
| 83 | <code>            finally:</code> | 无论成功失败均恢复原日志级别。 |
| 84 | <code>                logging.set_verbosity(verbosity)</code> | 无论成功失败均恢复原日志级别。 |
| 85 | <code>                if progress_enabled:</code> | 根据原开关恢复 Transformers 进度条，而非一律启用。 |
| 86 | <code>                    logging.enable_progress_bar()</code> | 根据原开关恢复 Transformers 进度条，而非一律启用。 |
| 87 | <code>                else:</code> | 根据原开关恢复 Transformers 进度条，而非一律启用。 |
| 88 | <code>                    logging.disable_progress_bar()</code> | 根据原开关恢复 Transformers 进度条，而非一律启用。 |
| 89 | <code>                progress_bar_states.clear()</code> | 恢复 Hub 状态字典原内容，避免加载过程全局副作用留存。 |
| 90 | <code>                progress_bar_states.update(hub_progress)</code> | 恢复 Hub 状态字典原内容，避免加载过程全局副作用留存。 |
| 91 | <code>        return cls(model, tokenizer, task, readout, method=method)</code> | 通过构造器统一完成最终校验并返回模型/tokenizer/readout 的组合。 |
| 92 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 93 | <code>    def measure(self, prompt: str, response: str, source: str,</code> | measure 接收 prompt/response/source 和全局回答序号，返回 (对齐 measurement pack, 原始编码)，全程不写文件。 |
| 94 | <code>                 answer_index: int = 0) -&gt; tuple[dict, dict]:</code> | measure 接收 prompt/response/source 和全局回答序号，返回 (对齐 measurement pack, 原始编码)，全程不写文件。 |
| 95 | <code>        """Return aligned native observations and original token metadata, without I/O."""</code> | measure 接收 prompt/response/source 和全局回答序号，返回 (对齐 measurement pack, 原始编码)，全程不写文件。 |
| 96 | <code>        encoded_source = encode_source(self.tokenizer, prompt, source)</code> | 来源必须在 prompt 中唯一出现；构造 present/absent 提示及来源 mask。 |
| 97 | <code>        encoded_response = encode_response(self.tokenizer, response)</code> | 保留已有回答原 token IDs 和字符 offset，并作默认标点分段。 |
| 98 | <code>        if not encoded_response["answer_ids"]:</code> | 拒绝分词后没有 token 的回答，保证底层拼接/状态推断非空。 |
| 99 | <code>            raise ValueError("response must contain at least one token")</code> | 拒绝分词后没有 token 的回答，保证底层拼接/状态推断非空。 |
| 100 | <code>        if self.readout.segmentation != "punct":</code> | 非 punct 配置先重新计算依赖分段；构造器已排除了 precomputed。 |
| 101 | <code>            from .attention import attention_profiles</code> | 延迟加载 attention 路由和合法切点/分段算法，仅此可选路径需要。 |
| 102 | <code>            from .segmentation import legal_atoms, predictive_segment, segment</code> | 延迟加载 attention 路由和合法切点/分段算法，仅此可选路径需要。 |
| 103 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 104 | <code>            graph = attention_profiles(self.model, encoded_source["prompt_with_source"],</code> | 对原提示与实际回答做 post-token attention 观测；其位置与预测 logp 行相差一个 token。 |
| 105 | <code>                                       encoded_response["answer_ids"])</code> | 对原提示与实际回答做 post-token attention 观测；其位置与预测 logp 行相差一个 token。 |
| 106 | <code>            atoms, _ = legal_atoms(encoded_response)</code> | legal_atoms 排除词内和 offset 重叠处的非法切点，不使用标签。 |
| 107 | <code>            fitted = (predictive_segment(graph["fit"], graph["hold"], graph["available"], atoms)</code> | dependency_predictive 用 head 奇偶两视图选惩罚；普通 dependency 只对 fit 视图无监督分段，两视图并非独立样本。 |
| 108 | <code>                      if self.readout.segmentation == "dependency_predictive" else</code> | dependency_predictive 用 head 奇偶两视图选惩罚；普通 dependency 只对 fit 视图无监督分段，两视图并非独立样本。 |
| 109 | <code>                      segment(graph["fit"], graph["available"], atoms))</code> | dependency_predictive 用 head 奇偶两视图选惩罚；普通 dependency 只对 fit 视图无监督分段，两视图并非独立样本。 |
| 110 | <code>            cuts = fitted["cuts"]</code> | 取优化得到的原 token 边界数组。 |
| 111 | <code>            encoded_response["units"] = [{"start": int(a), "stop": int(b)}</code> | 将相邻切点变为半开 [start,stop) units，替换默认标点分段，仍不重分词。 |
| 112 | <code>                                         for a, b in zip(cuts[:-1], cuts[1:])]</code> | 将相邻切点变为半开 [start,stop) units，替换默认标点分段，仍不重分词。 |
| 113 | <code>        if self._observer is None:</code> | 首次测量才创建 Observer，并缓存投影 Gram 供后续回答复用；模型权重若被外部修改，缓存不会自动失效。 |
| 114 | <code>            from .observer import Observer</code> | 首次测量才创建 Observer，并缓存投影 Gram 供后续回答复用；模型权重若被外部修改，缓存不会自动失效。 |
| 115 | <code> </code> | 首次测量才创建 Observer，并缓存投影 Gram 供后续回答复用；模型权重若被外部修改，缓存不会自动失效。 |
| 116 | <code>            self._observer = Observer(self.model)</code> | 首次测量才创建 Observer，并缓存投影 Gram 供后续回答复用；模型权重若被外部修改，缓存不会自动失效。 |
| 117 | <code>        observed = self._observer.measure(encoded_source, encoded_response,</code> | 观测四种来源/历史条件；只有 conditional 开启完整词表收集。默认 shared 仍会测量其不用的 local/route。 |
| 118 | <code>                                          full_distribution=self.readout.method == 'conditional')</code> | 观测四种来源/历史条件；只有 conditional 开启完整词表收集。默认 shared 仍会测量其不用的 local/route。 |
| 119 | <code>        return answer_rows(observed, encoded_response, answer_index), encoded_response</code> | answer_rows 过滤无 offset/特殊 token、严格对齐原位置，连同完整编码返回。 |
| 120 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 121 | <code>    def fit(self, examples: Iterable[Mapping[str, str]]):</code> | fit 输入未标注例子的可迭代映射，仅给 R4 估计参考分布，返回 self 便于链式调用。缺字段 KeyError 向上传播。 |
| 122 | <code>        """Fit references from unlabeled examples with prompt/response/source strings."""</code> | fit 输入未标注例子的可迭代映射，仅给 R4 估计参考分布，返回 self 便于链式调用。缺字段 KeyError 向上传播。 |
| 123 | <code>        if isinstance(self.readout, SharedDetector):</code> | 固定 shared 不需要 FIT，主动拒绝误用，防止误称用新数据训练了默认方法。 |
| 124 | <code>            raise ValueError('The fixed shared method needs no FIT; select method=r4 to fit R4 references')</code> | 固定 shared 不需要 FIT，主动拒绝误用，防止误称用新数据训练了默认方法。 |
| 125 | <code>        batches = [self.measure(example["prompt"], example["response"], example["source"],</code> | 逐例测量，并用 enumerate 分配不同全局 answer_index；取测量包而非文本编码。 |
| 126 | <code>                                 index)[0]</code> | 逐例测量，并用 enumerate 分配不同全局 answer_index；取测量包而非文本编码。 |
| 127 | <code>                   for index, example in enumerate(examples)]</code> | 逐例测量，并用 enumerate 分配不同全局 answer_index；取测量包而非文本编码。 |
| 128 | <code>        if not batches:</code> | 没有训练例子则明确拒绝，不保留伪空参考。 |
| 129 | <code>            raise ValueError("fit requires at least one example")</code> | 没有训练例子则明确拒绝，不保留伪空参考。 |
| 130 | <code>        pack = {name: np.concatenate([batch[name] for batch in batches])</code> | 按第一包字段逐项沿 token 行拼接，默认假设各包 schema 一致。 |
| 131 | <code>                for name in batches[0]}</code> | 按第一包字段逐项沿 token 行拼接，默认假设各包 schema 一致。 |
| 132 | <code>        self.readout.fit(pack)</code> | 真正参考拟合委托 readout.fit，不在 API 复制公式。 |
| 133 | <code>        return self</code> | 返回同一对象，其 readout.references 已被更改。 |
| 134 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 135 | <code>    def measure_responses(self, prompt: str, responses: Iterable[str], source: str):</code> | measure_responses 对给定回答变体返回 [N_j,2] 原生 logp 矩阵及各自编码；两个世界使用匹配 SDPA，只是观测接口，不选修复、不判事实。 |
| 136 | <code>        """Replay supplied response variants; return raw logp matrices and encodings.</code> | measure_responses 对给定回答变体返回 [N_j,2] 原生 logp 矩阵及各自编码；两个世界使用匹配 SDPA，只是观测接口，不选修复、不判事实。 |
| 137 | <code> </code> | measure_responses 对给定回答变体返回 [N_j,2] 原生 logp 矩阵及各自编码；两个世界使用匹配 SDPA，只是观测接口，不选修复、不判事实。 |
| 138 | <code>        Both source conditions use matched SDPA. This observation API does not</code> | measure_responses 对给定回答变体返回 [N_j,2] 原生 logp 矩阵及各自编码；两个世界使用匹配 SDPA，只是观测接口，不选修复、不判事实。 |
| 139 | <code>        score factuality or select repairs, and preserves each raw response axis.</code> | measure_responses 对给定回答变体返回 [N_j,2] 原生 logp 矩阵及各自编码；两个世界使用匹配 SDPA，只是观测接口，不选修复、不判事实。 |
| 140 | <code>        """</code> | measure_responses 对给定回答变体返回 [N_j,2] 原生 logp 矩阵及各自编码；两个世界使用匹配 SDPA，只是观测接口，不选修复、不判事实。 |
| 141 | <code>        from .counterfactual import observe_responses</code> | 使用 counterfactual.observe_responses 共用前缀缓存调度。 |
| 142 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 143 | <code>        encoded_source = encode_source(self.tokenizer, prompt, source)</code> | 两世界的来源编码只做一次。 |
| 144 | <code>        encoded = [encode_response(self.tokenizer, text) for text in responses]</code> | 每个回答独立分词，变体的 token 轴不被强行拼成相同长度。 |
| 145 | <code>        return observe_responses(self.model, encoded_source,</code> | 把各自原始回答 IDs 交给观测器，并把编码一起返回供调用方映射 offsets。 |
| 146 | <code>                                 [row['answer_ids'] for row in encoded]), encoded</code> | 把各自原始回答 IDs 交给观测器，并把编码一起返回供调用方映射 offsets。 |
| 147 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 148 | <code>    def save(self, path: str &#124; Path) -&gt; None:</code> | save 委托实际读出器保存：共享配置或 R4 references；不保存大模型、tokenizer、历史测量。 |
| 149 | <code>        """Save the shared configuration or the explicit legacy R4 references."""</code> | save 委托实际读出器保存：共享配置或 R4 references；不保存大模型、tokenizer、历史测量。 |
| 150 | <code>        self.readout.save(Path(path))</code> | save 委托实际读出器保存：共享配置或 R4 references；不保存大模型、tokenizer、历史测量。 |
| 151 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 152 | <code>    def predict(self, prompt: str, response: str, source: str) -&gt; list[dict]:</code> | predict 是用户入口，返回有效 token 的 ID、原文片段、半开字符 offset 和 float 分数。 |
| 153 | <code>        """Return token_id/text/start/end/score; offsets index response characters."""</code> | predict 是用户入口，返回有效 token 的 ID、原文片段、半开字符 offset 和 float 分数。 |
| 154 | <code>        if isinstance(self.readout, Detector) and self.readout.references is None:</code> | R4 未 FIT/加载参考时先报错；即使 response='' 也先检查参考，此优先级是当前真实行为。 |
| 155 | <code>            raise ValueError("Fit reference examples or load references before predict")</code> | R4 未 FIT/加载参考时先报错；即使 response='' 也先检查参考，此优先级是当前真实行为。 |
| 156 | <code>        if not response:</code> | 空字符串回答返回空列表，不做来源校验或模型测量；仅有空白但有 token 的回答仍按测量路径处理。 |
| 157 | <code>            return []</code> | 空字符串回答返回空列表，不做来源校验或模型测量；仅有空白但有 token 的回答仍按测量路径处理。 |
| 158 | <code>        pack, encoded = self.measure(prompt, response, source)</code> | 执行统一观测与原始 token 轴对齐。 |
| 159 | <code>        if self.readout.method == 'proof_revision':</code> | proof_revision 才从来源/回答编译额外 proof pack；不是默认方法的一部分，其证明规则应单独审计。 |
| 160 | <code>            from .revision import proof_pack</code> | proof_revision 才从来源/回答编译额外 proof pack；不是默认方法的一部分，其证明规则应单独审计。 |
| 161 | <code> </code> | proof_revision 才从来源/回答编译额外 proof pack；不是默认方法的一部分，其证明规则应单独审计。 |
| 162 | <code>            pack = proof_pack(pack, source, response, encoded['offsets'])</code> | proof_revision 才从来源/回答编译额外 proof pack；不是默认方法的一部分，其证明规则应单独审计。 |
| 163 | <code>        scores = self.readout.score(pack)</code> | 用选择的读出器计算最终有效 token 分数，不在 API 改排序。 |
| 164 | <code>        return token_scores(pack, encoded, scores)</code> | 按编码 offsets 切原 response 字符串，避免 decode 后文本差异污染展示。 |
