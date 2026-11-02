# FRED 原问题的 label-free 回连：本次 exact 路线恢复 0

2026-11-03。完整、独立的 metadata-only 扫描已完成并封存。固定官方 RagBench revision 97808f3e5fd16ede40bbff6c2949af8139b2eb7b 的 FinQA/TATQA 三个 split，只投影 id/question/documents 三列；对 FRED 全部 162+130 个原答的来源，严格按 documents[0] 原 UTF-8 字节回连。结果为 **0 个匹配、0 个可恢复问题，292 答/252 个 source 全部保留为 unknown**。没有 fuzzy join、strip、首条选择或利用回答消歧。

| 原组 | 原回答 | 原来源 | 扫描 upstream rows | 恢复问题的原答 | 未知原答 |
|---|---:|---:|---:|---:|---:|
| FRED FinQA | 162 | 142 | 16,562 | 0 | 162 |
| FRED TATQA | 130 | 110 | 33,104 | 0 | 130 |
| 合计 | 292 | 252 | 49,666 | 0 | 292 |

未知原因均为 no_exact_first_document_match。这里否定的是**该固定 upstream snapshot、该已冻结 first-document 原字节规则**的回连，不证明所有历史版本或未发布中间 crosswalk 都不存在。不能从 0 推出原问题永远不可恢复，也不能假定问题已合法恢复以推进自然 QA。

## 规则在列值读取前冻结

[attempt-v2/plan.json](attempt-v2/plan.json)、[freeze.json](attempt-v2/freeze.json) 先固定本地 source-only 投影、完整原 id 顺序与 source SHA、官方文件 inventory/revision/LFS SHA、转换脚本、脚本身份和精确规则，再开始读取 Parquet 选定列值。规则为：

    SHA256(exact UTF-8 local source) == SHA256(documents[0])
    AND local source == documents[0]
    AND matched rows have exactly one distinct nonempty question byte string.

重复文档对应多个不同问题应 unknown；重复 row 但问题字节相同才允许唯一 Q。所有匹配 row id/路径/ordinal 都保留。实际本次第一项匹配数就是 0，没有进入“从多个 Q 中挑一个”的阶段。

本地 inference JSON 流只解码 id/source_id/source。prompt、response 及其他未知值通过冻结的 lexical-skip helper 消耗而不 value-decode；并未读取编辑前回答或原生成答案。官方 Parquet 只对 id/question/documents 调用 iter_batches(columns=...) 并检查实际 batch schema。其他列不会 materialize 为 Arrow/Python 值；没有读取其统计值或用于筛选。HTTP 范围和 footer 可能包含未解码的邻近编码字节，不能冒称网络上未接收到任何其他列的物理字节。

## 官方 source transform 审查

[source-pipeline-inspection.json](source-pipeline-inspection.json) 保留官方三个 data_preparation 脚本的固定 revision、SHA、相关行与 AST 检查。固定版本为 afdab2c54075ade8c890331da099cc9b25d1f6a5：

- [convert_format.py](https://github.com/pegasi-ai/fine-grained-editting/blob/afdab2c54075ade8c890331da099cc9b25d1f6a5/data_preparation/convert_format.py) 第 38 行取 ast.literal_eval(row["documents"])[0]，再原样插入 reference wrapper；最后只导出 completion/prompt。SHA256：99dd66cb4e52a3a07319d78393ecd644958c0f9bdeed2f78c56a06f78c51a6e2。
- [insert_errors.py](https://github.com/pegasi-ai/fine-grained-editting/blob/afdab2c54075ade8c890331da099cc9b25d1f6a5/data_preparation/insert_errors.py) 第 455 行也只读取 documents[0]，用于 error insertion 的参考。没有找到对 documents 字段的赋值。
- [verify_responses.py](https://github.com/pegasi-ai/fine-grained-editting/blob/afdab2c54075ade8c890331da099cc9b25d1f6a5/data_preparation/verify_responses.py) 所查代码没有 documents 下标读取或赋值；它核查/修正响应标记并保存 dataframe。没有建立可额外变换 source 的官方规则。

本地 fred_text 提取只按固定 wrapper 切割 source，strip 只用于“是否空串”的断言，没有把 strip 后的 source 返回。因此没有官方依据在本次 exact join 上追加去空白、大小写/数字/标点/Unicode 改写或其他“增匹配率”的归一化。所查代码也未绑定原 RagBench→中间输入 CSV 的历史 dataset revision 或 native row crosswalk。现有证据无法确定 0 匹配的具体上游原因；没有据此猜一个变换。

[FRED 原论文 §2](https://arxiv.org/html/2507.20930v2#S2) 确认来自 RagBench；[官方 RagBench 固定 schema](https://huggingface.co/datasets/galileo-ai/ragbench/blob/97808f3e5fd16ede40bbff6c2949af8139b2eb7b/README.md) 提供上述三字段。原 FinQA/TAT-QA 的 question/native IDs 虽存在，FRED 最终 CSV 没有保留它们。本轮没有将无名 CSV 行号猜作 native ID，也没有改用原未编辑 R 找题目。

## 文件、网络与完整性

成功运行位于 attempt-v2：现有系统 Python 3.13.9/pyarrow 环境，未安装依赖或改变 CPU/GPU 环境。六个 Parquet 的三字段投影完整保存为 projections/*.jsonl.gz，共 38,214,196 bytes；总 bounded-range 接收 55,271,218 bytes，官方原 Parquet inventory 合计 154,432,437 bytes。投影各文件都有本地 compressed/uncompressed SHA256、完整 row count 和每个接收 range 的 offset/length/SHA。原文件的 upstream_lfs_sha256 来自官方 pinned tree；**未完整下载原 Parquet，故不称本地验证过其 whole-file hash**。

运行约 173.79 秒。0 模型、0 tokenizer、0 QA/GPU、0 target gold、0 answer/response/label 列值解码；未修改既有 inputs、predictions 或 gold。最初匿名直连在列值读取前失败，根目录 failure.json 与第一份脚本/投影保留。独立 attempt-v2 明确屏蔽 Authorization、Cookie 与 netrc，允许系统 transport/proxy/CA；没有使用 HF 登录、用户 token、私有凭据或保存 signed URL。没有对科学规则作失败后放宽。

成功结果 [result.json](attempt-v2/result.json) SHA256：4850c6adf71bcb286a2fd3105292d0c4e749590b7f1d8c9f2217999af4b31492；成功 seal SHA256：87eb0246a444e7f0a76cc9099e5ada61d09cd548f1acf69acddb5b1d77847831。全 292 个原 id 与 unknown 原因在两个 recovery.json 文件中；本次没有可插入模型的新问题。即便未来唯一 Q 回连成功，也只是相对于明确 upstream 版本的 metadata 证据，不自动恢复原 generator prompt、全部 documents、消息角色或问题在当前保留来源上的可答性。

## 下一自然入口只建议一项：锚与修复集合的准入库存

H011 已知 person-slot 的桥接不能代替自然 136 个 occurrence 的准入。下一项只建议做**锚/修复歧义库存**，不直接跑 136 个自由 QA，不因 7B 可用而搜模型或 prompt：记录哪个原问题/事件/单位/时间条件有可核查 provenance，哪个只是 R 中可能错误的另一个参数；对没有唯一锚的关系，保留所有已证实可行的支持修复及 unknown，不把其中某个 person/event 改动强行认定为唯一错误位置。无法证明修复集合完整时，也只报 incomplete/unknown，不把有限枚举包装成新的覆盖证书检测器。

这一区分有经典先例：[Reiter 的诊断理论](https://www.sciencedirect.com/science/article/pii/0004370287900622) 将不一致观察对应到可能异常组件；[Arenas、Bertossi、Chomicki 的 consistent-query-answer 论文（作者原文）](https://www.cs.toronto.edu/~marenas/publications/pods99.pdf) 通过所有最小 repair 考虑稳定答案。它们不是“找到最可能编辑 token”的保证。这里借用的是问题表述，不能把已知数据库约束/完备性定理直接套在有漏检、别名、算术和非断言的自然文本上，也不是新算法声明。

对 S={wins(X,A),wins(Y,B)}、R=wins(X,B)，person 与 event 两条支持修复都应被保留；若有可验证的原 Q=“谁赢 B”，它会规定 event 锚并缩小允许修复域，而不是从 S、R 神奇恢复原编辑原因。现有 [QAGS](https://aclanthology.org/2020.acl-main.450/) 和 [FEQA](https://aclanthology.org/2020.acl-main.454/) 的 generated-Q 一致性不等于这个 original-Q provenance。该库存的可证伪点是“声称唯一可答的自然 proposal 实际存在另一同样合法锚/修复”，而不是再造 toy 100% 或对不同方法挑较佳分数。完整自然预测/标签评价仍需另行冻结；本轮不执行。
