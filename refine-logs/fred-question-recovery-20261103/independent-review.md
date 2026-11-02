# FRED source-only question recovery：独立只读静审

结论：**应用层字段投影、exact source join和歧义保留合同通过静态检查；未发现用R/answer/labels做连接或消歧的路径。** 本审查只读脚本、plan/freeze、词法skip helper与source-pipeline-inspection；没有打开实际local-source-projections、Parquet projections或真实response/gold，没有重跑恢复、测试、模型或tokenizer。

v2脚本 SHA256为 `07c086863bca9853bdb90811f048a03d0291896f258a0c76497c8dc4a5f89927`，与plan相符；词法helper当前SHA与plan一致，plan字节hash与freeze一致。此处未独立重hash真实projection，因此不能把本静审称作所有数据payload的独立重验。

- 本地 SourceJSON 覆盖父object方法，只对 `id/source_id/source` 调scalar；其他字段走skip。helper的未知字符串用 `string(False)`消耗转义边界，不调用json.loads解释其值；未知数组/对象递归跳过。字段名会解析，整份输入也会流式哈希，但没有形成或输出R/label值对象。helper导入有main guard，不会执行其旧Hallu样本读取程序。
- 上游只调用 `iter_batches(columns=['id','question','documents'], use_threads=False)`，并在batch schema和每行字典再次要求恰这三列；没有to_pandas、整表to_pylist或其他列投影。Arrow21官方文档支持按列选择批次；这是逻辑列物化边界，不是网络字节级隔离保证。[ParquetFile21](https://arrow.apache.org/docs/21.0/python/generated/pyarrow.parquet.ParquetFile.html)
- 每个来源先比UTF-8 SHA，再比完整原字符串，固定只取documents[0]；没有strip/规范化/fuzzy/ID-first/response匹配。全部三个split均处理，batch行数需与文件metadata一致；所有本地162/130原ID保留。匹配行及不同question字节全部保留，只在恰一个非空question时恢复；strip仅用于判断是否全空白，保留的question及其SHA不被trim。零匹配/多问句都unknown，不选第一条。
- plan/freeze写在上游列值处理前；上游完整Parquet LFS SHA是官方固定仓库声明，脚本没有下载/本地验证整文件SHA。逻辑三列projection的压缩与解压内容分别本地hash，不能因此升级成原Parquet完整身份已实测。恢复也不宣称得到原generator的完整prompt或全部documents。

保留一项表述边界：Parquet footer可能含未选列的statistics/原始min-max表示，HTTP范围也可能携带邻近列字节；脚本没有访问statistics.min/max或把未选列变成逻辑row数组。应将结论写成“未物化、未输出或使用未选列的逻辑值”，不要扩张成“从未接收/解析任何可能包含answer信息的物理字节”。官方统计API区分logical min/max与physical min_raw/max_raw；本静审没有检查具体文件是否携带这些统计。[Statistics21](https://arrow.apache.org/docs/21.0/python/generated/pyarrow.parquet.Statistics.html)

作者报告六split/49666行得到零exact documents[0]匹配，全部292原response ID仍unknown；本审查未读结果数据独立复数。这个负结果只适用于已固定revision与exact第一文档合同，不能推成原question绝对不可恢复、来源不属于任何RagBench版本或一般关系定位不可能。三份官方preparation脚本中未见documents赋值/strip，仅限定已检查代码范围；尚未验证原generator实际RagBench revision和中间CSV来源。因此不应据零匹配追fuzzy、改变源规范化或用R选择候选问题。
