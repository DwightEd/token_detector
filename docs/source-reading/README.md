# 代码讲解导航与真实进度

建议先打开[公式与机制交互指南](VISUAL_GUIDE.html)，拖动滑条理解观测与共享，再读[完整公式与证明](THEORY.html)和[论文主张](PAPER_STORY.html)。[整个方法](METHOD.html)与[逐行阅读器](READING.html)也有浏览器版。下载后打开HTML即可离线排版，不再需要直接打开Markdown看LaTeX。

GitHub不直接运行HTML；在线可读[公式原文](THEORY.md)、[方法原文](METHOD.md)和[论文叙事原文](PAPER_STORY.md)。下面每个已讲解文件也有Markdown版。

当前完成主库全部21个Python文件、2421行、128个函数/嵌套定义的具体讲解，覆盖默认推断完整路径、R4、sequence/witness、条件分布、依赖分段、proofs/revision、命令入口及匹配SDPA观测。每行旁边给出真实操作、目的、轴/精度/状态副作用。多行表达式共用一项完整操作说明；空白行标为无运行效果。源码SHA绑定1210120748b3a05bd271bc79debcbed4fdc0c964。

**全项目尚未全部讲完。** 原始98个Python文件共15610行、636个定义，主库以外剩余77文件/508定义待逐行讲解。新增两测试文件与渲染工具另列在inventory.current_additions，保持待逐行解释状态。没有把自动AST清单冒充逐行完成。非Python资源没有伪造函数覆盖。

科学状态：[论文对照](../../refine-logs/theory-grounding-20261102/LITERATURE.md)、[当前完整方案](../../refine-logs/theory-grounding-20261102/FINAL_PROPOSAL.md)、[独立审查摘要](../../refine-logs/theory-grounding-20261102/REVIEW_SUMMARY.md)。[用户已确认](../../refine-logs/theory-grounding-20261102/USER_CONSTRAINTS.md)坚持无监督，真值只用于最终评价。已有理论推断不等于事实标签识别；没有新的自然数据效果或顶会就绪结论。

[VALIDATION.md](VALIDATION.md)记录实际CPU检查与未执行路径。`SOURCE_INVENTORY.json`可机读；`SOURCE_INVENTORY_initial.json`保留初始待处理范围。后续修改源码前先核对hash，不能沿用过期行号。

重建公式页先安装可选docs依赖 `python -m pip install -e '.[docs]'`，运行 `python tools/render_math_guide.py`；逐行页运行 `python tools/render_source_reading.py`。渲染器只排版已写内容并检查源码hash，不自动升级讲解覆盖。MathJax3.2.2本地资源及许可证随文档保存；这是固定离线版本，不声称是最新版本。交互页数据调用生产数值函数预计算，全部是构造例子，不读取任务标注或执行LM。

| 原始文件 | 行数 | 函数/嵌套定义 | 讲解状态 |
|---|---:|---:|---|
| [myproject/__init__.py](files/myproject___init__.md) | 6 | 0 | 已逐行讲解 |
| [myproject/__main__.py](files/myproject___main__.md) | 3 | 0 | 已逐行讲解 |
| [myproject/api.py](files/myproject_api.md) | 164 | 7 | 已逐行讲解 |
| [myproject/attention.py](files/myproject_attention.md) | 56 | 2 | 已逐行讲解 |
| [myproject/counterfactual.py](files/myproject_counterfactual.md) | 54 | 2 | 已逐行讲解 |
| [myproject/detector.py](files/myproject_detector.md) | 200 | 14 | 已逐行讲解 |
| [myproject/distribution.py](files/myproject_distribution.md) | 106 | 7 | 已逐行讲解 |
| [myproject/graph.py](files/myproject_graph.md) | 37 | 3 | 已逐行讲解 |
| [myproject/inputs.py](files/myproject_inputs.md) | 71 | 3 | 已逐行讲解 |
| [myproject/measure.py](files/myproject_measure.md) | 71 | 5 | 已逐行讲解 |
| [myproject/model.py](files/myproject_model.md) | 68 | 5 | 已逐行讲解 |
| [myproject/observer.py](files/myproject_observer.md) | 39 | 2 | 已逐行讲解 |
| [myproject/pooling.py](files/myproject_pooling.md) | 153 | 11 | 已逐行讲解 |
| [myproject/proofs.py](files/myproject_proofs.md) | 554 | 23 | 已逐行讲解 |
| [myproject/revision.py](files/myproject_revision.md) | 211 | 9 | 已逐行讲解 |
| [myproject/routing.py](files/myproject_routing.md) | 68 | 6 | 已逐行讲解 |
| [myproject/run.py](files/myproject_run.md) | 47 | 1 | 已逐行讲解 |
| [myproject/segmentation.py](files/myproject_segmentation.md) | 136 | 11 | 已逐行讲解 |
| [myproject/sequence.py](files/myproject_sequence.md) | 218 | 8 | 已逐行讲解 |
| [myproject/shared.py](files/myproject_shared.md) | 105 | 7 | 已逐行讲解 |
| [myproject/source.py](files/myproject_source.md) | 54 | 2 | 已逐行讲解 |
| `experiments/__init__.py` | 1 | 0 | 待讲解 |
| `experiments/__main__.py` | 3 | 0 | 待讲解 |
| `experiments/assertion_pilot.py` | 348 | 13 | 待讲解 |
| `experiments/binding_dependency_pilot.py` | 295 | 7 | 待讲解 |
| `experiments/binding_transfer.py` | 251 | 3 | 待讲解 |
| `experiments/case_audit.py` | 94 | 2 | 待讲解 |
| `experiments/conditional_dependencies.py` | 72 | 3 | 待讲解 |
| `experiments/constraint_pilot.py` | 383 | 12 | 待讲解 |
| `experiments/contextual_probe.py` | 271 | 8 | 待讲解 |
| `experiments/counterfactual_repairs.py` | 135 | 4 | 待讲解 |
| `experiments/data.py` | 123 | 14 | 待讲解 |
| `experiments/datasets.py` | 371 | 13 | 待讲解 |
| `experiments/delta_local_integrity_20261028.py` | 452 | 18 | 待讲解 |
| `experiments/delta_local_run.py` | 356 | 10 | 待讲解 |
| `experiments/delta_observe.py` | 33 | 1 | 待讲解 |
| `experiments/dependency_probe.py` | 270 | 8 | 待讲解 |
| `experiments/evaluate.py` | 371 | 11 | 待讲解 |
| `experiments/export_hallurag.py` | 63 | 2 | 待讲解 |
| `experiments/fit_ablation.py` | 149 | 4 | 待讲解 |
| `experiments/fullgap_dataset.py` | 849 | 30 | 待讲解 |
| `experiments/hidden.py` | 60 | 1 | 待讲解 |
| `experiments/inputs.py` | 58 | 3 | 待讲解 |
| `experiments/local_evaluation.py` | 61 | 1 | 待讲解 |
| `experiments/loo_pooling.py` | 112 | 2 | 待讲解 |
| `experiments/lumina_dataset.py` | 417 | 13 | 待讲解 |
| `experiments/measure.py` | 58 | 2 | 待讲解 |
| `experiments/native_dataset.py` | 581 | 9 | 待讲解 |
| `experiments/native_response.py` | 258 | 7 | 待讲解 |
| `experiments/pipeline.py` | 195 | 6 | 待讲解 |
| `experiments/pooling_run.py` | 212 | 7 | 待讲解 |
| `experiments/prevalence_probe.py` | 180 | 8 | 待讲解 |
| `experiments/provenance.py` | 45 | 2 | 待讲解 |
| `experiments/relational_transfer.py` | 109 | 3 | 待讲解 |
| `experiments/research/__init__.py` | 1 | 0 | 待讲解 |
| `experiments/research/__main__.py` | 3 | 0 | 待讲解 |
| `experiments/research/assertions.py` | 364 | 16 | 待讲解 |
| `experiments/research/binding_dependency.py` | 150 | 5 | 待讲解 |
| `experiments/research/binding_witnesses.py` | 151 | 7 | 待讲解 |
| `experiments/research/constraint_interaction.py` | 226 | 14 | 待讲解 |
| `experiments/research/constraints.py` | 54 | 1 | 待讲解 |
| `experiments/research/contextual_alignment.py` | 89 | 3 | 待讲解 |
| `experiments/research/delta_local.py` | 69 | 4 | 待讲解 |
| `experiments/research/dependence.py` | 51 | 3 | 待讲解 |
| `experiments/research/fingerprints.py` | 87 | 5 | 待讲解 |
| `experiments/research/hierarchy.py` | 76 | 3 | 待讲解 |
| `experiments/research/lumina.py` | 206 | 10 | 待讲解 |
| `experiments/research/relational.py` | 86 | 2 | 待讲解 |
| `experiments/research/relational_shared.py` | 142 | 3 | 待讲解 |
| `experiments/research/run.py` | 94 | 4 | 待讲解 |
| `experiments/research/sequence_controls.py` | 90 | 4 | 待讲解 |
| `experiments/research/spans.py` | 61 | 5 | 待讲解 |
| `experiments/research/state.py` | 128 | 10 | 待讲解 |
| `experiments/research/state_partition.py` | 52 | 3 | 待讲解 |
| `experiments/research/table_bindings.py` | 129 | 7 | 待讲解 |
| `experiments/run.py` | 374 | 2 | 待讲解 |
| `experiments/source_write_integrity_20261028.py` | 261 | 13 | 待讲解 |
| `experiments/source_write_observe.py` | 123 | 8 | 待讲解 |
| `experiments/source_write_run.py` | 246 | 9 | 待讲解 |
| `experiments/span_markup.py` | 61 | 3 | 待讲解 |
| `experiments/span_pilot.py` | 279 | 13 | 待讲解 |
| `experiments/span_resolution.py` | 292 | 10 | 待讲解 |
| `experiments/transfer_diagnostics.py` | 285 | 5 | 待讲解 |
| `experiments/verify.py` | 79 | 3 | 待讲解 |
| `experiments/write_competition_integrity_20261028.py` | 258 | 13 | 待讲解 |
| `tests/__init__.py` | 1 | 0 | 待讲解 |
| `tests/test_api.py` | 169 | 11 | 待讲解 |
| `tests/test_bootstrap.py` | 48 | 1 | 待讲解 |
| `tests/test_dataset_pipeline.py` | 311 | 21 | 待讲解 |
| `tests/test_dataset_selection.py` | 131 | 10 | 待讲解 |
| `tests/test_datasets.py` | 258 | 17 | 待讲解 |
| `tests/test_experiment_json.py` | 10 | 1 | 待讲解 |
| `tests/test_hallurag_export.py` | 54 | 3 | 待讲解 |
| `tests/test_measurement.py` | 105 | 9 | 待讲解 |
| `tests/test_method.py` | 52 | 4 | 待讲解 |
| `tests/test_shared_readout.py` | 57 | 4 | 待讲解 |
| `tests/test_token_projection.py` | 183 | 12 | 待讲解 |
| `tools/verify_frozen.py` | 7 | 0 | 待讲解 |
