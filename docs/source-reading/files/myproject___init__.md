# myproject/__init__.py 逐行讲解

基线：`1210120748b3a05bd271bc79debcbed4fdc0c964`；SHA-256：`cde213e7a2883957788ba353947e57bcfbbe272a010d3cfd235191ffc4ec6f83`；6 行。

源码与注释逐行并排。一个跨行表达式共用同一操作说明；这不代表各行独立执行。完整调用路径、输入输出轴和理论边界见 [METHOD.md](../METHOD.md)。未执行的 GPU 路径见 [验证边界](../VALIDATION.md)。

| 函数（含嵌套） | 原始行号 | 作用与契约入口 |
|---|---|---|


| 行 | 原始代码 | 操作、目的与边界 |
|---|---|---|
| 1 | <code>"""Token-level source risk with a shared latent state and individual evidence."""</code> | 包说明：共享状态与个体来源证据，命名没有声称校准事实概率。 |
| 2 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 3 | <code>from .api import HallucinationDetector</code> | 公开HallucinationDetector组合文本API；导入api会依次导入读出内核，但不会加载模型权重。 |
| 4 | <code>from .shared import SharedDetector</code> | 公开SharedDetector便于直接给已测full-gap打分。 |
| 5 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 6 | <code>__all__ = ["HallucinationDetector", "SharedDetector"]</code> | __all__限制star import的推荐公共符号，历史Detector仍可由明确模块路径导入。 |
