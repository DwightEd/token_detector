# myproject/run.py 逐行讲解

基线：`1210120748b3a05bd271bc79debcbed4fdc0c964`；SHA-256：`f7dd726881b201e0a1e02a962f38ee7b5e87066b38ececf9f4ef384fa799a578`；47 行。

源码与注释逐行并排。一个跨行表达式共用同一操作说明；这不代表各行独立执行。完整调用路径、输入输出轴和理论边界见 [METHOD.md](../METHOD.md)。未执行的 GPU 路径见 [验证边界](../VALIDATION.md)。

| 函数（含嵌套） | 原始行号 | 作用与契约入口 |
|---|---|---|
| `main` | 12–43 | main 是 token-detect/r4-detect 与 python -m myproject 共用入口，无返回值，有终端/文件副作用。 |

| 行 | 原始代码 | 操作、目的与边界 |
|---|---|---|
| 1 | <code>"""Detect an existing response and emit only its final per-token scores."""</code> | CLI 只输出已有回答的最终 token 分数，方法库本身保持无I/O。 |
| 2 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 3 | <code>import argparse</code> | argparse 解析并验证命令选项。 |
| 4 | <code>import json</code> | json 读取输入并输出Unicode记录。 |
| 5 | <code>from pathlib import Path</code> | Path 用于本地模型、输入、输出和参考路径。 |
| 6 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 7 | <code>from .api import HallucinationDetector</code> | CLI 调用同一公共文本API，不复制测量/算法。 |
| 8 | <code>from .detector import RECIPES</code> | 任务和方法选项来自库定义，避免CLI接受库不支持的值。 |
| 9 | <code>from .shared import METHODS</code> | 任务和方法选项来自库定义，避免CLI接受库不支持的值。 |
| 10 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 11 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 12 | <code>def main() -&gt; None:</code> | main 是 token-detect/r4-detect 与 python -m myproject 共用入口，无返回值，有终端/文件副作用。 |
| 13 | <code>    command = argparse.ArgumentParser(__doc__)</code> | 模块docstring作为命令说明；--help 在加载模型前退出。 |
| 14 | <code>    command.add_argument("--model", type=Path, required=True)</code> | --model 必填，本地路径；不是远端服务URL。 |
| 15 | <code>    command.add_argument("--references", type=Path)</code> | --references 可选，指定后默认走R4，不能与共享方法混用。 |
| 16 | <code>    command.add_argument("--method", choices=METHODS,</code> | --method 显式受METHODS约束；缺省由API选择shared或带参考时的r4。help列的是主要方法而非全部细节。 |
| 17 | <code>                         help='shared default; persistent V2, witness V3, responsibility diagnostic, or legacy r4')</code> | --method 显式受METHODS约束；缺省由API选择shared或带参考时的r4。help列的是主要方法而非全部细节。 |
| 18 | <code>    command.add_argument("--task", choices=tuple(RECIPES), default="QA")</code> | 任务只允许三种冻结名称，默认QA。 |
| 19 | <code>    command.add_argument("--device", default="cuda:0")</code> | 默认device=cuda:0；CPU检查--help并不意味着实际模型能在CPU按同速度运行。 |
| 20 | <code>    command.add_argument("--input", type=Path, help="JSON with prompt, source and response")</code> | 支持一个UTF8 JSON文件含prompt/source/response。 |
| 21 | <code>    command.add_argument("--prompt")</code> | 也可直接传三个文本选项，必须一起提供。 |
| 22 | <code>    command.add_argument("--source")</code> | 也可直接传三个文本选项，必须一起提供。 |
| 23 | <code>    command.add_argument("--response")</code> | 也可直接传三个文本选项，必须一起提供。 |
| 24 | <code>    command.add_argument("--output", type=Path)</code> | 输出路径可省略，缺省写stdout。 |
| 25 | <code>    args = command.parse_args()</code> | 解析参数；不合法选项在任何模型加载前退出。 |
| 26 | <code>    direct = (args.prompt, args.source, args.response)</code> | 集中三个直接文本值，用于检查输入方式互斥/完整。 |
| 27 | <code>    if args.input is not None:</code> | 选择JSON输入时不能再混传任一直接文本；明确UTF8读取。JSON必需键及类型由后续使用校验/异常。 |
| 28 | <code>        if any(value is not None for value in direct):</code> | 选择JSON输入时不能再混传任一直接文本；明确UTF8读取。JSON必需键及类型由后续使用校验/异常。 |
| 29 | <code>            command.error("use either --input or --prompt/--source/--response")</code> | 选择JSON输入时不能再混传任一直接文本；明确UTF8读取。JSON必需键及类型由后续使用校验/异常。 |
| 30 | <code>        example = json.loads(args.input.read_text(encoding="utf-8"))</code> | 选择JSON输入时不能再混传任一直接文本；明确UTF8读取。JSON必需键及类型由后续使用校验/异常。 |
| 31 | <code>    else:</code> | 无JSON时必须三个文本都不是None，然后组装相同例子字典；空字符串是显式输入而非缺参。 |
| 32 | <code>        if any(value is None for value in direct):</code> | 无JSON时必须三个文本都不是None，然后组装相同例子字典；空字符串是显式输入而非缺参。 |
| 33 | <code>            command.error("provide --input or all of --prompt/--source/--response")</code> | 无JSON时必须三个文本都不是None，然后组装相同例子字典；空字符串是显式输入而非缺参。 |
| 34 | <code>        example = {"prompt": args.prompt, "source": args.source, "response": args.response}</code> | 无JSON时必须三个文本都不是None，然后组装相同例子字典；空字符串是显式输入而非缺参。 |
| 35 | <code>    detector = HallucinationDetector.from_pretrained(</code> | 统一from_pretrained加载本地模型并选择读出/任务/设备，缺资源不自动下载。 |
| 36 | <code>        args.model, task=args.task, references=args.references, device=args.device, method=args.method,</code> | 统一from_pretrained加载本地模型并选择读出/任务/设备，缺资源不自动下载。 |
| 37 | <code>    )</code> | 统一from_pretrained加载本地模型并选择读出/任务/设备，缺资源不自动下载。 |
| 38 | <code>    result = detector.predict(example["prompt"], example["response"], example["source"])</code> | 预测用户已有response，不生成回答。 |
| 39 | <code>    output = json.dumps(result, ensure_ascii=False) + "\n"</code> | 序列化原文Unicode而非ASCII转义，并加一个换行。 |
| 40 | <code>    if args.output is None:</code> | 无输出文件时只把JSON写stdout，end=''避免额外空行。 |
| 41 | <code>        print(output, end="")</code> | 无输出文件时只把JSON写stdout，end=''避免额外空行。 |
| 42 | <code>    else:</code> | 有输出路径时按UTF8写文件；不自动创建父目录。 |
| 43 | <code>        args.output.write_text(output, encoding="utf-8")</code> | 有输出路径时按UTF8写文件；不自动创建父目录。 |
| 44 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 45 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 46 | <code>if __name__ == "__main__":</code> | 直接运行此模块才调用main；被导入时不执行CLI。 |
| 47 | <code>    main()</code> | 直接运行此模块才调用main；被导入时不执行CLI。 |
