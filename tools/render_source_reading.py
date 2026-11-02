"""Render explicit line notes, refusing gaps or stale source bytes."""
import hashlib
import html
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
directory = root / 'docs/source-reading'
inventory = json.loads((directory / 'SOURCE_INVENTORY.json').read_text(encoding='utf-8'))
by_path = {item['path']: item for item in inventory['files']}
annotations = {}
for file in sorted(directory.glob('line-notes*.json')):
    for path, notes in json.loads(file.read_text(encoding='utf-8')).items():
        if path in annotations:
            raise ValueError(f'Duplicate annotation file: {path}')
        annotations[path] = notes

sections, problems = [], []
output = directory / 'files'
output.mkdir(exist_ok=True)
for path, notes in annotations.items():
    raw = (root / path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    entry = by_path[path]
    if digest != entry['sha256']:
        raise ValueError(f'Stale source snapshot: {path}')
    lines = raw.decode('utf-8').splitlines()
    mapped = {}
    for specification, note in notes.items():
        endpoints = list(map(int, specification.split('-')))
        start, stop = endpoints[0], endpoints[-1]
        if not 1 <= start <= stop <= len(lines):
            raise ValueError(f'Invalid range: {path}:{specification}')
        for number in range(start, stop + 1):
            if number in mapped:
                raise ValueError(f'Overlapping explanation: {path}:{number}')
            mapped[number] = note
    missing = [number for number, line in enumerate(lines, 1) if line.strip() and number not in mapped]
    if missing:
        problems.append({'path': path, 'missing': missing})
        continue
    rows, markup = [], []
    for number, line in enumerate(lines, 1):
        note = mapped.get(number, '空白行：分隔定义/逻辑段，不执行运算。')
        escaped = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('|', '&#124;')
        rows.append(f'| {number} | <code>{escaped or " "}</code> | {note} |')
        markup.append(f'<tr id="{path.replace("/", "-")}-{number}"><td>{number}</td>'
                      f'<td><code>{html.escape(line)}</code></td><td>{html.escape(note)}</td></tr>')
    functions = []
    for function in entry['functions']:
        header = function['line']
        functions.append(f"| `{function['name']}` | {header}–{function['end_line']} | {mapped.get(header, '')} |")
    body = (f'# {path} 逐行讲解\n\n'
            f"基线：`{inventory['base_revision']}`；SHA-256：`{digest}`；{len(lines)} 行。\n\n"
            '源码与注释逐行并排。一个跨行表达式共用同一操作说明；这不代表各行独立执行。'
            '完整调用路径、输入输出轴和理论边界见 [METHOD.md](../METHOD.md)。'
            '未执行的 GPU 路径见 [验证边界](../VALIDATION.md)。\n\n'
            '| 函数（含嵌套） | 原始行号 | 作用与契约入口 |\n|---|---|---|\n'
            + '\n'.join(functions) + '\n\n| 行 | 原始代码 | 操作、目的与边界 |\n|---|---|---|\n'
            + '\n'.join(rows) + '\n')
    destination = output / (path.replace('/', '_').removesuffix('.py') + '.md')
    destination.write_text(body, encoding='utf-8')
    sections.append(f'<details><summary>{html.escape(path)} · {len(lines)} 行</summary>'
                    f'<p>SHA-256: {digest}</p><table><thead><tr><th>行</th><th>源码</th><th>解释</th></tr></thead>'
                    '<tbody>\n' + '\n'.join(markup) + '</tbody></table></details>')

if problems:
    print(json.dumps(problems, ensure_ascii=False, indent=2))
    raise SystemExit(1)

page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>token_detector 源码逐行讲解</title>
<style>body{font:16px/1.65 system-ui,sans-serif;margin:24px;color:#172033;background:#f8fafc}
h1{font-size:26px}summary{cursor:pointer;font-size:20px;padding:12px;background:#e5edf7;margin-top:12px}
table{border-collapse:collapse;width:100%;background:white}td,th{padding:8px;border:1px solid #d6deea;text-align:left;vertical-align:top}
td:first-child{width:40px;color:#64748b}td:nth-child(2){width:46%;white-space:pre-wrap}code{font:13px/1.5 Consolas,monospace}
p{overflow-wrap:anywhere}a{color:#1658ab}</style><h1>token_detector：真实源码逐行讲解</h1>
<p>固定源码基线 1210120748b3a05bd271bc79debcbed4fdc0c964。展开文件查看源码与中文说明。
只展示已经逐行阅读并有具体说明的文件；全项目待讲解范围保存在 SOURCE_INVENTORY.json，未展示不代表不存在。</p>
<p><strong><a href="METHOD_V3.html">先看：V3当前研究方法、优化与公式 →</a></strong> · <a href="VISUAL_GUIDE.html">V1构造例与指标交互</a></p>
<p><a href="METHOD_V3.html">V3当前研究方法</a> · <a href="RESULTS_CURRENT.html">全数据集对照</a> · <a href="INTEGRATED_DESIGN.html">组合设计</a> · <a href="EXPERIMENT_PLAN.html">实验计划</a> · <a href="METHOD.html">库调用路径</a> · <a href="THEORY.html">完整公式</a> · <a href="VALIDATION.html">实际验证</a></p>
'''
(directory / 'READING.html').write_text(page + ''.join(sections) + '</html>', encoding='utf-8')
print(json.dumps({'files': len(annotations), 'source_lines': sum(by_path[path]['line_count'] for path in annotations),
                  'functions': sum(len(by_path[path]['functions']) for path in annotations),
                  'annotation_gaps': 0}, indent=2))
