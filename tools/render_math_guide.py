"""Build offline math articles and toy visualizations from production functions.

Install the optional docs extra. No network, LM execution, fitting or task labels.
"""
import hashlib
import html
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit

import markdown
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from myproject.pooling import cavity_scores, grouped_scores
from experiments.local_evaluation import auc_pair_decomposition

DIRECTORY = ROOT / 'docs/source-reading'
ARTICLES = {
    'THEORY': (DIRECTORY/'THEORY.md', '完整公式与证明'),
    'METHOD': (DIRECTORY/'METHOD.md', '整个方法与调用路径'),
    'PAPER_STORY': (DIRECTORY/'PAPER_STORY.md', '怎样组织论文主张'),
    'VALIDATION': (DIRECTORY/'VALIDATION.md', '验证结果与边界'),
    'METHOD_V3': (DIRECTORY/'METHOD_V3.md', 'V3：模型、优化与代码'),
    'RESULTS_CURRENT': (ROOT/'docs/RESULTS_CURRENT.md', '全数据集方法对照'),
    'INTEGRATED_DESIGN': (DIRECTORY/'INTEGRATED_DESIGN.md', '组合设计与验证'),
    'EXPERIMENT_PLAN': (ROOT/'refine-logs/EXPERIMENT_PLAN.md', '组合方法实验计划'),
}


def article(source):
    """Protect TeX before Markdown interprets underscores/backslash escapes."""
    text = source.read_text(encoding='utf-8')
    equations = []

    def preserve(match):
        equations.append(match.group(1))
        return '\n\n<div data-equation="' + str(len(equations) - 1) + '"></div>\n\n'

    protected = re.sub(r'\\\[([\s\S]*?)\\\]', preserve, text)
    rendered = markdown.markdown(protected, extensions=['tables', 'fenced_code', 'toc'])
    rendered = re.sub(r'<pre><code class="language-mermaid">[\s\S]*?</code></pre>',
        '<div class="steps" role="img" aria-label="原token编码到来源对比、推断和原字符输出">'
        '<span>保留原token/字符轴</span><b>→</b><span>来源有/无条件</span><b>→</b>'
        '<span>固定回答测量</span><b>→</b><span>对齐有效token</span><b>→</b>'
        '<span>shared / sequence / R4</span><b>→</b><span>原字符位置+分数</span></div>', rendered)
    for i, equation in enumerate(equations):
        rendered = rendered.replace(f'<div data-equation="{i}"></div>',
            '<div class="math-block">\\[' + html.escape(equation) + '\\]</div>')
    # Sources can live outside the output directory. Resolve links against the
    # actual source, then point known articles to their offline HTML copies.
    targets = {path.resolve(): DIRECTORY/f'{name}.html'
               for name,(path,_) in ARTICLES.items()}
    def local_link(match):
        href = html.unescape(match.group(1))
        parts = urlsplit(href)
        if parts.scheme or parts.netloc or not parts.path:
            return match.group(0)
        path = (source.parent/parts.path).resolve()
        target = targets.get(path,path)
        relative = Path(os.path.relpath(target,DIRECTORY)).as_posix()
        suffix = ('?'+parts.query if parts.query else '')+('#'+parts.fragment if parts.fragment else '')
        return 'href="'+html.escape(relative+suffix,quote=True)+'"'
    rendered = re.sub(r'href="([^"]+)"',local_link,rendered)
    return rendered, len(equations)


def render_article(name):
    source,title = ARTICLES[name]
    body, count = article(source)
    page = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><link rel="stylesheet" href="assets/guide.css">
<script>window.MathJax={tex:{inlineMath:[['\\(','\\)']],displayMath:[['\\[','\\]']]},svg:{fontCache:'local'},options:{enableMenu:false}};</script>
<script defer src="assets/mathjax-3.2.2/tex-svg-full.js"></script>
<style>@media print{body[data-article="EXPERIMENT_PLAN"]{font-size:14px;line-height:1.65}}</style></head><body data-article="__ARTICLE__">
<header><div class="eyebrow">TOKEN DETECTOR / 本地离线阅读</div><h1>__TITLE__</h1>
<nav><a href="METHOD_V3.html">V3方法</a><a href="RESULTS_CURRENT.html">全数据集对照</a><a href="INTEGRATED_DESIGN.html">组合设计</a><a href="EXPERIMENT_PLAN.html">实验计划</a><a href="VISUAL_GUIDE.html">交互演示</a><a href="THEORY.html">完整公式</a><a href="READING.html">逐行代码</a></nav>
<p>公式已在浏览器排版；可用浏览器“打印 → 保存为PDF”。本页不需要联网。</p></header>
<main><article class="article">__BODY__</article></main></body></html>'''
    (DIRECTORY / f'{name}.html').write_text(
        page.replace('__TITLE__', title).replace('__ARTICLE__',name).replace('__BODY__', body), encoding='utf-8')
    return count


def visual_data():
    """Cache genuine default Beta inference and a separate deletion transform."""
    context = []
    for neighbor in np.linspace(-3, 3, 61):
        values = np.array([neighbor, neighbor, .2])
        result = cavity_scores(values)
        # Core ordering property, with tolerance only for arithmetic ties.
        delta = values[:, None] - values[None, :]
        inferred = result['score'][:, None] - result['score'][None, :]
        strict = abs(delta) > 1e-10
        assert np.array_equal(np.sign(delta[strict]), np.sign(inferred[strict]))
        context.append({'neighbor': float(neighbor), 'score': result['score'].tolist(),
                        'cavity': result['cavity_log_odds'].tolist()})
    values = np.array([.1, .7, .8, .9, .4, .5, 2.7, 2.8])
    labels = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    answers, units = np.zeros(8, int), np.repeat([0, 1], 4)
    means = np.repeat([values[:4].mean(), values[4:].mean()], 4)
    shared = grouped_scores(values, answers, units)['score']
    rows = []
    for weight in np.linspace(0, 1, 101):
        scores = (1 - weight) * values + weight * means
        rows.append({'lambda': float(weight), 'scores': scores.tolist(),
                     'metrics': auc_pair_decomposition(labels, scores, answers, units)})
    initial = rows[0]['metrics']
    assert rows[-1]['metrics']['pooled']['auroc'] > initial['pooled']['auroc']
    assert rows[-1]['metrics']['within_span']['auroc'] == .5
    assert all(row['metrics']['within_span']['auroc'] == initial['within_span']['auroc']
               for row in rows[:-1])
    return {'kind': 'synthetic_mathematical_fixture_not_empirical_accuracy',
            'context': context, 'pooling': rows, 'labels': labels.tolist(),
            'raw_values': values.tolist(), 'color_limits': [float(values.min()), float(values.max())],
            'raw_metrics': initial,
            'shared_metrics': auc_pair_decomposition(labels, shared, answers, units),
            'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                             for name in ['myproject/pooling.py', 'experiments/local_evaluation.py']}}


def main():
    vendor = DIRECTORY/'assets/mathjax-3.2.2/VENDOR_MANIFEST.json'
    for item in json.loads(vendor.read_text(encoding='utf-8'))['files']:
        assert hashlib.sha256((ROOT/item['file']).read_bytes()).hexdigest() == item['sha256']
    counts = {name: render_article(name) for name in ARTICLES}
    data = visual_data()
    packed = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('<', r'\u003c')
    template = (ROOT/'tools/templates/theory_visual.html').read_text(encoding='utf-8')
    (DIRECTORY/'VISUAL_GUIDE.html').write_text(template.replace('__VISUAL_DATA__', packed), encoding='utf-8')
    (DIRECTORY/'assets/visual-data.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'display_equations': counts, 'context_samples': len(data['context']),
                      'pooling_samples': len(data['pooling']), 'new_empirical_results': False}, indent=2))


if __name__ == '__main__':
    main()
