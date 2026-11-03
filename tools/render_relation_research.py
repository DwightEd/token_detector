"""Render H017 repository notes with the existing offline MathJax assets.

Requires the docs extra. Reads Markdown only; no models, scores or annotations.
"""
import hashlib
import html
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

import markdown


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'docs/source-reading'
ARTICLES = {
    'H017_PROGRESS': (ROOT/'docs/results/h017_relation_binding.md', '关系错绑：实验进展与证据'),
    'H017_MATH': (ROOT/'docs/research/h017_path_derivation.md', '高注意力为何不等于正确关系'),
    'H017_PROTOCOL': (ROOT/'docs/research/h017_address_payload_protocol.md', '四来源与Q/K/V干预：冻结协议'),
    'H017_NATURAL': (ROOT/'docs/results/h017_natural_temporal_observation.md', '自然时间错例：原词概率与全部层头'),
    'H017_READOUT': (ROOT/'docs/results/h017_existing_readout_chain.md', '旧R4/V3：单位告警在哪一步下降'),
}


def render(name, source, title):
    text = source.read_text(encoding='utf-8')
    formulas = []
    def preserve(match):
        formulas.append((bool(match.group(1)), match.group(2) or match.group(3)))
        return f'<span data-math="{len(formulas)-1}"></span>'
    text = re.sub(r'(\$\$)([\s\S]*?)\$\$|\$([^\n$]+?)\$', preserve, text)
    body = markdown.markdown(text, extensions=['tables', 'fenced_code'])
    for index, (display, formula) in enumerate(formulas):
        opener, closer = ('\\[', '\\]') if display else ('\\(', '\\)')
        tag = 'div' if display else 'span'
        body = body.replace(f'<span data-math="{index}"></span>',
            f'<{tag} class="equation">{opener}{html.escape(formula)}{closer}</{tag}>')
    body = re.sub(r'<pre><code class="language-mermaid">[\s\S]*?</code></pre>',
        '<div class="flow"><span>来源主体／条件／值</span><b>→</b><span>Q/K检索与V传值</span>'
        '<b>→</b><span>后续层与完整logits</span><b>→</b><span>已有回答的数值token</span></div>', body)
    targets = {path.resolve(): DESTINATION/f'{key}.html' for key,(path,_) in ARTICLES.items()}
    def link(match):
        attribute, href = match.group(1), html.unescape(match.group(2))
        parts = urlsplit(href)
        if parts.scheme or parts.netloc or not parts.path:
            return match.group(0)
        original = (source.parent/parts.path).resolve()
        target = targets.get(original, original)
        relative = Path(os.path.relpath(target, DESTINATION)).as_posix()
        suffix = ('?'+parts.query if parts.query else '')+('#'+parts.fragment if parts.fragment else '')
        return attribute+'="'+html.escape(relative+suffix, quote=True)+'"'
    body = re.sub(r'(href|src)="([^"]+)"', link, body)
    navigation = ' · '.join(f'<a href="{key}.html">{html.escape(item[1])}</a>'
                            for key,item in ARTICLES.items())
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><meta name="source-sha256" content="__SHA__">
<script>window.MathJax={tex:{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']]},svg:{fontCache:'local'},options:{enableMenu:false}};</script>
<script defer src="assets/mathjax-3.2.2/tex-svg-full.js"></script>
<style>body{max-width:1040px;margin:40px auto;padding:0 24px;color:#17253a;font:17px/1.8 system-ui,sans-serif;background:#fbfcff}nav{font-size:14px}a{color:#245bc7}h1,h2{line-height:1.35}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:15px}td,th{border:1px solid #d5ddeb;padding:10px;text-align:left}th{background:#edf2fb}pre{overflow:auto;padding:16px;background:#edf2f8}code{font-size:.88em}.equation{overflow-x:auto}.flow{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:20px;background:#edf2fb;border-radius:12px}.flow span{background:white;padding:8px 12px;border:1px solid #cbd7eb}footer{font-size:13px;border-top:1px solid #d5ddeb;margin-top:40px;padding-top:16px}@media print{body{font-size:12px;margin:0}nav{display:none}}</style>
</head><body><nav>__NAV__</nav><main>__BODY__</main><footer>H017本地离线阅读 · 原Markdown与GitHub公式是同一来源。打开本文件即可阅读，无需启动服务器。<br>源码SHA256：__SHA__</footer></body></html>'''
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    for key,value in {'__TITLE__':html.escape(title),'__NAV__':navigation,'__BODY__':body,'__SHA__':digest}.items():
        page = page.replace(key,value)
    destination = DESTINATION/f'{name}.html'
    destination.write_text(page, encoding='utf-8')
    return destination, len(formulas)


def main():
    if not (DESTINATION/'assets/mathjax-3.2.2/tex-svg-full.js').is_file():
        raise FileNotFoundError('Existing offline MathJax asset is required')
    for name,(source,title) in ARTICLES.items():
        destination, count = render(name, source, title)
        print(f'{destination.relative_to(ROOT)}: {count} formulas')


if __name__ == '__main__':
    main()
