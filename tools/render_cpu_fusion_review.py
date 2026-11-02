"""Render the archived, rejected CPU baseline from sanitized aggregate evidence."""

import json
from math import ceil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = ['QA', 'Summary', 'Data2txt', 'FinQA', 'TATQA']
METHODS = {'r4': 'R4', 'witness': 'V3 / witness', 'fusion': '固定等权秩融合'}


def render(evidence):
    assert set(evidence['tasks']) == set(TASKS)
    assert evidence['all_scores_sealed_before_labels'] and not evidence['selection_or_weight_search']
    assert sum(t['population']['responses_scored'] for t in evidence['tasks'].values()) == 2992
    assert sum(t['population']['tokens_scored'] for t in evidence['tasks'].values()) == 448202
    lines = [
        '# 固定等权秩融合：已完成但未采用的CPU基线',
        '',
        '这是一次已归档的低成本基线检查，不是新的主线方法。五任务全部评分封存后才读取最终标签；没有搜索权重或按标签选择任务配置。'
        '**金融两任务均弱于V3，未采用该基线；它没有完成逐任务AUROC≥0.8目标。** 下一步先解释原始FN/FP的来源与机制。',
        '',
        f"完成时间：`{evidence['completed_utc']}`。复用完整历史缓存，共2992答、448202个原始有效token；本轮没有调用语言模型。"
        f"本地恢复目录：`{evidence['cache_recovery']['canonical_local_root']}`。原文件及原token轴已核对SHA256。",
        '',
        '## 实际做了什么',
        '',
        'R4和V3各自的最终风险分数先在完整任务人群内转换成经验中位秩，再各取一半。高分仍表示高风险。'
        '这只统一两路分数的排序尺度，没有新增语义观测、关系验证或训练。它是依赖整个任务批次的transductive基线，不能直接当作逐条API方法。',
        '',
        r'\[',
        r'r_m(i)=\frac{1}{N}\left(\sum_{j=1}^{N}\mathbf{1}[s_m(j)<s_m(i)]+'
        r'\frac12\sum_{j=1}^{N}\mathbf{1}[s_m(j)=s_m(i)]\right),\quad m\in\{R4,V3\}.',
        r'\]',
        '',
        r'\[',
        r's_{\mathrm{fusion}}(i)=\tfrac12r_{R4}(i)+\tfrac12r_{V3}(i).',
        r'\]',
        '',
        '两边的0.5在查看本轮标签前固定。秩变换保留各路内部次序，但平均后的整体次序可以变好或变坏，没有AUROC必然提高的保证。'
        '所有原始评分token均保留；unknown只允许在最终评价中排除。本轮五任务实际unknown为0。',
        '',
        '## 完整人群与三路结果',
        '',
        '| 任务 | 回答数 | token数 | 正例token数 | 聚类数 | 聚类依据 |',
        '|---|---:|---:|---:|---:|---|',
    ]
    for task in TASKS:
        t = evidence['tasks'][task]
        p = t['population']
        lines.append(f"| {task} | {p['responses_scored']} | {p['tokens_scored']} | "
                     f"{t['methods']['r4']['errors']} | {p['sources_evaluated']} | {t['bootstrap_type']} |")
    lines += [
        '',
        'RAGTruth使用人工字符跨度及官方source_id聚类。FRED金融任务使用发布的合成编辑跨度，按完整referenceText的SHA256聚类；'
        '142／110表示不同来源文本，不表示142／110家公司或独立财报。',
        '',
        '| 任务 | 方法 | AUROC | AP | top5% TP | FP | FN |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for task in TASKS:
        for method, title in METHODS.items():
            value = evidence['tasks'][task]['methods'][method]
            alarm = value['top5pct']
            assert alarm['tp'] + alarm['fp'] == ceil(.05 * value['tokens'])
            assert alarm['tp'] + alarm['fn'] == value['errors']
            lines.append(f"| {task} | {title} | {value['auroc']:.6f} | {value['ap']:.6f} | "
                         f"{alarm['tp']} | {alarm['fp']} | {alarm['fn']} |")
    lines += [
        '',
        'top5%在每个完整任务上按最终分数排序，同分按原始顺序稳定打破。它不是score>0阈值；改变告警预算不改变AUROC。'
        '五个已评任务中，R4、V3、融合分别有1、3、1项点值达到0.8。HalluRAG仍未完成融合评价，不能从总体目标中移除。',
        '',
        '## 配对差值与解释',
        '',
        '下表均为“融合−对照”。按source_id整体重采样1000次，seed=17；所有区间均为名义95%配对bootstrap区间，'
        '每项1000次有定义、0次退化，未做多重比较校正。这些已暴露人群仍是探索证据。',
        '',
        '| 任务 | 对照 | ΔAUROC［95%区间］ | ΔAP［95%区间］ |',
        '|---|---|---:|---:|',
    ]
    for task in TASKS:
        for baseline in ['r4', 'witness']:
            d = evidence['tasks'][task]['paired_differences']['fusion_minus_' + baseline]
            columns = []
            for metric in ['auroc', 'ap']:
                interval = d['bootstrap'][metric]
                assert d['bootstrap']['repetitions'] == 1000 and d['bootstrap']['seed'] == 17
                assert interval['defined'] == 1000 and interval['degenerate'] == 0
                lo, hi = interval['interval95']
                columns.append(f"{d['point'][metric]:+.6f} [{lo:+.6f}, {hi:+.6f}]")
            lines.append(f"| {task} | {METHODS[baseline]} | {' | '.join(columns)} |")
    lines += [
        '',
        '- Summary的AUROC从V3的0.762337升到0.769380，但差值区间［−0.001229，0.012952］跨0；'
        'AP从0.104562升到0.124156，差值区间［0.012369，0.030730］为正。它仍未达到0.8。',
        '- FinQA／TATQA融合AUROC为0.751483／0.792889，低于V3的0.802656／0.842992；'
        '两任务的AUROC和AP差值区间相对V3均完全为负。固定融合损害了现有金融优势。',
        '- QA没有超过R4的点值；Data2txt的AUROC没有超过R4，AP点值增加但相对R4的差值区间跨0。',
        '',
        '因此未晋升这个基线，金融研究候选仍保留V3，软件默认shared／V1也未改变。'
        '这一次负结果既不能证明所有模块组合无效，也不能支持一般性的“R4与V3已经实现有效互补”。不宣称新SOTA或顶会方法。',
        '',
        'HalluRAG旧R4完整配对分数仍缺失，现有统计表不能代替token数组计算融合或配对区间。'
        '冻结语言模型的语义核验器仅为PROPOSED：未采集、未评分、未评价。当前优先做来源、回答、标注和各阶段分数对应的FN/FP机制审计，不另造启发式评分。',
        '',
        '## 可追溯记录',
        '',
        f"实际运行：`{evidence['provenance']['run']}`。公开[汇总证据JSON](../results/cpu_fusion_evidence.json)保留三路指标、人群、配对区间和各评价seal/hash。",
        '',
        f"评价协议SHA256：`{evidence['provenance']['evaluation_protocol_sha256']}`。",
        '',
        f"{evidence['provenance']['source_module_count']}项运行源码／资源hash清单的规范JSON摘要：`{evidence['provenance']['source_roster_sha256']}`。"
        '此处绑定当前CPU评分／评价实现，不据此补造历史GPU生产者的源码或权重认证。',
        '',
        '| 本轮关键实现 | 运行时SHA256 |',
        '|---|---|',
    ]
    for name, digest in evidence['provenance']['source_files_sha256'].items():
        lines.append(f'| `{name}` | `{digest}` |')
    lines += ['', '[历史R4/V3与错例](PERFORMANCE_REVIEW.md) · [当前任务表](../../refine-logs/EXPERIMENT_TRACKER.md)', '']
    return '\n'.join(lines)


def main():
    source = ROOT / 'docs/results/cpu_fusion_evidence.json'
    evidence = json.loads(source.read_text(encoding='utf-8'))
    target = ROOT / 'docs/source-reading/CPU_FUSION_REVIEW.md'
    target.write_text(render(evidence), encoding='utf-8')
    print(json.dumps({'artifact': target.relative_to(ROOT).as_posix(), 'tasks': len(TASKS),
                      'new_detector': False, 'default_promoted': False}))


if __name__ == '__main__':
    main()
