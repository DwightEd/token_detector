"""Posthoc decomposition of sealed query choices; never generates scores."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STAGE = HERE / 'posthoc_review_v1'
def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
seal = read(STAGE / 'seal.json')
assert all(sha(STAGE / name) == digest for name, digest in seal.items())
queries = read(STAGE / 'queries.json')
source_plan = read(HERE / 'masked_reconstruction_v1/plan.json')
groups = defaultdict(list)
for row in queries:
    groups[row['burden'], row['source_key']].append(row)
records, summaries = [], {}
for (burden, source_key), rows in sorted(groups.items()):
    source = source_plan['prepared']['sources'][source_key]['source']
    rows = sorted(rows, key=lambda row: source.index(row['event']))
    assert len(rows) == 2 and all(not row['tie'] for row in rows)
    assert len(set(rows[0]['candidates'])) == 2 and rows[0]['candidates'] == rows[1]['candidates']
    assert rows[0]['reference_candidate'] != rows[1]['reference_candidate']
    choices = [row['greedy_candidate'] for row in rows]
    correct = sum(row['greedy_correct'] for row in rows)
    collapsed = choices[0] == choices[1]
    assert (correct == 1) == collapsed
    category = 'same_name_for_both_event_roles' if collapsed else 'both_roles_correct' if correct == 2 else 'both_roles_reversed'
    records.append(dict(burden=burden, source_key=source_key,
        relation_family=rows[0]['relation_family'], vocabulary=rows[0]['vocabulary'],
        source_direction=rows[0]['source_direction'], events_in_source_order=[row['event'] for row in rows],
        query_keys=[row['query_key'] for row in rows], choices=choices,
        references=[row['reference_candidate'] for row in rows], correct_queries=correct, category=category,
        collapsed_choice_is_first_event_reference=collapsed and choices[0] == rows[0]['reference_candidate'],
        conditional_probabilities=[row['conditional_pool_probabilities'] for row in rows],
        raw_canonical_log_odds=[row['raw_canonical_log_odds'] for row in rows],
        candidate_mass=[row['candidate_mass'] for row in rows]))
for burden in (0, 4, 12):
    rows = [row for row in records if row['burden'] == burden]
    counts = {name: sum(row['category'] == name for row in rows) for name in
        ['same_name_for_both_event_roles', 'both_roles_correct', 'both_roles_reversed']}
    assert len(rows) == sum(counts.values()) == 8
    assert sum(row['correct_queries'] for row in rows) == (
        counts['same_name_for_both_event_roles'] + 2 * counts['both_roles_correct'])
    summaries[str(burden)] = dict(source_matrices=8, **counts,
        collapsed_choice_is_first_event_reference=sum(row['collapsed_choice_is_first_event_reference'] for row in rows),
        correct_queries=sum(row['correct_queries'] for row in rows))
value = dict(schema='h011-posthoc-role-collapse-v1', scope='Existing raw choices only; no new model calls or method scores.',
    original_queries_sha256=sha(STAGE / 'queries.json'), original_CPU_seal_sha256=sha(STAGE / 'seal.json'),
    identity='For two distinct true roles/two candidates/no ties: N_correct_queries=2*N_both_correct+N_collapsed; N_sources=N_both_correct+N_collapsed+N_both_reversed.',
    by_burden=summaries, records=records,
    limitations=['A collapsed output is observable behavior, not proof of an internal attention or generation cause.',
        'Event and person anchors remain logically sufficient in the toy grammar; this does not prove an information-theoretic loss from erasure.',
        'No method/threshold/prompt is selected from these posthoc outcomes.'])
(HERE / 'role-collapse-audit.json').write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(HERE / 'role-collapse-audit.md').write_text(
    '# H011 封存预测的角色坍缩审计\n\n零新增推理。两个不同事件的真姓名不同、候选只有两人且无tie，因此每个来源要么双正确、双反向，要么两个问题都选同一人（恰好对一个）。\n\n'
    '| 后缀命题数 | 两角色选同名 | 双正确 | 双反向 | 同名时选第一事件答案 |\n|---|---:|---:|---:|---:|\n' +
    ''.join(f"| {b} | {s['same_name_for_both_event_roles']}/8 | {s['both_roles_correct']}/8 | {s['both_roles_reversed']}/8 | {s['collapsed_choice_is_first_event_reference']} |\n"
            for b, s in summaries.items()) +
    '\n这些是模型输出行为，不证明内部注意力或原始生成原因。toy题面仍含足以唯一确定两角色的锚，不能说擦值必然造成信息论不可识别。不得据此调prompt或偏置去选择赢家。\n',
    encoding='utf-8')
print(json.dumps(summaries, ensure_ascii=False, indent=2))
