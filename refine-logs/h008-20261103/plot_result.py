"""Plot sealed-report metrics only; no model, tokenizer, or new truth reads."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPORT = ROOT / 'docs/results/h008_source_reconstruction.json'
DESTINATION = ROOT / 'docs/results/h008_source_reconstruction_comparison'
RUNTIME = HERE / 'plot-runtime.json'
started = time.monotonic()
assert not RUNTIME.exists()
assert all(not DESTINATION.with_suffix(s).exists() for s in ('.png', '.pdf'))
report = json.loads(REPORT.read_text(encoding='utf-8'))
rows = report['h007_fixed_scope_comparison']
summary = report['summary']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.4), gridspec_kw={'width_ratios': [1.3, 1]})
burdens = [r['response_burden'] for r in rows]
for key, label, color, marker in [
    ('h007_whole_auroc', 'H007 whole-answer verifier', '#32658c', 'o'),
    ('h007_target_focus_auroc', 'H007 target-focus verifier', '#b06c35', 's'),
    ('h008_secondary_case_auroc', 'H008 source-only QA (secondary)', '#267255', '^')]:
    values = [r[key] for r in rows]
    left.plot(burdens, values, color=color, marker=marker, linewidth=1.8, label=label)
    for i, (x, y) in enumerate(zip(burdens, values)):
        left.annotate(f'{y:.3f}', (x, y), xytext=(0, -16 if key == 'h007_target_focus_auroc' else 7),
                      textcoords='offset points', ha='center', color=color, fontsize=9)
left.axhline(.5, color='#999999', linewidth=.8, linestyle=':', zorder=0)
left.set(title='Same 48 constructed answers: case AUROC', xlabel='Unrelated response-suffix sentences',
         ylabel='AUROC (secondary for H008)', ylim=(.30, 1.10), xticks=burdens)
left.legend(loc='lower left', frameon=False, fontsize=9)
left.grid(axis='y', alpha=.16)
counts = [(summary['source_qa']['correct'], summary['source_qa']['unique_queries']),
          (summary['sourceflip']['correctly_reversed_pairs'], summary['sourceflip']['pairs']),
          (summary['eventflip']['correctly_reversed_pairs'], summary['eventflip']['pairs'])]
positions = list(range(3))
right.bar(positions, [a / b for a, b in counts], width=.58, color=['#267255', '#4b8875', '#73a493'])
for x, (a, b) in zip(positions, counts):
    right.text(x, 1.025, f'{a}/{b}', ha='center', va='bottom', fontweight='bold')
right.set(title='H008 primary qualification', ylabel='Correct fraction',
          xticks=positions, xticklabels=['Source QA', 'Source flip', 'Event flip'], ylim=(0, 1.16))
right.grid(axis='y', alpha=.16)
right.set_axisbelow(True)
fig.suptitle('CONSTRUCTED TOY ONLY — no natural-data performance claim', fontsize=13, fontweight='bold')
fig.text(.5, .065, 'Different tasks, inputs, output spaces and budgets; this comparison does not isolate label bias.',
         ha='center', fontsize=9)
fig.text(.5, .025, 'H008 reuses identical source-QA requests across suffix burdens; the exact toy parser can solve the bindings.',
         ha='center', fontsize=9)
fig.tight_layout(rect=[0, .12, 1, .93], w_pad=3)
for extension in ('.png', '.pdf'):
    fig.savefig(DESTINATION.with_suffix(extension), dpi=240, bbox_inches='tight')
plt.close(fig)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
runtime = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), status='completed',
    executable=sys.executable, python=platform.python_version(), matplotlib=matplotlib.__version__,
    argv=sys.argv, working_directory=str(Path.cwd()), script_sha256=sha(Path(__file__)),
    input_report=str(REPORT), input_report_sha256=sha(REPORT),
    plot_seconds=time.monotonic() - started, new_model_calls=0, new_tokenizer_calls=0,
    new_ground_truth_reads=0, input_is_already_released_result_report=True,
    natural_task_performance_claim=False,
    outputs={str(DESTINATION.with_suffix(s).relative_to(ROOT)): sha(DESTINATION.with_suffix(s))
             for s in ('.png', '.pdf')})
with RUNTIME.open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(runtime, stream, ensure_ascii=False, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps(dict(status=runtime['status'], runtime=str(RUNTIME), outputs=runtime['outputs'])))
