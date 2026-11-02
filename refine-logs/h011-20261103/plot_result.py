"""Publication/exportable figure of fixed, previously exposed qualifications."""
import json
from pathlib import Path
import platform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
summary = json.loads((HERE / 'posthoc_review_v1/summary.json').read_text(encoding='utf-8'))
positions, labels = range(4), ['Source question\n(H008)', 'Masked R\n0 extra facts', 'Masked R\n4 extra facts', 'Masked R\n12 extra facts']
series = [
    ('Closed-candidate QA', [16] + [summary['by_burden'][str(b)]['correct'] for b in (0, 4, 12)], 16),
    ('Source-flip pairs', [8] + [summary['by_burden'][str(b)]['sourceflip']['correctly_reversed_pairs'] for b in (0, 4, 12)], 8),
    ('Event-flip pairs', [8] + [summary['by_burden'][str(b)]['eventflip']['correctly_reversed_pairs'] for b in (0, 4, 12)], 8)]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42})
fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
for axis, (title, values, total) in zip(axes, series, strict=True):
    axis.bar(positions, values, color=['#246a73', '#d48755', '#d48755', '#d48755'], width=.64)
    axis.set_xticks(list(positions), labels, fontsize=8)
    axis.set_ylim(0, total * 1.16)
    axis.set_yticks(list(range(0, total + 1, total // 4)))
    axis.set_title(title)
    axis.grid(axis='y', alpha=.18)
    axis.set_axisbelow(True)
    for position, value in zip(positions, values, strict=True):
        axis.text(position, value + total * .025, f'{value}/{total}', ha='center')
axes[0].set_ylabel('Correct')
fig.suptitle('H011: source-only QA ability does not transfer to the fixed masked-response question', fontsize=12)
fig.text(.5, .01, 'Same frozen 3B; all fixed arms retained. Exposed toy qualification only; no natural-task efficacy or novelty claim.', ha='center', fontsize=8)
fig.tight_layout(rect=(0, .05, 1, .93))
for extension in ('png', 'pdf'):
    fig.savefig(ROOT / f'docs/results/h011_masked_reconstruction_comparison.{extension}', dpi=180, bbox_inches='tight')
(HERE / 'plot-runtime.json').write_text(json.dumps(dict(python=platform.python_version(),
    matplotlib=matplotlib.__version__, backend=matplotlib.get_backend(), plotted_values=series), indent=2) + '\n', encoding='utf-8')
