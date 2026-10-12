"""
Paired tests between two models on the same questions (issue 34 of mahdikhashan/jku-thesis-drafts).
Reads the per-question files <task>_write_out_info.json that scripts/compare_stages.sh writes into
<run dir>/<model>/<task>/. The models can come from different run directories, because the harness
gives each question the same doc_id in every run. CPU only, no GPU.

For each comparison "A vs B" on one task:
   - n, the accuracy of A and of B, and A minus B in points
   - the paired SE of the difference and z = difference / paired SE
   - McNemar's exact test: b = questions that only A answers right, c = questions that only B answers right,
     two-sided binomial p-value of b against b + c at 0.5
   - a paired bootstrap 95% interval of the difference (resampled questions, 10,000 samples, seed 0)
   - the Holm-adjusted p-value over all comparisons of the call (family-wise error 0.05)
Metric: acc_norm for arc_challenge and hellaswag (as Table 9 of the Lizard paper), acc for the other tasks.
MMLU: the accuracy over questions, not the harness mean over subjects.

-> python scripts/paired_tests.py OUT.md \
     --compare "window_rope vs teacher" results/window_rope_stage2/stages/stage2/piqa results/teacher_x0/teacher/piqa \
     --compare "window_rope vs teacher" results/window_rope_stage2/stages/stage2/arc_easy results/teacher_x0/teacher/arc_easy
   Writes OUT.md (a table) and OUT.json (all numbers).
"""
import argparse
import json
import math
import os
import random
from os.path import join

NORM_TASKS = ('arc_challenge', 'hellaswag')
BOOTSTRAP_SAMPLES = 10_000


def as_float(x):
    """A per-question score as a float. WinoGrande writes acc as the string 'True' or 'False'"""
    if isinstance(x, str) and x in ('True', 'False'):
        return float(x == 'True')
    return float(x)


def read_questions(task_dir):
    """Task name and {(task, doc_id): 0 or 1} from all *_write_out_info.json files of one model and task"""
    files = sorted(f for f in os.listdir(task_dir) if f.endswith('_write_out_info.json'))
    if not files:
        raise SystemExit(f'No *_write_out_info.json in {task_dir}')
    tasks = [f[:-len('_write_out_info.json')] for f in files]
    name = 'mmlu' if all(t.startswith('hendrycksTest-') for t in tasks) else tasks[0]
    metric = 'acc_norm' if name in NORM_TASKS else 'acc'
    scores = {}
    for f, task in zip(files, tasks):
        with open(join(task_dir, f)) as fh:
            for d in json.load(fh):
                scores[(task, d['doc_id'])] = as_float(d[metric])
    return name, metric, scores


def binom_two_sided(k, n):
    """Exact two-sided p-value of k successes in n trials at p = 0.5"""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    log_p = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
    return min(1.0, 2 * sum(math.exp(x) for x in log_p))


def bootstrap_interval(b, c, n, samples=BOOTSTRAP_SAMPLES, seed=0):
    """95% percentile interval of (b* - c*) / n over resampled questions. Each question gives +1 (only A
    right), -1 (only B right) or 0, so a resample needs only the counts of the three kinds"""
    try:
        import numpy as np
        draws = np.random.default_rng(seed).multinomial(n, [b / n, c / n, 1 - (b + c) / n], size=samples)
        diffs = sorted(((draws[:, 0] - draws[:, 1]) / n).tolist())
    except ImportError:  # slower, without numpy
        rng = random.Random(seed)
        values = [1] * b + [-1] * c + [0] * (n - b - c)
        diffs = sorted(sum(rng.choices(values, k=n)) / n for _ in range(samples // 10))
    return diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]


def compare(label, dir_a, dir_b):
    task_a, metric, a = read_questions(dir_a)
    task_b, _, b_scores = read_questions(dir_b)
    if task_a != task_b:
        raise SystemExit(f'{label}: different tasks ({task_a} and {task_b})')
    common = sorted(set(a) & set(b_scores))
    n = len(common)
    if n < 2:
        raise SystemExit(f'{label}, {task_a}: fewer than 2 common questions')
    d = [a[q] - b_scores[q] for q in common]
    mean = sum(d) / n
    se = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1) / n)
    only_a = sum(1 for x in d if x > 0)
    only_b = sum(1 for x in d if x < 0)
    low, high = bootstrap_interval(only_a, only_b, n)
    return {'label': label, 'task': task_a, 'metric': metric, 'n': n,
            'acc_a': 100 * sum(a[q] for q in common) / n, 'acc_b': 100 * sum(b_scores[q] for q in common) / n,
            'diff': 100 * mean, 'se': 100 * se, 'z': mean / se if se > 0 else None,
            'only_a': only_a, 'only_b': only_b, 'p_mcnemar': binom_two_sided(only_a, only_a + only_b),
            'ci_low': 100 * low, 'ci_high': 100 * high, 'dir_a': dir_a, 'dir_b': dir_b}


def holm(rows):
    """Holm-adjusted p-values, in place"""
    order = sorted(range(len(rows)), key=lambda i: rows[i]['p_mcnemar'])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(rows) - rank) * rows[i]['p_mcnemar']))
        rows[i]['p_holm'] = running


def markdown(rows):
    out = ['| Comparison | Task (metric) | n | A | B | A − B | Paired SE | z | Only A / only B | McNemar p | Holm p | 95% bootstrap |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        z = '–' if r['z'] is None else f"{r['z']:.1f}"
        out.append(f"| {r['label']} | {r['task']} ({r['metric']}) | {r['n']:,} | {r['acc_a']:.1f} | {r['acc_b']:.1f} | "
                   f"{r['diff']:+.1f} | {r['se']:.2f} | {z} | {r['only_a']} / {r['only_b']} | {r['p_mcnemar']:.2g} | "
                   f"{r['p_holm']:.2g} | [{r['ci_low']:+.1f}, {r['ci_high']:+.1f}] |")
    out += ['', 'Accuracy in %. A difference is clear when the Holm p-value is below 0.05.']
    return '\n'.join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('out', help='output Markdown file; the JSON file gets the same name with .json')
    parser.add_argument('--compare', nargs=3, action='append', required=True, metavar=('LABEL', 'DIR_A', 'DIR_B'),
                        help='a label and the two <run dir>/<model>/<task> directories (A minus B)')
    args = parser.parse_args()
    rows = [compare(*c) for c in args.compare]
    holm(rows)
    text = markdown(rows)
    with open(args.out, 'w') as f:
        f.write(text + '\n')
    with open(os.path.splitext(args.out)[0] + '.json', 'w') as f:
        json.dump(rows, f, indent=1)
    print(text)


if __name__ == '__main__':
    main()
