"""Make the new plots of the thesis.

Every number comes from the project notes in docs/. The comment above each data block names
the source document. The per-subject MMLU values come from data/mmlu_per_subject.csv.

Output: vector PDF files in figures/. Run from the folder docs/thesis/src:
    python plots/make_plots.py
"""
import csv
import datetime as dt
import pathlib

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SRC = pathlib.Path(__file__).resolve().parent.parent
OUT = SRC / 'figures'
OUT.mkdir(exist_ok=True)

# Palette: the colors of the JKU template (jkureport.sty), checked with the dataviz validator.
# Categorical order blue, green, purple, red passes the adjacent-pair checks. For plots that compare all
# pairs (scatter), the marker shape is a second encoding, because blue and purple are close for protan vision.
JKU_BLUE, JKU_GREEN, JKU_PURPLE, JKU_RED = '#046E98', '#4CAC4E', '#9C477B', '#E73729'
JKU_GREY, JKU_DARK_GREY = '#808288', '#5A5A5A'
SURFACE = '#FFFFFF'
INK, INK2, MUTED = '#000000', JKU_DARK_GREY, JKU_GREY
GRID, AXIS = '#E3E4E6', '#B3B4B8'
BLUE = JKU_BLUE
SERIES = [JKU_BLUE, JKU_GREEN, JKU_PURPLE, JKU_RED]
MARKERS = ['o', 's', '^', 'D']


def _mix(c1, c2, f):
    a = np.array([int(c1[i:i + 2], 16) for i in (1, 3, 5)], float)
    b = np.array([int(c2[i:i + 2], 16) for i in (1, 3, 5)], float)
    return '#%02X%02X%02X' % tuple(np.round(a + (b - a) * f).astype(int))


# Sequential ramp of JKU blue: light tint to dark shade
BLUE_RAMP = [_mix('#FFFFFF', JKU_BLUE, f) for f in np.linspace(0.18, 1.0, 9)] + \
            [_mix(JKU_BLUE, '#012A3B', f) for f in np.linspace(0.25, 1.0, 4)]
SEQ = LinearSegmentedColormap.from_list('jkublue', ['#FFFFFF'] + BLUE_RAMP)

# The sans font of the JKU template (Public Sans), from the fonts folder of the template
FONT_DIR = SRC / 'fonts'
for _f in sorted(FONT_DIR.glob('PublicSans-*.ttf')):
    font_manager.fontManager.addfont(str(_f))
FONT = 'Public Sans' if any(FONT_DIR.glob('PublicSans-Regular.ttf')) else 'DejaVu Sans'

TEXT_WIDTH = 5.3  # inches, the text width of the JKU template (384 pt)

plt.rcParams.update({
    'font.family': FONT, 'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'text.color': INK, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.8, 'axes.facecolor': SURFACE, 'figure.facecolor': 'white',
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'grid.linestyle': '-',
    'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'lines.linewidth': 1.6, 'lines.solid_capstyle': 'round',
    'pdf.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
})


def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf')
    plt.close(fig)
    print('wrote', name)


def bar_labels(ax, bars, fmt='{:.1f}', dy=0.8, size=7):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, fmt.format(b.get_height()),
                ha='center', va='bottom', fontsize=size, color=INK2)


# 1. Which keys each attention gets weight from (a toy example, no data)
def attention_patterns():
    n, w, gamma = 20, 5, 0.82
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    causal = (j <= i).astype(float)
    window = ((j <= i) & (j > i - w)).astype(float)
    gated = np.where(j <= i, gamma ** (i - j), 0.0)
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.35))
    panels = [(0.7 * causal, '(a) Softmax attention'),
              (window, f'(b) Window branch (w = {w})'),
              (gated, '(c) Gated branch')]
    for ax, (m, title) in zip(axes, panels):
        shown = m.copy()
        if ax is axes[1]:
            shown = np.concatenate([np.full((n, 1), 0.55), np.full((n, 1), np.nan), shown], axis=1)
        ax.imshow(shown, cmap=SEQ, vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(title, color=INK, pad=4)
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True), s.set_color(AXIS)
        ax.set_xlabel('key position $j$', labelpad=2)
    axes[0].set_ylabel('query position $i$', labelpad=2)
    axes[1].annotate('sinks', (0, n - 0.5), xytext=(0, -9), textcoords='offset points', fontsize=6.5,
                     color=INK2, ha='center', va='top')
    fig.subplots_adjust(wspace=0.12)
    save(fig, 'attention_patterns')


# 2. First Lizard model against the teacher and the paper (docs/07-results.md, Summary)
def scores_first_model():
    tasks = ['MMLU (5-shot)', 'PIQA', 'ARC-Easy']
    series = [('First Lizard model (this work)', [23.3, 67.95, 54.8]),
              ('Teacher, this harness', [31.7, 74.43, 65.32]),
              ('Lizard 1B, paper', [29.8, 74.8, 65.6]),
              ('Teacher 1B, paper', [31.0, 74.1, 65.4])]
    chance = [25, 50, 25]
    x = np.arange(len(tasks))
    width = 0.19
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.6))
    for k, (name, vals) in enumerate(series):
        bars = ax.bar(x + (k - 1.5) * width, vals, width * 0.92, color=SERIES[k], label=name,
                      edgecolor=SURFACE, linewidth=0.6)
        bar_labels(ax, bars, size=6.5)
    for xi, c in zip(x, chance):
        ax.plot([xi - 2.1 * width, xi + 2.1 * width], [c, c], color=INK2, lw=0.9)
        ax.text(xi + 2.15 * width, c, 'chance', fontsize=6.5, color=INK2, va='center')
    ax.set_xticks(x, tasks)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 90, 10))
    ax.grid(axis='x', visible=False)
    ax.legend(ncol=2, loc='upper left', bbox_to_anchor=(0, 1.03))
    save(fig, 'scores_first_model')


# 3. Stage 1 loss against PIQA and ARC-Easy after stage 1
# docs/14-liger-gla.md section 5, docs/experiments/xai-position-mse.md (accuracy table),
# docs/experiments/window-rope.md, window-rope-hybrid.md, lolcats-control.md
def loss_vs_accuracy():
    runs = [  # name, loss, PIQA, ARC-Easy, group
        ('Run 1 (fd128)', 3.2549, 57.6, 39.0, 0),
        ('Paper LR, bf16', 8.1641, 55.8, 34.1, 0),
        ('Paper LR, float32', 4.9478, 57.7, 35.7, 0),
        ('Config 1', 3.9764, 57.3, 36.5, 0),
        ('Config 2', 3.5092, 57.5, 35.6, 0),
        ('C1', 3.8092, 57.5, 34.6, 0),
        ('R1b', 3.0542, 55.9, 29.9, 0),
        ('window_rope', 1.2290, 61.5, 43.3, 1),
        ('window_rope + hybrid', 1.6172, 58.9, 36.0, 1),
        ('LoLCATs attention', 0.4442, 73.5, 62.8, 2),
    ]
    groups = ['Lizard, window without RoPE', 'Lizard, RoPE in the window', 'LoLCATs attention']
    teacher = (74.43, 65.32)
    label_at = {  # label positions in data coordinates (x, y) for each panel, with a thin leader line
        2: {'Run 1 (fd128)': (4.6, 61.5), 'Paper LR, bf16': (5.0, 53.0), 'R1b': (1.25, 53.5), 'window_rope': (1.6, 66.0),
            'window_rope + hybrid': (0.38, 56.0), 'LoLCATs attention': (0.62, 70.0)},
        3: {'Run 1 (fd128)': (4.6, 47.0), 'Paper LR, bf16': (5.2, 28.0), 'R1b': (1.4, 27.5), 'window_rope': (0.38, 50.0),
            'window_rope + hybrid': (0.38, 32.0), 'LoLCATs attention': (0.62, 57.0)}}
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.9), sharex=True)
    for ax, col, title, t in ((axes[0], 2, 'PIQA after stage 1', teacher[0]),
                              (axes[1], 3, 'ARC-Easy after stage 1', teacher[1])):
        for g in range(3):
            pts = [r for r in runs if r[4] == g]
            ax.scatter([r[1] for r in pts], [r[col] for r in pts], s=28, color=SERIES[g], marker=MARKERS[g],
                       edgecolor=SURFACE, linewidth=1.0, zorder=3, label=groups[g])
        for r in runs:
            if r[0] in label_at[col]:
                ax.annotate(r[0], (r[1], r[col]), xytext=label_at[col][r[0]], textcoords='data', fontsize=6.3,
                            color=INK2, ha='left', va='center',
                            arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.5, shrinkA=1, shrinkB=3))
        ax.axhline(t, color=INK2, lw=0.9)
        ax.text(0.42, t + 0.8, 'teacher (this harness)', fontsize=6.5, color=INK2)
        ax.set_xscale('log')
        ax.set_xticks([0.5, 1, 2, 4, 8], ['0.5', '1', '2', '4', '8'])
        ax.minorticks_off()
        ax.set_xlim(0.35, 12)
        ax.set_title(title)
        ax.set_xlabel('Stage 1 validation loss (log scale)')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_ylim(52, 78)
    axes[1].set_ylim(26, 70)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.02), columnspacing=1.2,
               handletextpad=0.3, fontsize=7.5)
    fig.subplots_adjust(bottom=0.25, wspace=0.22)
    save(fig, 'loss_vs_accuracy')


# 4. From stage 1 to stage 2 (docs/experiments/stage-difference.md, stage2-config1.md, window-rope.md)
def stage1_to_stage2():
    models = [('First Lizard model', (57.6, 67.95), (39.0, 54.8)),
              ('Config 1 (paper recipe)', (57.3, 67.7), (36.5, 54.9)),
              ('window_rope', (61.5, 73.1), (43.3, 63.8))]
    teacher = (74.43, 65.32)
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.5))
    nudge = {(1, 0): (0.6, 0.7), (1, 1): (-0.6, -0.7), (1, 2): (0, 0),   # (panel, model): (start dy, end dy)
             (2, 0): (0, -0.9), (2, 1): (0, 0.9), (2, 2): (0, 0)}
    for ax, k, title, t in ((axes[0], 1, 'PIQA', teacher[0]), (axes[1], 2, 'ARC-Easy', teacher[1])):
        for m, (name, *vals) in enumerate(models):
            v = vals[k - 1]
            d0, d1 = nudge[(k, m)]
            ax.plot([0, 1], v, color=SERIES[m], marker='o', ms=5, mec=SURFACE, mew=1.2, label=name)
            ax.text(1.06, v[1] + d1, f'{v[1]:.1f}', fontsize=7, color=INK2, va='center')
            ax.text(-0.06, v[0] + d0, f'{v[0]:.1f}', fontsize=7, color=INK2, va='center', ha='right')
        ax.axhline(t, color=INK2, lw=0.9)
        ax.text(0.5, t + 0.7, f'teacher {t:.1f}', fontsize=6.8, color=INK2, ha='center')
        ax.set_xticks([0, 1], ['after stage 1', 'after stage 2'])
        ax.set_xlim(-0.35, 1.35)
        ax.set_title(title)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_ylim(54, 78)
    axes[1].set_ylim(33, 69)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(bottom=0.22, wspace=0.25)
    save(fig, 'stage1_to_stage2')


# 5. Share of the teacher score (docs/07-results.md section 6, docs/experiments/window-rope.md,
#    lolcats-control.md). Teacher in this harness: PIQA 74.43, ARC-Easy 65.32.
def percent_of_teacher():
    teacher = {'PIQA': 74.43, 'ARC-Easy': 65.32}
    models = [('First Lizard model,\nafter stage 2', 67.95, 54.8),
              ('window_rope,\nafter stage 1', 61.5, 43.3),
              ('window_rope,\nafter stage 2', 73.1, 63.8),
              ('LoLCATs attention,\nafter stage 1 only', 73.5, 62.8)]
    y = np.arange(len(models))[::-1]
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.5))
    for yi in y:
        ax.axhline(yi, color=GRID, lw=0.6, zorder=0)
    for k, (task, col) in enumerate((('PIQA', 1), ('ARC-Easy', 2))):
        vals = [100 * m[col] / teacher[task] for m in models]
        ax.scatter(vals, y, s=34, color=SERIES[k], marker=MARKERS[k], edgecolor=SURFACE, linewidth=1.0, zorder=3,
                   label=task)
        for v, yi in zip(vals, y):
            ax.annotate(f'{v:.1f}', (v, yi), xytext=(0, 6 if k == 0 else -11), textcoords='offset points',
                        ha='center', fontsize=6.8, color=INK2)
    ax.axvline(100, color=INK2, lw=0.9)
    ax.text(99.6, len(models) - 0.55, 'teacher = 100%', fontsize=6.8, color=INK2, ha='right')
    ax.set_yticks(y, [m[0] for m in models])
    ax.set_ylim(-0.6, len(models) - 0.3)
    ax.set_xlim(60, 104)
    ax.set_xlabel('Accuracy as a share of the teacher in this harness (%)')
    ax.grid(axis='y', visible=False)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.42), ncol=2)
    save(fig, 'percent_of_teacher')


# 6. Letters on all MMLU questions (docs/experiments/window-rope/mmlu_full_summary.md)
def mmlu_letters():
    letters = ['A', 'B', 'C', 'D']
    series = [('Right answers', [22.9, 24.7, 25.5, 26.9]),
              ('Teacher', [49.6, 26.1, 21.8, 2.6]),
              ('window_rope after stage 2', [42.3, 38.0, 18.2, 1.4])]
    x = np.arange(4)
    width = 0.25
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.62, 2.4))
    for k, (name, vals) in enumerate(series):
        bars = ax.bar(x + (k - 1) * width, vals, width * 0.92, color=SERIES[k], label=name, edgecolor=SURFACE, lw=0.6)
        bar_labels(ax, bars, dy=0.6, size=6.3)
    ax.set_xticks(x, [f'"{c}"' for c in letters])
    ax.set_ylabel('Share of the questions (%)')
    ax.set_ylim(0, 58)
    ax.grid(axis='x', visible=False)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7.2, columnspacing=1.0,
              handletextpad=0.4)
    save(fig, 'mmlu_letters')


# 7. MMLU by subject group (docs/experiments/window-rope.md, finding 18)
def mmlu_groups():
    groups = ['STEM\n(3,153)', 'Human-\nities\n(4,705)', 'Social\nsciences\n(3,077)', 'Other\n(3,107)']
    teacher = [28.3, 28.8, 32.1, 36.2]
    student = [25.4, 26.1, 27.5, 28.9]
    gap = [2.8, 2.7, 4.7, 7.2]
    x = np.arange(4)
    width = 0.32
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.62, 2.4))
    b1 = ax.bar(x - width / 2, teacher, width * 0.92, color=SERIES[1], label='Teacher', edgecolor=SURFACE, lw=0.6)
    b2 = ax.bar(x + width / 2, student, width * 0.92, color=SERIES[2], label='window_rope after stage 2',
                edgecolor=SURFACE, lw=0.6)
    bar_labels(ax, b1, dy=0.3, size=6.5)
    bar_labels(ax, b2, dy=0.3, size=6.5)
    for xi, g, t in zip(x, gap, teacher):
        ax.text(xi, t + 4.6, f'gap {g:.1f}', ha='center', fontsize=6.8, color=INK)
    ax.axhline(25, color=INK2, lw=0.9)
    ax.set_xlim(-1.0, 3.5)
    ax.text(-0.98, 25.6, 'chance', fontsize=6.5, color=INK2, ha='left', va='bottom')
    ax.set_xticks(x, groups)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 52)
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper left', ncol=2)
    save(fig, 'mmlu_groups')


# 8. MMLU for each subject: teacher against window_rope (data/mmlu_per_subject.csv)
def mmlu_subjects():
    rows = list(csv.DictReader(open(SRC / 'data' / 'mmlu_per_subject.csv')))
    t = np.array([float(r['teacher']) for r in rows])
    s = np.array([float(r['window_rope_stage2']) for r in rows])
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.55, 2.9))
    ax.plot([10, 55], [10, 55], color=AXIS, lw=0.9)
    ax.text(47, 49.5, 'same score', fontsize=6.5, color=INK2, rotation=40)
    ax.axhline(25, color=INK2, lw=0.7)
    ax.axvline(25, color=INK2, lw=0.7)
    ax.scatter(t, s, s=16, color=BLUE, edgecolor=SURFACE, linewidth=0.8, zorder=3)
    for r, ti, si in zip(rows, t, s):
        if ti - si > 13 or ti > 45:
            ax.annotate(r['subject'].replace('_', ' ').replace('high school', 'HS'), (ti, si), xytext=(3, -7),
                        textcoords='offset points', fontsize=5.8, color=INK2)
    ax.set_xlim(12, 55), ax.set_ylim(12, 55)
    ax.set_xlabel('Teacher accuracy (%)')
    ax.set_ylabel('window_rope accuracy (%)')
    ax.set_aspect('equal')
    save(fig, 'mmlu_subjects')


# 9. Weight that the gate keeps after n tokens: gamma ** n (a calculation)
def gate_decay():
    n = np.arange(0, 2049)
    gammas = [0.9, 0.99, 0.999, 0.9999]
    colors = [BLUE_RAMP[3], BLUE_RAMP[6], BLUE_RAMP[9], BLUE_RAMP[12]]
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.62, 2.4))
    for g, c in zip(gammas, colors):
        ax.plot(n, g ** n, color=c, label=f'$\\gamma = {g}$')
    ax.axvline(128, color=INK2, lw=0.9)
    ax.text(140, 2e-6, 'end of the window\n(128 tokens)', fontsize=6.5, color=INK2)
    ax.set_yscale('log')
    ax.set_ylim(1e-6, 2)
    ax.set_xlim(0, 2048)
    ax.set_xticks([0, 128, 512, 1024, 1536, 2048])
    ax.set_xlabel('Distance $n$ from the query (tokens)')
    ax.set_ylabel('Weight kept, $\\gamma^n$')
    ax.legend(loc='lower left', bbox_to_anchor=(0.62, 0.02))
    save(fig, 'gate_decay')


# 10. Timeline of the project (PR history of lolcats and jku-thesis, docs/18-gap-timeline.md)
def timeline():
    events = [  # date, label, group: 0 code and infrastructure, 1 training, 2 analysis and evaluation
        ('2026-09-24', 'Reference code clean-up (jku-thesis)', 0),
        ('2026-09-26', 'Reference implementation and tests (jku-thesis)', 0),
        ('2026-09-29', 'Port to LoLCATs, Docker image, Hugging Face Jobs', 0),
        ('2026-09-29', 'Run 0 and Run 1. NaN crash and fix', 1),
        ('2026-09-30', 'Run 2. Evaluation: MMLU 23.3, PIQA 68.0', 2),
        ('2026-10-01', 'Gap analysis 1. Stage difference. Temperature', 2),
        ('2026-10-01', 'Feature dimension 32. Paper recipe in bf16 and float32', 1),
        ('2026-10-02', 'Config 1, config 2, stage 2 on config 1', 1),
        ('2026-10-02', 'Layer-wise MSE, attention weights, gap analysis 2', 2),
        ('2026-10-03', 'Lizard attention v2 with options', 0),
        ('2026-10-04', 'C1, R1, R1b. Liger and GLA review', 1),
        ('2026-10-08', 'Timeline and literature search', 2),
        ('2026-10-09', 'LoLCATs control. Position MSE. Branch fit', 2),
        ('2026-10-10', 'window_rope stage 1 and 2. hybrid. Teacher baseline', 1),
        ('2026-10-11', 'MMLU on all questions: 27.2 against 31.7', 2),
    ]
    names = ['Code and infrastructure', 'Training runs', 'Evaluation and analysis']
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 3.6))
    start = dt.date(2026, 9, 23)
    ys = np.arange(len(events))[::-1]
    for y, (d, label, g) in zip(ys, events):
        day = (dt.date.fromisoformat(d) - start).days
        ax.scatter(day, y, s=30, color=SERIES[g], marker=MARKERS[g], edgecolor=SURFACE, linewidth=1.0, zorder=3)
        ax.text(day + 0.45, y, label, va='center', fontsize=6.8, color=INK)
    ticks = [dt.date(2026, 9, 24) + dt.timedelta(days=k) for k in range(0, 19, 3)]
    ax.set_xticks([(t - start).days for t in ticks], [t.strftime('%d %b') for t in ticks])
    ax.set_xlim(0, 28)
    ax.set_yticks([])
    ax.set_ylim(-1, len(events))
    ax.grid(axis='y', visible=False)
    ax.spines['left'].set_visible(False)
    for g, name in enumerate(names):
        ax.scatter([], [], s=30, color=SERIES[g], marker=MARKERS[g], label=name)
    ax.legend(loc='lower left', bbox_to_anchor=(0, -0.3), ncol=3)
    save(fig, 'timeline')


# 11-14. Redraws of four XAI figures of the project notes, from their JSON files in docs/experiments
EXP = SRC.parent.parent / 'experiments'


def _load(folder, name):
    import json
    return json.load(open(EXP / folder / f'{name}.json'))


# 11. Layer-wise MSE of the Lizard stage 1 runs and the LoLCATs control (docs/experiments/xai-layer-wise-mse)
def xai_layer_mse():
    lizard = ['fd128_lolcats', 'fd32_lolcats', 'fd32_paper_bf16', 'fd32_paper_fp32', 'config1', 'v2_alphahead',
              'v2_perhead_hybrid']
    control = _load('xai-layer-wise-mse', 'lolcats_attention_fd128')
    layers = np.arange(16)
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.45))
    for ax, key, scale, title in ((axes[0], 'mse', 1000.0, '(a) 1000 × MSE (scale of the stage 1 loss)'),
                                  (axes[1], 'relative_mse', 1.0, '(b) Relative MSE (log scale)')):
        for k, name in enumerate(lizard):
            d = _load('xai-layer-wise-mse', name)
            ax.plot(layers, [l[key] * scale for l in d['layers']], color=AXIS, lw=1.0,
                    label='7 Lizard runs without RoPE' if k == 0 else None)
        ax.plot(layers, [l[key] * scale for l in control['layers']], color=JKU_BLUE, lw=1.8, marker='o', ms=3.5,
                mec=SURFACE, mew=0.8, label='LoLCATs attention (control)')
        ax.set_title(title)
        ax.set_xlabel('Layer')
        ax.set_xticks([0, 3, 6, 9, 12, 15])
    axes[1].set_yscale('log')
    axes[0].legend(loc='upper left')
    fig.subplots_adjust(wspace=0.28)
    save(fig, 'xai_layer_mse')


# 12. Layer-wise MSE of window_rope against Run 1 and the control (docs/experiments/xai-layer-wise-mse)
def rope_layer_mse():
    run1 = _load('xai-layer-wise-mse', 'fd128_lolcats')
    rope = _load('xai-layer-wise-mse', 'v2_window_rope')
    control = _load('xai-layer-wise-mse', 'lolcats_attention_fd128')
    layers = np.arange(16)
    m1 = np.array([l['mse'] for l in run1['layers']]) * 1000
    mr = np.array([l['mse'] for l in rope['layers']]) * 1000
    mc = np.array([l['mse'] for l in control['layers']]) * 1000
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.45))
    ax = axes[0]
    for vals, name, k in ((m1, 'Lizard Run 1', 0), (mr, 'window_rope', 1), (mc, 'LoLCATs control', 2)):
        ax.plot(layers, vals, color=SERIES[k], marker=MARKERS[k], ms=3.5, mec=SURFACE, mew=0.8, label=name)
    ax.set_title('(a) 1000 × MSE for each layer')
    ax.set_xlabel('Layer')
    ax.set_xticks([0, 3, 6, 9, 12, 15])
    ax.legend(loc='upper left')
    ax = axes[1]
    factor = m1 / mr
    bars = ax.bar(layers, factor, 0.62, color=JKU_GREEN, edgecolor=SURFACE, lw=0.6)
    for i in (0, 15):
        ax.text(i, factor[i] + 0.1, f'{factor[i]:.1f}×', ha='center', va='bottom', fontsize=7, color=INK2)
    ax.set_title('(b) Run 1 ÷ window_rope')
    ax.set_xlabel('Layer')
    ax.set_xticks([0, 3, 6, 9, 12, 15])
    ax.set_ylim(0, 6.2)
    ax.grid(axis='x', visible=False)
    fig.subplots_adjust(wspace=0.28)
    save(fig, 'rope_layer_mse')


def _heatmap(ax, matrix, cols, title, norm, fmt):
    ax.imshow(matrix, cmap=SEQ, norm=norm, aspect='auto', interpolation='nearest')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            shade = norm(v) if callable(norm) else v
            ax.text(j, i, fmt(v), ha='center', va='center', fontsize=5.2,
                    color='white' if shade > 0.55 else INK)
    ax.set_xticks(range(len(cols)), cols, fontsize=6.8)
    ax.set_yticks(range(matrix.shape[0]), range(matrix.shape[0]), fontsize=6.5)
    ax.set_title(title, fontsize=8)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


# 13. Branch decomposition at positions 0-127 (docs/experiments/xai-branch-fit)
def rope_branch_fit():
    from matplotlib.colors import LogNorm
    sets = ['G', 'W', 'WR', 'G+W', 'G+WR', 'trained']
    norm = LogNorm(vmin=1e-3, vmax=1.0, clip=True)
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 3.5))
    for ax, name, title in ((axes[0], 'fd128_lolcats', 'Lizard Run 1'), (axes[1], 'v2_window_rope', 'window_rope')):
        d = _load('xai-branch-fit', name)
        m = np.array([[l['fits']['0–127'][s]['relative_error'] for s in sets] for l in d['layers']])
        _heatmap(ax, m, sets, title, norm, lambda v: f'{v:.2g}' if v >= 0.001 else '<.001')
    axes[0].set_ylabel('Layer')
    fig.subplots_adjust(wspace=0.12)
    save(fig, 'rope_branch_fit')


# 14. Relative MSE by layer and position bucket (docs/experiments/xai-position-mse)
def rope_position_heatmap():
    from matplotlib.colors import Normalize
    buckets = ['0–127', '128–511', '512–2047']
    norm = Normalize(vmin=0.0, vmax=0.72)
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH * 0.82, 3.4))
    for ax, name, title in ((axes[0], 'fd128_lolcats', 'Lizard Run 1'), (axes[1], 'v2_window_rope', 'window_rope')):
        d = _load('xai-position-mse', name)
        m = np.array([[b['relative_mse'] for b in l['buckets']] for l in d['layers']])
        _heatmap(ax, m, buckets, title, norm, lambda v: f'{v:.2f}')
        ax.set_xlabel('Query positions', fontsize=7.5)
    axes[0].set_ylabel('Layer')
    fig.subplots_adjust(wspace=0.15)
    save(fig, 'rope_position_heatmap')


if __name__ == '__main__':
    attention_patterns()
    scores_first_model()
    loss_vs_accuracy()
    stage1_to_stage2()
    percent_of_teacher()
    mmlu_letters()
    mmlu_groups()
    mmlu_subjects()
    gate_decay()
    timeline()
    xai_layer_mse()
    rope_layer_mse()
    rope_branch_fit()
    rope_position_heatmap()
