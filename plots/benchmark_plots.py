"""Render the README figures from the checked-in benchmark CSVs."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / 'outputs' / 'figures'
TEAL, BLUE, ORANGE, INK = '#087f8c', '#4865cf', '#dc8536', '#233548'


def read_csv(name):
    with (ROOT / 'outputs' / name).open() as source:
        return list(csv.DictReader(source))


def save(fig, name, title, subtitle):
    fig.suptitle(title, x=0.07, y=0.98, ha='left', fontsize=21, fontweight='bold', color=INK)
    fig.text(0.07, 0.90, subtitle, fontsize=10, color='#566778')
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.16, top=0.77, wspace=0.24)
    fig.savefig(FIGURES / name, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(FIGURES / name)


def main():
    rows = read_csv('benchmark.csv')
    # A missing or mixed matrix must not silently become a misleading chart.
    expected = {(length, dtype, temp, graph)
                for length in ('', '1024', '8192', '32768')
                for dtype in ('bf16', 'fp8') for temp in (0.0, 0.5)
                for graph in (False, True)}
    data = {(r['prompt_tokens'], r['draft_dtype'], float(r['temperature']), r['graph_mode'] == 'True'): r
            for r in rows}
    assert len(rows) == len(data) and set(data) == expected, 'Expected the complete 32-row benchmark matrix'
    assert all(r['prefill_backend'] == 'flash_attention_3' and r['lookahead'] == '3' for r in rows)
    for r in rows:
        assert np.isclose(float(r['speedup']), float(r['ar_total_e2e_s']) / float(r['spec_total_e2e_s']))
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': INK,
        'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
        'figure.facecolor': '#f7f9fc', 'axes.facecolor': '#f7f9fc',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
        'axes.edgecolor': '#cad3df', 'axes.axisbelow': True,
        'grid.color': '#dfe5ee', 'grid.linewidth': 0.8, 'axes.grid': True,
        'axes.grid.axis': 'y', 'xtick.major.size': 0, 'ytick.major.size': 0,
        'legend.frameon': False,
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1), sharey=True)
    maximum = max(float(data['', dtype, temp, graph]['speedup'])
                  for dtype in ('bf16', 'fp8') for temp in (0.0, 0.5) for graph in (False, True))
    for ax, temp in zip(axes, (0.0, 0.5)):
        for graph, offset, color in ((False, -0.19, TEAL), (True, 0.19, BLUE)):
            values = [float(data['', dtype, temp, graph]['speedup']) for dtype in ('bf16', 'fp8')]
            bars = ax.bar(np.arange(2) + offset, values, width=0.34, color=color,
                          label='Both graphed' if graph else 'Both eager', zorder=3)
            ax.bar_label(bars, labels=[f'{v:.3f}×' for v in values], padding=6, fontsize=11)
        ax.axhline(1, color=INK, linestyle='--', linewidth=1)
        ax.set(xticks=[0, 1], xticklabels=['BF16 draft', 'FP8 draft'],
               title=f'Temperature {temp:g}', ylim=(0, maximum * 1.25))
    axes[0].set_ylabel('End-to-end speedup over target-only AR')
    axes[1].legend(loc='upper right', fontsize=10)
    save(fig, 'short_speedup.png', 'Speculation on short prompts',
         'H100 80GB  ·  FP8 Qwen3-14B target / Qwen3-1.7B draft  ·  lookahead 3  ·  64 new tokens  ·  18 requests per bar')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    lengths = ('1024', '8192', '32768')
    context_sizes = [int(length) for length in lengths]
    for ax, temp in zip(axes, (0.0, 0.5)):
        for dtype, color in (('bf16', TEAL), ('fp8', ORANGE)):
            for graph in (False, True):
                values = [float(data[length, dtype, temp, graph]['speedup']) for length in lengths]
                ax.plot(context_sizes, values, color=color, linestyle='-' if graph else '--',
                        marker='o' if graph else 's', linewidth=2, markersize=6,
                        label=f'{dtype.upper()} / {"graphs" if graph else "eager"}')
        ax.axhline(1, color=INK, linestyle=':', linewidth=1)
        ax.set(xticks=context_sizes, xticklabels=['1,024', '8,192', '32,768'],
               xlabel='Prompt tokens', title=f'Temperature {temp:g}', ylim=(0, None))
    axes[0].set_ylabel('End-to-end speedup over target-only AR')
    axes[1].legend(loc='lower left', ncol=2, fontsize=9)
    save(fig, 'context_speedup.png', 'How context length changes the payoff',
         'Synthetic repeated-text prompts  ·  64 new tokens  ·  3 repetitions  ·  includes eager prefill  ·  4 / 16 / 16 KV splits')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    stages = ('prefill', 'draft', 'verify', 'append')
    colors = ('#c8d5e8', TEAL, BLUE, ORANGE)
    for ax, dtype in zip(axes, ('bf16', 'fp8')):
        selected = [data[length, dtype, 0.0, True] for length in ('', *lengths)]
        times = np.array([[float(r[f'profile_{stage}_total_ms']) / int(r['profile_requests'])
                           for r in selected] for stage in stages])
        total = times.sum(axis=0)
        assert np.all(total > 0)
        bottom = np.zeros(4)
        for stage, color, values in zip(stages, colors, times / total):
            ax.bar(range(4), values * 100, bottom=bottom, color=color, width=0.65, label=stage.capitalize())
            bottom += values * 100
        for i, ms in enumerate(total):
            ax.text(i, 103, f'{ms / 1000:.2f} s', ha='center', fontsize=10)
        ax.set(xticks=range(4), xticklabels=['Short', '1K', '8K', '32K'],
               title=f'{dtype.upper()} draft', ylim=(0, 117))
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_formatter(PercentFormatter())
    axes[0].set_ylabel('Share of measured stage time')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=4, bbox_to_anchor=(0.52, 0.01))
    save(fig, 'stage_breakdown.png', 'Where speculative time goes',
         'Temperature 0 / CUDA graphs  ·  labels: summed stage time per request  ·  separate profiling pass; excludes time outside stages')

    splits = read_csv('kv_split_benchmark.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, queries, label in zip(axes, ('1', '4'), ('AR: 1 query token', 'Verify: 4 query tokens')):
        for count, color in zip(('1', '4', '8', '16'), ('#a6b2c3', ORANGE, BLUE, TEAL)):
            selected = [next(r for r in splits if r['model_shape'] == 'qwen3_14b'
                             and r['query_tokens'] == queries and r['kv_splits'] == count
                             and r['valid_kv_tokens'] == length) for length in lengths]
            ax.plot(context_sizes, [float(r['attention_us']) for r in selected],
                    color=color, marker='o', linewidth=2, label=f'{count} split(s)')
        ax.set(xticks=context_sizes, xticklabels=['1,024', '8,192', '32,768'],
               xlabel='Valid KV tokens', title=label, ylim=(0, None))
    axes[0].set_ylabel('Attention kernel latency (µs)')
    axes[0].legend(loc='upper left', fontsize=9)
    save(fig, 'kv_splits.png', 'Parallel KV splits help at long context',
         'Isolated FA3 kernel  ·  Qwen3-14B attention shape  ·  BF16 KV  ·  graphed  ·  2,000 calls per point; not end-to-end latency')


if __name__ == '__main__':
    main()
