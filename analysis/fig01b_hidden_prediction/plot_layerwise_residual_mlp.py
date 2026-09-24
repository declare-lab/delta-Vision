"""Render the 2,000-step post-RMSNorm residual MLP results without GPU work."""
from pathlib import Path
import csv
import json
import hashlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'artifacts/diagnostics/initial_token_postnorm_all36_20260916/results.csv'
OUT = ROOT / 'artifacts/figures/layerwise_residual_mlp_20260917'
OUT.mkdir(parents=True, exist_ok=True)
DATASETS = [('realworldqa', 'RealWorldQA', '#0072B2'), ('mmstar', 'MMStar', '#D55E00'), ('sqa', 'SQA', '#009E73')]
with SOURCE.open() as handle:
    rows = list(csv.DictReader(handle))
data = {}
for key, _, _ in DATASETS:
    selected = sorted((r for r in rows if r['dataset'] == key), key=lambda r: int(r['layer']))
    assert [int(r['layer']) for r in selected] == list(range(36))
    data[key] = {field: np.array([float(r[field]) for r in selected]) for field in selected[0] if field != 'dataset'}
    assert data[key]['mlp_mse'][0] == 0

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.titlesize': 11, 'axes.labelsize': 10,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 240})


def style(ax):
    ax.set_xlim(0, 35)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 35])
    ax.set_xlabel('Layer index')
    ax.grid(axis='y', alpha=.2, linewidth=.6)
    ax.set_axisbelow(True)


def save(fig, stem):
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / f'{stem}.{extension}', bbox_inches='tight')
    plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.65))
for (key, label, color), marker in zip(DATASETS, ['o', 's', '^']):
    d = data[key]
    axes[0].plot(d['layer'], d['mlp_cosine'], color=color, label=label, lw=1.7,
                 marker=marker, markevery=5, ms=3.5)
    # Exact zero at layer 0 is explicitly omitted; no epsilon is substituted.
    axes[1].plot(d['layer'][1:], d['mlp_mse'][1:], color=color, label=label, lw=1.7,
                 marker=marker, markevery=5, ms=3.5)
axes[0].set(title='(a) Cosine similarity', ylabel='Cosine similarity', ylim=(.65, 1.015))
axes[1].set(title='(b) Reconstruction error', ylabel='MSE (log scale)', yscale='log', ylim=(5e-5, 20))
for ax in axes:
    style(ax)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.5, 1.015))
fig.text(.5, .015, 'Post-RMSNorm hidden reconstruction · 2,000 training steps · Layer 0 MSE = 0 (omitted on log axis)',
         ha='center', fontsize=8, color='#555555')
fig.subplots_adjust(top=.79, bottom=.20, wspace=.30)
save(fig, 'layerwise_mlp')

# Additional figure: the recorded no-MLP control is crucial for attributing
# reconstruction quality to the learned predictor rather than to E alone.
fig, axes = plt.subplots(2, 3, figsize=(10.8, 5.6), sharex=True, sharey='row')
for col, (key, label, color) in enumerate(DATASETS):
    d = data[key]
    for row, metric in enumerate(('cosine', 'mse')):
        ax = axes[row, col]
        sl = slice(None) if metric == 'cosine' else slice(1, None)
        ax.plot(d['layer'][sl], d['mlp_' + metric][sl], color=color, lw=1.8)
        ax.plot(d['layer'][sl], d['identity_' + metric][sl], color='#777777', lw=1.4, ls='--')
        style(ax)
        if row == 0:
            ax.set_title(label)
            ax.set_xlabel('')
            ax.set_ylim(0, 1.02)
        else:
            ax.set_yscale('log')
            ax.set_ylim(5e-5, 100)
axes[0, 0].set_ylabel('Cosine similarity')
axes[1, 0].set_ylabel('MSE (log scale)')
fig.legend([Line2D([], [], color='black', lw=1.8), Line2D([], [], color='#777777', ls='--', lw=1.4)],
           [r'Residual MLP: $\mathrm{RMSNorm}_\ell(E_i)+f_\ell(E_i)$',
            r'No MLP: $\mathrm{RMSNorm}_\ell(E_i)$'],
           loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.5, 1.015))
fig.text(.5, .013, 'Target: native post-RMSNorm visual hidden · Layer 0 MSE = 0 (omitted on log axes)',
         ha='center', fontsize=8, color='#555555')
fig.subplots_adjust(top=.88, bottom=.13, hspace=.16, wspace=.12)
save(fig, 'layerwise_mlp_with_control')

(OUT / 'provenance.json').write_text(json.dumps({'source': str(SOURCE),
    'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'metrics': 'full postnorm representation, equal image averaging; final step2000 checkpoint',
    'log_axis': 'layer0 exact-zero MSE excluded, no epsilon or smoothing',
    'error_bars': 'not plotted; aggregate source does not provide uncertainty'}, indent=2))
print(OUT)
