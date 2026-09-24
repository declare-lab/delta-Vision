"""Additional paired contrasts and per-layer fit plots; consumes completed runs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(root):
    result = json.loads((root / 'results.json').read_text())
    rows = [json.loads(line) for shard in range(8) for line in (root / f'eval_{shard}.jsonl').open()]
    contrasts = {}
    doc = ['# Conditional information and trajectory replacement', '',
           'All inputs are initial E_i. Width 2560; frozen Qwen3-VL-4B; FA2; DeepStack off.', '',
           '| Dataset | Contrast | Accuracy change (pp) | Paired 95% CI |',
           '|---|---|---:|---|']
    for name in result:
        rr = [r for r in rows if r['benchmark'] == name]
        n = len(rr)
        indices = np.random.default_rng(44).integers(0, n, (10000, n))
        contrasts[name] = {}
        for student, reference in [('hidden_mlp', 'hidden_mean'), ('hidden_mlp', 'hidden_identity'),
                                   ('attn_mlp', 'cross_mean'), ('attn_mlp', 'attn_mean')]:
            delta = np.array([r['predictions'][student]['score'] - r['predictions'][reference]['score'] for r in rr])
            ci = np.quantile(delta[indices].mean(1) * 100, [.025, .975])
            key = f'{student} - {reference}'
            contrasts[name][key] = dict(delta_pp=float(delta.mean() * 100), ci95_pp=ci.tolist(),
                                       helped=int((delta > 0).sum()), harmed=int((delta < 0).sum()))
            doc.append(f'| {name} | {key} | {delta.mean()*100:+.2f} | [{ci[0]:+.2f}, {ci[1]:+.2f}] |')
    (root / 'paired_contrasts.json').write_text(json.dumps(contrasts, indent=2) + '\n')
    (root / 'PAIRED_CONTRASTS.md').write_text('\n'.join(doc) + '\n')
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), sharex='row')
    for j, name in enumerate(('sqa', 'realworldqa', 'mmstar')):
        for i, head in enumerate(('cross', 'hidden')):
            ax = axes[i, j]
            for which, style in [('mlp', '-'), ('mean', '--')]:
                fit = result[name]['fit'][which]
                xs = np.arange(35) + (1 if head == 'hidden' else 0)
                ys = [fit[f'{head}_{l}']['r2'] for l in range(35)]
                ax.plot(xs, ys, style, label=which)
            ax.axhline(0, color='gray', lw=.7)
            ax.set_title(f'{name}: {head}')
            ax.set_xlabel('Decoder layer (0-based)')
            ax.set_ylabel('R² (held-out, original units)')
            ax.grid(alpha=.2)
            ax.legend()
    fig.tight_layout()
    fig.savefig(root / 'fit_by_layer.pdf', bbox_inches='tight')
    fig.savefig(root / 'fit_by_layer.png', dpi=180, bbox_inches='tight')


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('root', type=Path)
    main(p.parse_args().root)
