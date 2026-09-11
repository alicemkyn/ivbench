"""Regenerate the paper figures from results/ (no model fitting; reads only released result tables).

  python scripts/make_figures.py [--out figures]

  figures/fig2_benchmark_skill.png       paper Figure 1  (benchmark construction; held-out skill, intervention shown/hidden)
  figures/fig3_interventional_forest.png paper Figure 2  (interventional skill by system and stratum)
  figures/fig4_reliability.png           paper Figure 3  (conformal coverage under shift; abstention; LM interval coverage)
  figures/figA_probes.png                paper Figure 4  (feature groups; seen/unseen; LM gain by area)
File names predate the final figure numbering and are kept so that paper/main.tex compiles unchanged.
paper Figure 5 (NeurIPS 2025 workshop screen) is drawn from results/screen/neurips2025_wmhs_labelled.csv and shipped as
figures/wmhs_gap_map.png.
"""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), '..'); R = os.path.join(ROOT, 'results')
FOCAL, GREY, GREY2, ACC = '#2c5f8a', '#9a9a9a', '#8a8a8a', '#b5423b'
plt.rcParams.update({'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
                     'legend.fontsize': 6.5, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 120})


def letter(ax, s):
    ax.text(-0.08, 1.08, s, transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='right')


def load():
    d = {}
    d['ab'] = pd.read_csv(f'{R}/positive_control/v2_ablation_seeds.csv')
    d['arch'] = pd.read_csv(f'{R}/positive_control/v2_architectures.csv')
    d['mol'] = pd.read_csv(f'{R}/positive_control/v2_same_molecule.csv')
    d['fg'] = pd.read_csv(f'{R}/positive_control/v2_feature_groups.csv')
    d['su'] = pd.read_csv(f'{R}/positive_control/v2_seen_unseen.csv')
    d['nc'] = pd.read_csv(f'{R}/positive_control/v2_noise_ceiling.csv')
    d['cov'] = pd.read_csv(f'{R}/shift/v2_coverage_under_shift.csv')
    d['selg'] = pd.read_csv(f'{R}/shift/v2_selective_curves.csv')
    d['sell'] = pd.read_csv(f'{R}/llm/llm_v2_selective.csv')
    d['area'] = pd.read_csv(f'{R}/llm/llm_v2_by_area.csv')
    d['lm'] = json.load(open(f'{R}/llm/llm_canonical_numbers.json'))
    d['card'] = json.load(open(os.path.join(ROOT, 'data', 'datacard.json')))
    return d


def mean_ci(df):
    return df.skill.mean(), df.ci_low.mean(), df.ci_high.mean()


def fig_benchmark(d, out):
    ab, lm, card = d['ab'], d['lm'], d['card']
    b = ab[ab.family == 'binary']
    gb_b, gb_f = b[(b.model == 'blind_ctrl') & (b.reference == 'train_mean')], b[(b.model == 'full') & (b.reference == 'train_mean')]
    bars = [('Gradient boosting\n(registry-trained)', 'hidden', *mean_ci(gb_b)), ('Gradient boosting\n(registry-trained)', 'shown', *mean_ci(gb_f)),
            ('Language model\n(reads protocol)', 'hidden', *lm['sonnet']['B']), ('Language model\n(reads protocol)', 'shown', *lm['sonnet']['A'])]
    flow = [('Trials harvested\n(completed phase 2/3, results posted)', card['trials_harvested']),
            ('Parseable single-class\nnumeric primary outcome', card['trials_parsed']),
            ('Arms passing inclusion\n(n \u2265 20, valid outcome)', card['arm_trials']),
            ('Trials contributing a\nwithin-trial contrast', card['contrast_trials_final'])]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.4, 2.8), gridspec_kw=dict(width_ratios=[1.0, 1.15], wspace=0.7))
    ys = np.arange(len(flow))[::-1]
    ax0.barh(ys, [f[1] for f in flow], color='#3f6f99', height=0.55)
    for y_, (_, v) in zip(ys, flow): ax0.text(v + 600, y_, f'{v:,}', va='center', fontsize=7)
    ax0.set_yticks(ys); ax0.set_yticklabels([f[0] for f in flow], fontsize=6.8); ax0.set_xlim(0, 40000)
    ax0.set_xticks([0, 10000, 20000, 30000]); ax0.set_xticklabels(['0', '10k', '20k', '30k']); ax0.set_xlabel('trials'); ax0.set_title('Benchmark construction', loc='left')
    groups = list(dict.fromkeys(bb[0] for bb in bars)); w = 0.32
    for i, g in enumerate(groups):
        for j, cond in enumerate(['hidden', 'shown']):
            _, _, est, lo, hi = [bb for bb in bars if bb[0] == g and bb[1] == cond][0]
            ax1.bar(i + (j - 0.5) * w, est, w * 0.95, color=GREY if cond == 'hidden' else FOCAL, label=f'intervention {cond}' if i == 0 else None)
            ax1.plot([i + (j - 0.5) * w] * 2, [lo, hi], color='#333', lw=1)
    nc = d['nc']; ceil = nc[nc.family == 'binary'].ceiling_skill_vs_mean.iloc[0]
    ax1.axhline(ceil, color=ACC, ls='--', lw=1); ax1.annotate(f'noise ceiling {ceil:.2f}', (1.5, ceil), xytext=(0, 3), textcoords='offset points', fontsize=6.2, color=ACC, ha='right')
    ax1.set_xticks(range(len(groups))); ax1.set_xticklabels(groups, fontsize=6.5); ax1.set_ylim(0, 1); ax1.set_xlim(-0.7, 1.7)
    ax1.set_ylabel('skill vs global mean\n(risk difference, 1,020 test contrasts)'); ax1.legend(frameon=False, loc='upper left', bbox_to_anchor=(0, 0.86))
    ax1.set_title('Held-out accuracy hides the difference', loc='left'); letter(ax0, 'a'); letter(ax1, 'b')
    fig.savefig(os.path.join(out, 'fig2_benchmark_skill.png'), dpi=300, bbox_inches='tight'); plt.close(fig)


def fig_forest(d, out):
    ab, arch, mol, lm = d['ab'], d['arch'], d['mol'], d['lm']
    gb_iv = ab[(ab.family == 'binary') & (ab.model == 'full') & (ab.reference == 'blind_ctrl')]
    gb_c = ab[(ab.family == 'continuous') & (ab.model == 'full') & (ab.reference == 'blind_ctrl')]
    pb = arch[arch.arch.str.startswith('GBT on PubMedBERT')]
    sm = mol[(mol.normalization == 'union') & (mol.same_ctrl_kind) & (mol['mode'] == 'residual') & (mol.k == 3)].iloc[0]
    S, H = lm['sonnet'], lm['haiku']
    rows = [('Gradient boosting, binary risk difference (5 seeds)', *mean_ci(gb_iv), 'gbt'),
            ('Gradient boosting, continuous SMD', gb_c.skill.iloc[0], gb_c.ci_low.iloc[0], gb_c.ci_high.iloc[0], 'gbt'),
            ('PubMedBERT embeddings + gradient boosting (3 seeds)', pb.iv.mean(), pb.iv_lo.mean(), pb.iv_hi.mean(), 'gbt'),
            ('Same-molecule retrieval (prior trials of the drug)', sm.retrieval_interventional_skill, sm.ci_low, sm.ci_high, 'gbt'),
            ('LM (large), binary, all test contrasts', *S['IS'], 'lm'), ('LM (large), binary, trials completed after cutoff', *S['IS_post'], 'lm'),
            ('LM (large), continuous SMD', *lm['cont']['IS'], 'lm'), ('LM (small), binary, all test contrasts', *H['IS'], 'lm'),
            ('LM (small), binary, trials completed after cutoff', *H['IS_post'], 'lm')]
    fig, ax = plt.subplots(figsize=(7.4, 3.6)); ys = list(range(len(rows)))[::-1]
    for y_, (lab, est, lo, hi, kind) in zip(ys, rows):
        c = FOCAL if kind == 'lm' else GREY2
        ax.plot([lo, hi], [y_, y_], color=c, lw=2.2, solid_capstyle='round'); ax.plot(est, y_, 'o', color=c, ms=6.5, zorder=3)
    ax.axvline(0, color='#333', lw=1); ax.axvspan(-0.35, 0, color='#eeeeee', zorder=0); ax.axhline(4.5, color='#bbb', lw=0.8, ls=':')
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlim(-0.3, 0.95); ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
    ax.set_xlabel('interventional skill  (error removed by revealing the intervention, vs. intervention-blind reference)')
    ax.text(0.93, 7.0, 'registry-trained models', color=GREY2, ha='right', va='center', fontsize=6.5)
    ax.text(0.93, 2.0, 'protocol-reading language models', color=FOCAL, ha='right', va='center', fontsize=6.5)
    ax.set_title('Registry-trained simulators do not use the intervention; language models do', loc='left')
    fig.savefig(os.path.join(out, 'fig3_interventional_forest.png'), dpi=300, bbox_inches='tight'); plt.close(fig)


def fig_reliability(d, out):
    cv = d['cov'][d['cov'].recipe == 'crossfit']; selg = d['selg']; sell = d['sell']; lm = d['lm']
    order = ['in_distribution', 'temporal_2023plus'] + sorted(r for r in cv.regime.unique() if r.startswith('loao'))
    short = {'in_distribution': 'in-dist.', 'temporal_2023plus': '2023+'}
    short.update({r: r.replace('loao_', '').replace('immune_inflam', 'immune').replace('neuro_psych', 'neuro').replace('cardiometabolic', 'cardio') for r in order if r.startswith('loao')})
    fig = plt.figure(figsize=(7.4, 3.0)); gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1.0, 0.7], wspace=0.55)
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))
    for i, rg in enumerate(order):
        for j, (iv, c) in enumerate([('constant', GREY2), ('normalized', FOCAL)]):
            rr = cv[(cv.regime == rg) & (cv.interval == iv)].iloc[0]; x = i + (j - 0.5) * 0.24
            axA.plot(x, rr.coverage, 'o', color=c, ms=4.5, label=f'{iv} width' if i == 0 else None); axA.plot([x, x], [rr.coverage_lo, rr.coverage_hi], color=c, lw=1)
    axA.axhline(0.9, color=ACC, ls='--', lw=1); axA.text(len(order) - 0.55, 0.906, 'nominal 90%', color=ACC, fontsize=6, ha='right')
    axA.axvline(1.5, color='#ccc', lw=0.8, ls=':'); axA.text(4.5, 0.985, 'held-out therapeutic area', color=GREY2, fontsize=6.5, ha='center', va='top')
    axA.set_xticks(range(len(order))); axA.set_xticklabels([short[r] for r in order], rotation=35, ha='right'); axA.set_ylim(0.7, 1.0)
    axA.set_ylabel('coverage of 90% conformal interval'); axA.legend(frameon=False, loc='lower left', ncol=2); axA.set_title('Conformal coverage slips under shift', loc='left')
    sg = selg[(selg.recipe == 'crossfit') & (selg.regime == 'temporal_2023plus')].sort_values('coverage')
    axB.plot(sg.coverage, sg.mae, 'o-', color=GREY2, ms=3.5, label='gradient boosting'); axB.plot(sell.coverage, sell.mae, 's-', color=FOCAL, ms=4.5, label='language model')
    axB.set_xlabel('fraction answered'); axB.set_ylabel('MAE (risk difference)'); axB.set_ylim(0, 0.16); axB.set_xticks([0.25, 0.5, 0.75, 1.0]); axB.legend(frameon=False, loc='upper left')
    axB.set_title('Abstention buys accuracy', loc='left')
    covA, covB = lm['sonnet'].get('cov_A', None), lm['sonnet'].get('cov_B', None)
    if covA is None:
        s = pd.read_csv(f'{R}/llm/llm_models_summary.csv'); r0 = s[(s.model == 'claude-sonnet-5') & (s.family == 'binary')].iloc[0]; covA, covB = r0.cov_A, r0.cov_B
    axC.bar([0, 1], [covA, covB], color=[FOCAL, GREY], width=0.6)
    for x_, v_ in zip([0, 1], [covA, covB]): axC.annotate(f'{v_:.2f}', (x_, v_), xytext=(0, 3), textcoords='offset points', ha='center', fontsize=6.5)
    axC.axhline(0.9, color=ACC, ls='--', lw=1); axC.set_ylim(0.5, 1.0); axC.set_xticks([0, 1]); axC.set_xticklabels(['shown', 'hidden']); axC.set_xlabel('LM, intervention')
    axC.set_ylabel('coverage of LM-stated 90% interval'); axC.set_title('LM overconfident', loc='left')
    for a_, l_ in zip((axA, axB, axC), 'abc'): letter(a_, l_)
    fig.savefig(os.path.join(out, 'fig4_reliability.png'), dpi=300, bbox_inches='tight'); plt.close(fig)


def fig_probes(d, out):
    fg = d['fg'][d['fg'].reference == 'design-only'].copy(); fg['label'] = fg.model.str.replace('design \\+ ', '', regex=True)
    su = d['su'].groupby(['variant', 'stratum']).agg(est=('interventional_skill', 'mean'), lo=('ci_low', 'mean'), hi=('ci_high', 'mean'), n=('n', 'first')).reset_index()
    ar = d['area']
    fig = plt.figure(figsize=(7.4, 2.7)); gs = fig.add_gridspec(1, 3, wspace=0.9); a1, a2, a3 = (fig.add_subplot(gs[0, i]) for i in range(3))
    ys = list(range(len(fg)))[::-1]
    for y_, r in zip(ys, fg.itertuples()):
        c = FOCAL if 'intervention' in r.label else GREY2; a1.plot([r.ci_low, r.ci_high], [y_, y_], color=c, lw=2.2); a1.plot(r.skill, y_, 'o', color=c, ms=5.5)
    a1.axvline(0, color='#333', lw=1); a1.set_yticks(ys); a1.set_yticklabels(fg.label); a1.set_xlabel('skill gain over design-only model'); a1.set_title('What carries the signal', loc='left')
    order = [('exact normalized name', 'seen'), ('exact normalized name', 'unseen'), ('token overlap', 'seen'), ('token overlap', 'unseen')]
    ys = list(range(len(order)))[::-1]
    for y_, (v, s) in zip(ys, order):
        rr = su[(su.variant.str.startswith(v.split()[0])) & (su.stratum == s)].iloc[0]
        a2.plot([rr.lo, rr.hi], [y_, y_], color=GREY2, lw=2.2); a2.plot(rr.est, y_, 'o', color=GREY2, ms=5.5); a2.text(rr.hi + 0.03, y_, f'n={int(rr.n)}', va='center', fontsize=6.5)
    a2.axvline(0, color='#333', lw=1); a2.set_yticks(ys); a2.set_yticklabels([f'{v.replace(" normalized", "")}\n{s}' for v, s in order]); a2.set_xlim(-0.35, 0.35)
    a2.set_xlabel('interventional skill (GBT)'); a2.set_title('Not an unseen-drug artefact', loc='left')
    ys = list(range(len(ar)))[::-1]
    for y_, r in zip(ys, ar.itertuples()):
        a3.plot([r.lo, r.hi], [y_, y_], color=FOCAL, lw=2.2); a3.plot(r.interventional, y_, 'o', color=FOCAL, ms=5.5); a3.text(r.hi + 0.04, y_, f'n={int(r.n)}', va='center', fontsize=6.5)
    a3.axvline(0, color='#333', lw=1); a3.set_yticks(ys); a3.set_yticklabels(ar.area.str.replace('_', ' / ')); a3.set_xlim(-0.5, 1.0)
    a3.set_xlabel('interventional skill (LM)'); a3.set_title('LM gain by therapeutic area', loc='left')
    for a_ in (a1, a2, a3): a_.spines['left'].set_visible(False); a_.tick_params(axis='y', length=0)
    for a_, l_ in zip((a1, a2, a3), 'abc'): letter(a_, l_)
    fig.savefig(os.path.join(out, 'figA_probes.png'), dpi=300, bbox_inches='tight'); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=os.path.join(ROOT, 'figures')); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); d = load()
    for f in (fig_benchmark, fig_forest, fig_reliability, fig_probes):
        f(d, a.out); print('wrote', f.__name__)


if __name__ == '__main__':
    main()
