"""Paper Figure 1 (file fig2_benchmark_skill.png): benchmark construction funnel and held-out skill, intervention hidden vs shown.

  python scripts/cr/figures/fig_fig2_benchmark_skill.py

(a) Trials harvested and trials with a parseable primary outcome from data/v2.1/datacard_v2.1.json; trials with arms passing
    inclusion from data/datacard.json (arm_trials; the v2.1 datacard does not restate this arm-level count and the arm
    inclusion rule is unchanged in v2.1); trials contributing a v2.1 contrast = distinct nct_id in the final v2.1 file.
(b) Skill against the global training mean on the v2.1 binary test split (results/cr/main.csv): registry GBT five-seed mean
    (skill_blind = intervention hidden, skill_full = shown) and Claude Sonnet 5 protocol-only; 95% trial-level bootstrap
    intervals. Noise ceiling: results/cr/design_noise_ceiling.csv, binary, 'all test', ceiling_skill_vs_mean.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figcommon import *  # noqa: E402,F403


def funnel():
    card21 = json.load(open(os.path.join(ROOT, 'data', 'v2.1', 'datacard_v2.1.json')))
    card2 = json.load(open(os.path.join(ROOT, 'data', 'datacard.json')))
    final = pd.read_csv(os.path.join(ROOT, 'data', 'v2.1', 'benchmark_contrasts_v2.1_final.csv.gz'), usecols=['nct_id'])
    return [('Trials harvested\n(completed phase 2/3, results posted)', int(card21['trials_harvested'])),
            ('Parseable single-class\nnumeric primary outcome', int(card21['trials_parsed'])),
            ('Arms passing inclusion\n(n \u2265 20, valid outcome)', int(card2['arm_trials'])),
            ('Trials contributing a\nwithin-trial contrast (v2.1)', int(final.nct_id.nunique()))]


def main():
    apply_style()
    flow = funnel()
    M = cr('main'); b = M[(M.version == 'v2.1') & (M.family == 'binary')]
    g = b[b.arm == 'gbt'].iloc[0]; s = b[b.arm == 'sonnet_protocol'].iloc[0]
    nc = cr('design_noise_ceiling'); nc = nc[(nc.version == 'v2.1') & (nc.family == 'binary') & (nc.subset == 'all test')]
    ceil = float(nc.ceiling_skill_vs_mean.iloc[0])
    n_test, n_trials = int(g.n), int(g.trials)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(5.5, 1.72), gridspec_kw=dict(width_ratios=[1.0, 0.85], wspace=0.55))
    ys = np.arange(len(flow))[::-1]
    ax0.barh(ys, [v for _, v in flow], color=BAR, height=0.55)
    for y_, (_, v) in zip(ys, flow):
        ax0.text(v + 600, y_, f'{v:,}', va='center', fontsize=SIZES[2])
    ax0.set_yticks(ys); ax0.set_yticklabels([l for l, _ in flow], fontsize=SIZES[2]); ax0.set_xlim(0, 44000)
    ax0.set_xticks([0, 10000, 20000, 30000]); ax0.set_xticklabels(['0', '10k', '20k', '30k']); ax0.set_xlabel('trials')
    ax0.set_title('Benchmark\nconstruction', loc='left')

    groups = ['GBT', 'LM']
    vals = {(0, 'hidden'): (g.skill_blind, g.skill_blind_lo, g.skill_blind_hi), (0, 'shown'): (g.skill_full, g.skill_full_lo, g.skill_full_hi),
            (1, 'hidden'): (s.skill_blind, s.skill_blind_lo, s.skill_blind_hi), (1, 'shown'): (s.skill_full, s.skill_full_lo, s.skill_full_hi)}
    w = 0.32
    for i in range(2):
        for j, cond in enumerate(['hidden', 'shown']):
            est, lo, hi = vals[(i, cond)]; x = i + (j - 0.5) * w
            ax1.bar(x, est, w * 0.95, color=GREY if cond == 'hidden' else FOCAL, label=f'intervention {cond}' if i == 0 else None)
            ax1.plot([x, x], [lo, hi], color='#333', lw=1)
    ax1.axhline(ceil, color=ACC, ls='--', lw=1)
    ax1.annotate(f'noise ceiling {ceil:.2f}', (1.55, ceil), xytext=(0, -3), textcoords='offset points', fontsize=SIZES[1], color=ACC, ha='right', va='top')
    ax1.set_xticks(range(2)); ax1.set_xticklabels(groups, fontsize=SIZES[1]); ax1.set_ylim(0, 1); ax1.set_xlim(-0.6, 1.9)
    ax1.set_ylabel('skill vs global mean')
    ax1.legend(frameon=False, loc='upper left', bbox_to_anchor=(-0.02, 0.86), handlelength=1.0, handletextpad=0.4, labelspacing=0.25)
    ax1.set_title('Held-out accuracy\nhides the difference', loc='left')
    letter(ax0, 'a', x=-0.62, y=1.26); letter(ax1, 'b', x=-0.3, y=1.26)
    print('funnel', [(l.replace(chr(10), ' '), v) for l, v in flow])
    print(f'GBT hidden {g.skill_blind:.3f} [{g.skill_blind_lo:.3f}, {g.skill_blind_hi:.3f}] shown {g.skill_full:.3f} [{g.skill_full_lo:.3f}, {g.skill_full_hi:.3f}]')
    print(f'Sonnet protocol-only hidden {s.skill_blind:.3f} [{s.skill_blind_lo:.3f}, {s.skill_blind_hi:.3f}] shown {s.skill_full:.3f} [{s.skill_full_lo:.3f}, {s.skill_full_hi:.3f}]')
    print(f'noise ceiling {ceil:.3f}; n={n_test} contrasts / {n_trials} trials')
    save(fig, 'fig2_benchmark_skill.png')


if __name__ == '__main__':
    main()
