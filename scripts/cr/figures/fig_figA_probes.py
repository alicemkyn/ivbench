"""Paper Figure 4 (file figA_probes.png): feature-group gains, seen vs unseen interventions, LM gain by therapeutic area (v2.1).

  python scripts/cr/figures/fig_figA_probes.py

(a) Skill gain of each 'design + <group>' registry model over the design-only model (results/cr/fg.csv, version v2.1,
    reference 'design-only'); 95% trial-level bootstrap intervals.
(b) Interventional skill of the registry GBT on test contrasts whose intervention was seen vs unseen in training
    (results/cr/seen.csv, v2.1, seed 'mean' rows; exact normalized name and token-Jaccard >= 0.5 variants).
(c) Interventional skill of Claude Sonnet 5 protocol-only by therapeutic area (results/cr/lmarea.csv, v2.1).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figcommon import *  # noqa: E402,F403


def main():
    apply_style()
    fg = cr('fg'); fg = fg[(fg.version == 'v2.1') & (fg.reference == 'design-only')].copy()
    fg['label'] = fg.model.str.replace('design \\+ ', '', regex=True)
    su = cr('seen'); su = su[(su.version == 'v2.1') & (su.seed.astype(str) == 'mean')]
    ar = cr('lmarea'); ar = ar[(ar.version == 'v2.1') & (ar.prompt == 'protocol-only')]

    fig = plt.figure(figsize=(7.4, 2.7)); gs = fig.add_gridspec(1, 3, wspace=0.9)
    a1, a2, a3 = (fig.add_subplot(gs[0, i]) for i in range(3))
    ys = list(range(len(fg)))[::-1]
    for y_, r in zip(ys, fg.itertuples()):
        interval(a1, y_, r.skill, r.ci_low, r.ci_high, FOCAL if 'intervention' in r.label else GREY2)
    a1.axvline(0, color='#333', lw=1); a1.set_yticks(ys); a1.set_yticklabels(fg.label); a1.set_xlabel('skill gain over design-only model')
    a1.set_title('What carries the signal', loc='left')

    order = [('exact normalized name', 'seen'), ('exact normalized name', 'unseen'), ('token-Jaccard>=0.5', 'seen'), ('token-Jaccard>=0.5', 'unseen')]
    lab = {'exact normalized name': 'exact name', 'token-Jaccard>=0.5': 'token overlap'}
    ys = list(range(len(order)))[::-1]; xmax = 0.4
    for y_, (v, s) in zip(ys, order):
        rr = su[(su.variant == v) & (su.stratum == s)].iloc[0]
        interval(a2, y_, rr.interventional_skill, rr.ci_low, rr.ci_high, GREY2)
        a2.text(xmax * 0.98, y_, f'n={int(rr.n)}', va='center', ha='right', fontsize=SIZES[1])
    a2.axvline(0, color='#333', lw=1); a2.set_yticks(ys); a2.set_yticklabels([f'{lab[v]}\n{s}' for v, s in order]); a2.set_xlim(-0.5, xmax); a2.set_xticks([-0.5, -0.25, 0, 0.25])
    a2.set_xlabel('interventional skill (GBT)'); a2.set_title('Seen molecules do not help', loc='left')

    ys = list(range(len(ar)))[::-1]; xmax3 = 1.2
    for y_, r in zip(ys, ar.itertuples()):
        interval(a3, y_, r.interventional, r.lo, r.hi, FOCAL)
        a3.text(xmax3 * 0.98, y_, f'n={int(r.n)}', va='center', ha='right', fontsize=SIZES[1])
    a3.axvline(0, color='#333', lw=1); a3.set_yticks(ys); a3.set_yticklabels(ar.area.str.replace('_', ' / ')); a3.set_xlim(-0.5, xmax3); a3.set_xticks([-0.5, 0, 0.5, 1.0])
    a3.set_xlabel('interventional skill (LM)'); a3.set_title('LM gain by therapeutic area', loc='left')
    for a_ in (a1, a2, a3):
        a_.spines['left'].set_visible(False); a_.tick_params(axis='y', length=0)
    for a_, l_ in zip((a1, a2, a3), 'abc'):
        letter(a_, l_)

    for r in fg.itertuples():
        print(f'(a) {r.label:24s} {r.skill:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]')
    for v, s in order:
        rr = su[(su.variant == v) & (su.stratum == s)].iloc[0]
        print(f'(b) {lab[v]:14s} {s:6s} n={int(rr.n)} IS {rr.interventional_skill:+.3f} [{rr.ci_low:+.3f}, {rr.ci_high:+.3f}]')
    for r in ar.itertuples():
        print(f'(c) {r.area:16s} n={int(r.n)} IS {r.interventional:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]')
    save(fig, 'figA_probes.png', width=None)  # appendix figure keeps its 7.4 in design and is scaled to the text width


if __name__ == '__main__':
    main()
