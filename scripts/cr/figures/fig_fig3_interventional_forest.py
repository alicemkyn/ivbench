"""Paper Figure 2 (file fig3_interventional_forest.png): interventional skill by system, benchmark v2.1.

  python scripts/cr/figures/fig_fig3_interventional_forest.py

Rows (all v2.1 binary risk difference, 878 contrasts / 655 trials, unless stated), point = interventional skill IS,
bar = 95% trial-level bootstrap interval:
  registry-trained: GBT binary (5-seed mean, results/cr/main.csv); GBT continuous SMD (5-seed mean, main.csv);
    GBT on PubMedBERT embeddings (5-seed mean, results/cr/arch.csv); same-molecule retrieval, name-keyed primary variant
    (results/cr/mol.csv: normalization 'name', same control kind, residual mode, k=3; evaluated on the matched subset).
  protocol-reading language models: Claude Sonnet 5, Claude Haiku 4.5, Llama 3.1 70B Instruct, protocol-only (main.csv);
    Claude Sonnet 5 continuous SMD, posted-record prompt (main.csv); Llama 3.1 70B protocol-only on the uncontaminated
    stratum (results/cr/exposure.csv, set v2.1, stratum 'uncontaminated').
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figcommon import *  # noqa: E402,F403


def rows():
    M = cr('main'); M = M[M.version == 'v2.1']
    A = cr('arch'); MO = cr('mol'); E = cr('exposure')
    pick = lambda fam, arm: M[(M.family == fam) & (M.arm == arm)].iloc[0]
    gb = pick('binary', 'gbt'); gc = pick('continuous', 'gbt')
    pb = A[A.arch.str.startswith('GBT on PubMedBERT')].iloc[0]
    sm = MO[(MO.version == 'v2.1') & (MO.normalization == 'name') & (MO.same_ctrl_kind) & (MO['mode'] == 'residual') & (MO.k == 3)].iloc[0]
    so, ha, ll = pick('binary', 'sonnet_protocol'), pick('binary', 'haiku_protocol'), pick('binary', 'llama_protocol')
    sc = pick('continuous', 'sonnet_cont_posted')
    un = E[(E['set'] == 'v2.1') & (E.stratum == 'uncontaminated') & (E.prompt == 'protocol-only')].iloc[0]
    return [(f'Gradient boosting, binary RD (n={int(gb.n):,}, 5 seeds)', gb.IS, gb.IS_lo, gb.IS_hi, 'gbt'),
            (f'Gradient boosting, continuous SMD (n={int(gc.n):,}, 5 seeds)', gc.IS, gc.IS_lo, gc.IS_hi, 'gbt'),
            (f'PubMedBERT + gradient boosting ({int(pb.seeds)} seeds)', pb.IS, pb.IS_lo, pb.IS_hi, 'gbt'),
            (f'Same-molecule retrieval, name-keyed (n={int(sm.n_subset)})',
             sm.retrieval_interventional_skill, sm.ci_low, sm.ci_high, 'gbt'),
            ('Claude Sonnet 5, protocol-only', so.IS, so.IS_lo, so.IS_hi, 'lm'),
            ('Claude Haiku 4.5, protocol-only', ha.IS, ha.IS_lo, ha.IS_hi, 'lm'),
            ('Llama 3.1 70B, protocol-only', ll.IS, ll.IS_lo, ll.IS_hi, 'lm'),
            (f'Sonnet 5, continuous SMD, posted-record (n={int(sc.n):,})', sc.IS, sc.IS_lo, sc.IS_hi, 'lm'),
            (f'Llama 3.1 70B, protocol-only, uncontaminated (n={int(un.n)})', un.IS, un.IS_lo, un.IS_hi, 'lm')]


def main():
    apply_style()
    R = rows(); n_reg = sum(r[4] == 'gbt' for r in R)
    fig, ax = plt.subplots(figsize=(5.5, 2.15)); ys = list(range(len(R)))[::-1]
    for y_, (lab, est, lo, hi, kind) in zip(ys, R):
        interval(ax, y_, est, lo, hi, FOCAL if kind == 'lm' else GREY2, ms=5)
    xlo, xhi = -0.3, 0.5
    ax.axvline(0, color='#333', lw=1); ax.axvspan(xlo, 0, color='#eeeeee', zorder=0)
    ax.axhline(len(R) - n_reg - 0.5, color='#bbb', lw=0.8, ls=':')
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in R], fontsize=SIZES[1]); ax.set_xlim(xlo, xhi); ax.set_ylim(-0.6, len(R) - 0.4)
    for t_, (lab, est, lo, hi, kind) in zip(ax.get_yticklabels(), R):
        t_.set_color(FOCAL if kind == 'lm' else '#555555')
    ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
    ax.set_xlabel('interventional skill')
    ax.set_title('Registry-trained simulators do not use\nthe intervention; language models do', loc='left')
    for lab, est, lo, hi, kind in R:
        print(f'{kind:3s} {est:+.3f} [{lo:+.3f}, {hi:+.3f}]  {lab.replace(chr(10), chr(32))}')
    save(fig, 'fig3_interventional_forest.png')


if __name__ == '__main__':
    main()
