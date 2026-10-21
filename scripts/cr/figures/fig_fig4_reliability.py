"""Paper Figure 3 (file fig4_reliability.png): conformal coverage under shift, selective prediction, LM-stated interval coverage.

  python scripts/cr/figures/fig_fig4_reliability.py

(a) Coverage of the 90% conformal interval, cross-fitted recipe only (results/cr/coverage.csv, recipe 'crossfit'), constant
    and normalized intervals, regimes in_distribution, temporal_2023plus and the leave-one-area-out regimes; bars are the
    trial-level bootstrap interval (coverage_lo/hi); dashed line = nominal 0.90.
(b) Selective prediction, MAE against fraction answered. Registry GBT: cross-fitted recipe on the temporal_2023plus (shifted)
    regime, recomputed here with scripts/cr/tables/_conformal.fit_regime (same code path and seed as tab_op.py); the
    abstention threshold is a quantile (0.05 .. 1.0) of the calibration (out-of-fold) error scale sigma, applied unchanged
    to the test set, so x is the realised answered fraction. The curve is written to results/cr/fig4b_selective_crossfit.csv.
    The calibration-chosen operating point for target MAE 0.10 is read from results/cr/op.csv (temporal_2023plus, crossfit,
    threshold chosen on calibration) and drawn as a distinct marker against the target line.
    Language model: Claude Sonnet 5 protocol-only on v2.1, results/cr/lmsel.csv (ranked by the model's own stated interval
    width; coverage 0.10 .. 1.0).
(c) Coverage of the LM-stated 90% interval, intervention shown (A) vs hidden (B), Claude Sonnet 5 protocol-only on v2.1,
    computed from results/llm/cr/protocol_{A,B}_predictions.csv for v2.1 'kept' cids and
    results/llm/cr/v21new_sonnet_binary_predictions.csv for repaired/added rows, assembled exactly as tab_main.py does
    (scripts/cr/tables/_common.lm_arm).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figcommon import *  # noqa: E402,F403
sys.path.insert(0, TABLES)
import _common as cm  # noqa: E402
import _conformal as cf  # noqa: E402

GRID = np.round(np.linspace(0.05, 1.0, 20), 3)
TARGET = 0.10


def gbt_curve():
    name, mk = cf.regimes('v2.1', which=('temporal_2023plus',))[0]
    pool, te_ = mk(); Rg = cf.fit_regime(name, pool, te_)
    d = Rg['crossfit']; y, p, s, sc = Rg['yte'], d['p'], d['sig_te'], d['sig_cal']; ae = np.abs(y - p)
    rows = []
    for q in GRID:
        cut = float(np.quantile(sc, q)); msk = s <= cut
        rows.append(dict(regime=name, recipe='crossfit', cal_quantile=q, sigma_cut=cut, answered_frac_test=float(msk.mean()),
                         n_answered=int(msk.sum()), mae_answered_test=float(ae[msk].mean()), mae_all_test=float(ae.mean()), n_test=len(y)))
    out = pd.DataFrame(rows); out['version'] = 'v2.1'
    out.to_csv(os.path.join(CR, 'fig4b_selective_crossfit.csv'), index=False)
    # consistency with the published operating point (same fit): the op.csv sigma cut must reproduce its answered fraction
    op = cr('op'); r = op[(op.regime == name) & (op.recipe == 'crossfit') & (op.threshold_chosen_on == 'calibration') & (op.target_mae == TARGET)].iloc[0]
    m2 = s <= r.sigma_cut
    assert abs(m2.mean() - r.answered_frac_test) < 1e-9 and abs(ae[m2].mean() - r.mae_answered_test) < 1e-9, 'op.csv not reproduced'
    return out, r


def lm_interval_coverage():
    D = cm.lm_arm('sonnet_protocol', 'v2.1'); y = D.truth.to_numpy()
    cov = {side: float(np.mean((y >= D[f'{side}_low']) & (y <= D[f'{side}_high']))) for side in 'AB'}
    return cov, len(D), int(D.nct_id.nunique())


def main():
    apply_style()
    cv = cr('coverage'); cv = cv[(cv.version == 'v2.1') & (cv.recipe == 'crossfit')]
    order = ['in_distribution', 'temporal_2023plus'] + sorted(r for r in cv.regime.unique() if r.startswith('loao'))
    short = {'in_distribution': 'in-dist.', 'temporal_2023plus': '2023+'}
    short.update({r: r.replace('loao_', '').replace('immune_inflam', 'immune').replace('neuro_psych', 'neuro').replace('cardiometabolic', 'cardio')
                  for r in order if r.startswith('loao')})
    curve, op = gbt_curve()
    sel = cr('lmsel'); sel = sel[(sel.version == 'v2.1') & (sel.prompt == 'protocol-only')].sort_values('coverage')
    cov, n_lm, t_lm = lm_interval_coverage()

    fig = plt.figure(figsize=(5.5, 2.05)); gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1.05, 0.62], wspace=0.55)
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))
    for i, rg in enumerate(order):
        for j, (iv, c) in enumerate([('constant', GREY2), ('normalized', FOCAL)]):
            rr = cv[(cv.regime == rg) & (cv.interval == iv)].iloc[0]; x = i + (j - 0.5) * 0.24
            axA.plot(x, rr.coverage, 'o', color=c, ms=4.5, label=f'{iv} width' if i == 0 else None)
            axA.plot([x, x], [rr.coverage_lo, rr.coverage_hi], color=c, lw=1)
    axA.axhline(0.9, color=ACC, ls='--', lw=1, label='nominal 90%')
    axA.axvline(1.5, color='#ccc', lw=0.8, ls=':'); axA.text(4.5, 1.0, 'held-out therapeutic area', color=GREY2, fontsize=SIZES[1], ha='center', va='top')
    axA.set_xticks(range(len(order))); axA.set_xticklabels([short[r] for r in order], rotation=35, ha='right'); axA.set_ylim(0.7, 1.01); axA.set_yticks([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    axA.set_xlim(-0.6, len(order) - 0.4)
    axA.set_ylabel('coverage, 90% conformal interval'); axA.legend(frameon=False, loc='lower left', ncol=1, handlelength=1.6, labelspacing=0.2, borderaxespad=0.2)
    axA.set_title('Conformal coverage\nslips under shift', loc='left')

    axB.plot(curve.answered_frac_test, curve.mae_answered_test, 'o-', color=GREY2, ms=3.5, label='gradient boosting')
    axB.plot(sel.coverage, sel.mae, 's-', color=FOCAL, ms=4.5, label='language model')
    axB.axhline(TARGET, color=ACC, ls='--', lw=1, label=f'target MAE {TARGET:.2f}')
    axB.plot(op.answered_frac_test, op.mae_answered_test, marker='D', color=ACC, ms=5.5, mec='white', mew=0.6, ls='none', zorder=5,
             label='calibration-chosen\noperating point')
    axB.set_xlabel('fraction answered'); axB.set_ylabel('MAE (risk difference)'); axB.set_ylim(0, 0.22); axB.set_yticks(np.arange(0, 0.201, 0.04)); axB.set_xlim(0, 1.05)
    axB.set_xticks([0.25, 0.5, 0.75, 1.0]); axB.legend(frameon=False, loc='upper left', handlelength=1.6, borderaxespad=0.2, labelspacing=0.3)
    axB.set_title('Ranking helps; error\ncontrol does not transfer', loc='left')

    axC.bar([0, 1], [cov['A'], cov['B']], color=[FOCAL, GREY], width=0.6)
    for x_, v_ in zip([0, 1], [cov['A'], cov['B']]):
        axC.annotate(f'{v_:.2f}', (x_, v_), xytext=(0, 3), textcoords='offset points', ha='center', fontsize=SIZES[1])
    axC.axhline(0.9, color=ACC, ls='--', lw=1); axC.set_ylim(0.5, 1.0); axC.set_xticks([0, 1]); axC.set_xticklabels(['shown', 'hidden'])
    axC.set_xlabel('intervention'); axC.set_ylabel('coverage, LM-stated 90% interval'); axC.set_title('LM intervals\noverconfident', loc='left')
    for a_, l_ in zip((axA, axB, axC), 'abc'):
        letter(a_, l_, x=-0.12, y=1.24)

    for rg in order:
        for iv in ('constant', 'normalized'):
            rr = cv[(cv.regime == rg) & (cv.interval == iv)].iloc[0]
            print(f'(a) {rg:22s} {iv:10s} {rr.coverage:.3f} [{rr.coverage_lo:.3f}, {rr.coverage_hi:.3f}] n={int(rr.n_test)}')
    print('(b) GBT curve (answered, MAE):', [(round(a, 3), round(b, 3)) for a, b in zip(curve.answered_frac_test, curve.mae_answered_test)])
    print(f'(b) operating point: answered {op.answered_frac_test:.3f}, realised MAE {op.mae_answered_test:.3f}, target {TARGET}, meets {bool(op.meets_target_on_test)}')
    print('(b) LM curve (coverage, MAE):', [(round(a, 2), round(b, 3)) for a, b in zip(sel.coverage, sel.mae)])
    print(f'(c) LM-stated 90% interval coverage: shown {cov["A"]:.3f}, hidden {cov["B"]:.3f} (n={n_lm} contrasts / {t_lm} trials)')
    save(fig, 'fig4_reliability.png')


if __name__ == '__main__':
    main()
