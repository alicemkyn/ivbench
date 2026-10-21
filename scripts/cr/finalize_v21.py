"""Final v2.1 benchmark: applies the author-approved post-repair edits to the output of build_v21.py.

  python scripts/cr/finalize_v21.py   (reads data/v2.1/benchmark_contrasts_v2.1.csv.gz, writes ..._final.csv.gz)

Edits (all approved by the author after the v2.1 re-audit; see results/v2.1/v21_repair_report.md):
  * Remove 8 test contrasts that failed the audit of all new or re-paired v2.1 test contrasts: 7 pairing failures
    (sub-group / cohort comparator: c21_0000, c21_0002, c21_0024; pooled group: k21_0005, k21_0006, k21_0021;
    same molecule, different schedule: k21_0007) and c21_0051, whose comparator value cannot be re-derived from the
    current record.
  * NCT01023581 (add-on design): the two v2.1 pairs of alogliptin monotherapy against metformin monotherapy have an
    active comparator; their ctrl_kind is set to active_comparator.
"""
import os
import pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
EXCLUDE = ['c21_0000', 'c21_0002', 'c21_0024', 'k21_0005', 'k21_0006', 'k21_0021', 'k21_0007', 'c21_0051']
RELABEL = [('NCT01023581', 'Alogliptin 12.5 BID', 'active_comparator'), ('NCT01023581', 'Alogliptin 25 QD', 'active_comparator')]


def main():
    src = os.path.join(ROOT, 'data', 'v2.1', 'benchmark_contrasts_v2.1.csv.gz')
    C = pd.read_csv(src, low_memory=False)
    for nct, tg, kind in RELABEL:
        m = (C.nct_id == nct) & (C.treat_group == tg)
        assert m.sum() == 1, (nct, tg, int(m.sum()))
        C.loc[m, 'ctrl_kind'] = kind
    assert C.cid.isin(EXCLUDE).sum() == len(EXCLUDE)
    C = C[~C.cid.isin(EXCLUDE)].reset_index(drop=True)
    C.to_csv(os.path.join(ROOT, 'data', 'v2.1', 'benchmark_contrasts_v2.1_final.csv.gz'), index=False)
    print(len(C), C.groupby(['family', 'split_time']).size().to_dict())


if __name__ == '__main__':
    main()
