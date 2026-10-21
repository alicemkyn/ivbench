# ivbench: an intervention-blind test of interventional validity for trial simulators

<p align="center">
  <img src="figures/graphical_abstract.png" alt="Graphical abstract: a benchmark of real within-trial contrasts, an intervention-blind control, and the result that registry-trained simulators have no interventional skill while protocol-reading language models have some" width="100%">
</p>

Code and data release accompanying the camera-ready paper

> **Does Your Trial Simulator Know What the Drug Does? An Intervention-Blind Test of Interventional Validity on 7,207 Registered Trials**
> Alicem Koyun. NeurIPS 2026 Workshop on World Models for High-Stakes Health (WMHS), oral.

The benchmark (v2.1) is 9,996 within-trial contrasts (treatment-arm outcome minus comparator-arm outcome on the primary
endpoint) from 7,207 completed phase 2/3 trials with results posted to ClinicalTrials.gov; the submitted version's
benchmark (v2: 11,274 contrasts from 8,011 trials) is kept as a sensitivity set. The measurement is **interventional skill**:
how much error a simulator removes relative to the *same* model trained with every intervention-identifying field withheld.

## Layout

```
src/ivbench/
  harvest.py       pull completed phase 2/3 interventional trials with posted results (ClinicalTrials.gov API v2)
  parse.py         study record -> arm-level rows (primary outcome, baseline covariates, participant flow, arm role)
  build.py         inclusion rules, within-trial contrasts (risk difference / SMD), temporal + area splits, data card
  models.py        featurizer (TF-IDF+SVD text, ordinal categoricals, numerics), HistGBT, skill + trial bootstrap,
                   split conformal, selective prediction
  specs.py         the four feature specifications (blind, blind_ctrl, full, interv_only) and model/eval wrappers
  llm_prompts.py   exact prompt templates (binary and continuous headers), redaction of intervention tokens,
                   response parser; used unchanged for every language-model run in the paper
scripts/
  run_pipeline.sh          harvest -> parse -> build (regenerates data/ from the live registry; network, ~5 min)
  run_headline.py          seed-0 headline: intervention-blind ablation, conformal coverage, selective curve (~1 min)
  run_positive_control.py  five-seed ablation, noise ceiling, temporal cuts and strata, arm-level task,
                           same-molecule retrieval (~4 min)
  run_shift.py             split conformal under temporal and leave-one-area-out shift, operating points (minutes)
  run_llm_arm.py           protocol-reader language-model arm (needs an API key; see below)
  make_paper_tables.py     cross-checks the numbers quoted in the paper against results/ -> results/paper_numbers.json
  make_figures.py          regenerates the SUBMITTED version's figures (v2) from results/ (no model fitting)
  cr/figures/              camera-ready figures: one script per figure plus make_figures.sh (run by run_camera_ready.sh)
configs/features.yaml      feature groups, for reference
data/                      v2 benchmark (submitted version): benchmark_contrasts.csv.gz (11,274 rows), benchmark_arms.csv.gz
                           (40,450 rows), datacard.json; data/v2.1/: the primary v2.1 benchmark
                           (benchmark_contrasts_v2.1_final.csv.gz, 9,996 rows; benchmark_arms_v2.1.csv.gz, 40,450 rows; datacard_v2.1.json)
results/                   every number in the paper is a cell in one of these files; results/README.md maps each
                           paper table and figure to its source file
figures/                   paper figures as shipped (see mapping below) and the graphical abstract
```

The LaTeX source of the paper is provided separately as supplementary material and is not part of this repository.

## Reproducing the camera-ready paper

The camera-ready paper is regenerated in two layers. The **deterministic layer** is what a reader should run: it uses only
files committed here and makes no network or model calls. The **API layer** documents how those committed inputs were
obtained; it can be re-run, but its outputs depend on external services that change over time.

### Deterministic layer (no network, no model calls)

```bash
conda env create -f environment.yml && conda activate ivbench
bash scripts/run_camera_ready.sh            # regenerate results/cr/*.csv, tables/*.tex and figures/*.png (uses cached registry fits)
bash scripts/run_camera_ready.sh --fresh    # refit every registry model first
bash scripts/run_camera_ready.sh --check    # refit, regenerate, and diff every table and figure against the committed copy (about 25 min)
```

Inputs: the v2.1 benchmark (`data/v2.1/`, produced from the frozen registry harvest; `scripts/cr/finalize_v21.py` applies
the author-approved post-audit edits), CT.gov-derived trial flags (`data/cr/`), the archived per-call language-model
outputs (`results/llm/`, `results/llm/cr/`) and the PubMedBERT embeddings (`results/cr/embeddings_v21_binary.npz`).
Each table is written by one script; full-precision numbers go to `results/cr/<label>.csv`. Bootstraps are over trials
with fixed seeds (registry models 1,000 resamples, language-model arms 2,000, seed 0; five-seed registry means use one shared
set of resamples). Primary results are on v2.1; v2 values are kept in the CSVs as sensitivity rows.

| paper | label | content | script | LaTeX | numbers |
|---|---|---|---|---|---|
| Table 1 | `tab:main` | main results: registry and LM rows | `scripts/cr/tables/tab_main.py` | `tables/main.tex` | `results/cr/main.csv` |
| Table 4 | `tab:lmruns` | all language-model runs | `scripts/cr/tables/tab_lmruns.py` | `tables/lmruns.tex` | `results/cr/lmruns.csv` |
| Table 5 | `tab:contamination` | contamination strata | `scripts/cr/tables/tab_contamination.py` | `tables/contamination.tex` | `results/cr/contamination.csv` |
| Table 6 | `tab:seeds` | per-seed GBT | `scripts/cr/tables/tab_seeds.py` | `tables/seeds.tex` | `results/cr/seeds.csv` |
| Table 7 | `tab:robust` | robustness | `scripts/cr/tables/tab_robust.py` | `tables/robust.tex` | `results/cr/robust.csv` |
| Table 8 | `tab:arch` | registry architectures | `scripts/cr/tables/tab_arch.py` | `tables/arch.tex` | `results/cr/arch.csv` |
| Table 9 | `tab:fg` | feature-group gains (provisional: feature-group definitions re-derived, see notes) | `scripts/cr/tables/tab_fg.py` | `tables/fg.tex` | `results/cr/fg.csv` |
| Table 10 | `tab:seen` | seen vs unseen interventions (token-overlap strata re-derived) | `scripts/cr/tables/tab_seen.py` | `tables/seen.tex` | `results/cr/seen.csv` |
| Table 11 | `tab:level` | arm-level task | `scripts/cr/tables/tab_level.py` | `tables/level.tex` | `results/cr/level.csv` |
| Table 12 | `tab:mol` | same-molecule positive control (provisional for union/MeSH variants, see notes) | `scripts/cr/tables/tab_mol.py` | `tables/mol.tex` | `results/cr/mol.csv` |
| Table 13 | `tab:coverage` | conformal coverage under shift | `scripts/cr/tables/tab_coverage.py` | `tables/coverage.tex` | `results/cr/coverage.csv` |
| Table 14 | `tab:op` | answer-or-abstain operating points | `scripts/cr/tables/tab_op.py` | `tables/op.tex` | `results/cr/op.csv` |
| Table 15 | `tab:lmarea` | LM by therapeutic area | `scripts/cr/tables/tab_lmarea.py` | `tables/lmarea.tex` | `results/cr/lmarea.csv` |
| Table 16 | `tab:lmsel` | LM selective prediction | `scripts/cr/tables/tab_lmsel.py` | `tables/lmsel.tex` | `results/cr/lmsel.csv` |
| new | `tab:exposure` | exposure test (Llama 3.1 70B) | `scripts/cr/tables/tab_exposure.py` | `tables/exposure.tex` | `results/cr/exposure.csv` |
| new | `tab:lmrevision` | protocol-only, swap, repeat, pre-registered negative control | `scripts/cr/tables/tab_lmrevision.py` | `tables/lmrevision.tex` | `results/cr/lmrevision.csv` |
| new | `tab:blindness` | how blind is the blind reference | `scripts/cr/tables/tab_blindness.py` | `tables/blindness.tex` | `results/cr/blindness.csv` |
| new | `tab:design` | randomised-parallel and valid-conversion subsets | `scripts/cr/tables/tab_design.py` | `tables/design.tex` | `results/cr/design.csv` |

Table numbers are those of the submitted paper; camera-ready numbering may shift when the new tables are placed. Tables 2 (cohort counts) and 3 (feature groups) come from `data/v2.1/datacard_v2.1.json` and `src/ivbench/specs.py`.

### API layer (network and paid model calls; outputs archived above)

| step | script | service | archived output |
|---|---|---|---|
| Trial harvest (completed phase 2/3 trials with posted results) | `src/ivbench/harvest.py`, `scripts/run_pipeline.sh` | ClinicalTrials.gov API v2 | `data/benchmark_*.csv.gz` (v2, frozen) |
| Arm types and design fields for the v2.1 repair | `scripts/fetch_arm_types.py`, `scripts/v21_audit/fetch_ctgov_meta.py` | ClinicalTrials.gov | `data/v2.1/ctgov_arms_design_oct2026.jsonl.gz`, `data/cr/*.csv` |
| Intervention MeSH terms (same-molecule variants) | `scripts/cr/fetch_intervention_mesh.py` | ClinicalTrials.gov | `data/cr/intervention_mesh_terms.csv` (fetched Oct 2026) |
| v2.1 rebuild from the fetched records | `scripts/build_v21.py` with `src/ivbench/pairing_rules.py` | none once records are fetched | `data/v2.1/benchmark_contrasts_v2.1.csv.gz` |
| Pairing audits (model-assisted review) | `scripts/v21_audit/*.py` | ClinicalTrials.gov, Anthropic API | `results/v2.1/audit_*` |
| Claude runs: Sonnet 5 (`claude-sonnet-5`), Haiku 4.5 (`claude-haiku-4-5-20251001`), provider default temperature | `scripts/run_llm_arm.py`, `scripts/cr/lm_chunk_runner.py`, `scripts/cr/v21_lm_runner.py`; prompts in `src/ivbench/llm_prompts.py` | Anthropic API | `results/llm/*`, `results/llm/cr/*` |
| Llama 3.1 70B Instruct, BF16 dedicated deployment, temperature 0, JSON mode | `scripts/cr/llama_run.py` (`scripts/cr/run_llm_open.py` for any OpenAI-compatible endpoint) | Fireworks AI | `results/llm/cr/llama70b_*` |
| PubMedBERT sentence embeddings (`pritamdeka/S-PubMedBert-MS-MARCO`) | `scripts/cr/embed_v21.py` | Hugging Face Hub (model download) | `results/cr/embeddings_v21_binary.npz` |

Re-running the API layer will not reproduce the archived outputs exactly: hosted models change, the Claude runs used the
provider's default sampling temperature, and registry records are revised. The negative-control hypotheses were
pre-registered before any call was made (`results/llm/cr/negative_control_preregistration.md`).

### Known limits of the reproduction

* `tab:fg`: the original v2 feature-group probe code was not preserved; group definitions were re-derived and reproduce 2 of
  14 released v2 rows. The v2.1 numbers in this table are provisional.
* `tab:mol`: the union and MeSH variants need harvest-time intervention MeSH terms, which were not preserved; current terms
  were re-fetched (Oct 2026). On v2 this gives n = 342 vs 358 released and IS -0.032 vs 0.006. The name-keyed variants
  reproduce the released v2 rows; the union and MeSH rows are provisional.
* `tab:seen`: exact-name strata reproduce v2 exactly; token-overlap strata differ by 2 contrasts (matching rule re-derived).
* `tab:level`: the v2.1 port uses the repaired arm roles and excludes pooled or total arms (285 binary arms).
* Perturbation runs (swap, negative control, repeat, drug-class guess) exist for the 1,020 v2 binary test contrasts only.

## Reproducing the submitted (v2) version


```bash
conda env create -f environment.yml && conda activate ivbench

# 1. headline: seed-0 interventional skill, conformal coverage, selective curve  (~1 min)
python scripts/run_headline.py --contrasts data/benchmark_contrasts.csv.gz \
    --arms data/benchmark_arms.csv.gz --out results/headline --seed 0
# -> results/headline/headline.json : interventional_skill = -0.0799, CI [-0.163, -0.012]  (paper Table 6, seed 0)

# 2. positive controls and robustness (Tables 1, 6, 7, 11, 12; noise ceiling in Sec. 3)  (~4 min)
python scripts/run_positive_control.py --out results/positive_control

# 3. conformal coverage under shift (Fig. 3a-b, Tables 13-14)
(cd results/shift && python ../../scripts/run_shift.py)

# 4. cross-check every quoted number, then redraw the figures
python scripts/make_paper_tables.py
python scripts/make_figures.py
```

Step 2 reproduces `results/positive_control/v2_ablation_seeds.csv` (all five seeds), `v2_noise_ceiling.csv`,
`v2_robustness.csv` and `v2_level_task.csv` bit-for-bit, point estimates and bootstrap intervals alike (verified with
scikit-learn 1.9.0 on macOS/arm64; a different BLAS or scikit-learn version may move the last decimal). Intervals use
1,000 trial-level resamples (2,000 for the language-model arm) with a fixed seed; changing `--n-boot` moves only the
intervals. Seeds 1-4 differ from seed 0 by up to 0.07 in interventional skill (Table 6); the paper reports the
five-seed mean.

`run_positive_control.py` reproduces the `name` variant of the same-molecule retrieval control (Table 12). The
`union` variant reported as the headline value in Sec. 5.3 additionally keys on trial-level MeSH intervention terms
that live in the raw harvest (`data/trials_raw.jsonl.gz`, regenerated by `run_pipeline.sh`) and are not carried into
the released contrast table; all 24 variants are in `results/positive_control/v2_same_molecule.csv`.

## Regenerate the benchmark from the registry

```bash
bash scripts/run_pipeline.sh     # writes data/trials_raw.jsonl.gz, then data/benchmark_*.csv.gz and datacard.json
```

The registry is live: a fresh harvest contains trials whose results were posted after our snapshot
(2026-09-09, recorded in `data/datacard.json`) and occasionally corrected records. To reproduce the paper exactly,
use the released `data/` files.

## Language-model arm

The paper's runs used Anthropic models through the Messages API: `claude-sonnet-5` ("large") and
`claude-haiku-4-5-20251001` ("small"). `scripts/run_llm_arm.py` reproduces prompt construction byte-for-byte from
`src/ivbench/llm_prompts.py` (the templates are also printed in Appendix D of the paper) and the parse/retry rule
(one retry with `RETRY_SUFFIX` on parse failure, `max_tokens = 1500`).

```bash
export ANTHROPIC_API_KEY=...
python scripts/run_llm_arm.py --api anthropic --model claude-sonnet-5 --family binary --condition A \
    --out results/llm/rerun_sonnet_binary_A.csv
python scripts/run_llm_arm.py --api anthropic --model claude-sonnet-5 --family binary --condition B \
    --out results/llm/rerun_sonnet_binary_B.csv
```

`--api openai` targets any OpenAI-compatible `/v1/chat/completions` endpoint (`LLM_ENDPOINT`, `LLM_API_KEY`) for
replication with other model families. Condition A shows the intervention, condition B withholds it; interventional
skill is the skill of A with B as the reference, paired by contrast id. Every per-call output used in the paper
(both models, both conditions, both endpoint families, and the repeat run) is in `results/llm/`; the redaction audit of
the blind prompts is `results/llm/llm_redaction_audit_v2.csv`. Language-model outputs are not bit-reproducible across
provider updates; the released files are the record.

## Figure files and paper numbering

| file | paper | script | inputs |
|---|---|---|---|
| `figures/fig2_benchmark_skill.png` | Figure 1 | `scripts/cr/figures/fig_fig2_benchmark_skill.py` | `data/datacard.json`, `data/v2.1/datacard_v2.1.json`, `results/cr/main.csv`, `results/cr/design_noise_ceiling.csv` |
| `figures/fig3_interventional_forest.png` | Figure 2 | `scripts/cr/figures/fig_fig3_interventional_forest.py` | `results/cr/main.csv`, `arch.csv`, `mol.csv`, `exposure.csv` |
| `figures/fig4_reliability.png` | Figure 3 | `scripts/cr/figures/fig_fig4_reliability.py` | `results/cr/coverage.csv`, `op.csv`, `lmsel.csv`, `lmruns.csv`; panel (b) refits the cross-fitted temporal regime with `scripts/cr/tables/_conformal.py` and writes `results/cr/fig4b_selective_crossfit.csv` |
| `figures/figA_probes.png` | Figure 4 | `scripts/cr/figures/fig_figA_probes.py` | `results/cr/fg.csv`, `seen.csv`, `lmarea.csv` |
| `figures/wmhs_gap_map.png` | Figure 5 | drawn from `results/screen/neurips2025_wmhs_labelled.csv` (submitted version, unchanged) |
| `figures/graphical_abstract.png` | graphical abstract (not in the paper) | hand-drawn | |

All four camera-ready figures read every number from `results/cr/*.csv`; none types a value. Each is written as a 300 dpi PNG and as a vector PDF (the paper includes the PDF); the three main-text figures are laid out at the 5.5 in NeurIPS text width so their 7-8 pt labels print at size. File names predate the final numbering and are kept so the paper source compiles unchanged.

## Definitions

* **skill** = 1 - MSE(model) / MSE(reference). The reference is the global training mean unless stated.
* **interventional skill (IS)** = skill of the full model with the intervention-blind model of the same architecture
  as reference (`full` vs `blind_ctrl` in `specs.py`).
* Confidence intervals resample whole trials (`nct_id`), never rows.
* Binary endpoints: proportions, contrast = risk difference. Continuous: standardised mean difference with pooled SD.
* Temporal split: results posted before 2023 = train, 2023 or later = test.

## Licence and citation

Code: MIT. Data (`data/`) and result tables (`results/`): CC BY 4.0; derived from ClinicalTrials.gov, whose records are
in the public domain. See `LICENSE`. Citation metadata is in `CITATION.cff`. Source: https://github.com/alicemkyn/ivbench. The archived release is on Zenodo (DOI in `CITATION.cff` once the release is tagged).
