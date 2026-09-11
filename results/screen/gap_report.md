# NeurIPS 2025 workshop landscape → gaps for WMHS @ NeurIPS 2026

## What was screened

| step | count |
|---|---|
| NeurIPS 2025 workshops on OpenReview | 62 |
| workshops listed on the NeurIPS virtual site | 56 |
| workshops indexed here (health, causality, world-model, reliability, evaluation) | 19 |
| accepted papers indexed (title, authors, OpenReview id) | 1648 |
| abstracts retrieved | 1618 |
| clinically grounded papers labelled on 7 axes | 447 unique (467 rows incl. cross-listings) |

Labels come from the abstracts, not the PDFs: 455 by model extraction, 12 by hand. Abstract-level
labels under-report validation detail — which is why the 28-paper shortlist gets read in full.

## The funnel — the WMHS cell is empty

Of 447 clinically grounded papers:

| requirement (cumulative) | papers |
|---|---|
| models a patient trajectory or a trial arm | 26 |
| + intervention represented beyond a binary treatment flag | 8 |
| + explicit causal machinery | 5 |
| + uncertainty quantified or calibrated | 2 |
| + validated outside the training cohort | 1 |
| + validated against a real trial arm | 0 |

Single-axis rarities across all 447 papers: target-trial emulation appears **once**;
conformal prediction **once**; selective prediction / abstention **once**; calibration in 19;
validation against RCT ground truth in 4;
historical-trial replication in 1;
internal train/test split only in 233.
Intervention representation is absent in 369 papers;
protocol-level in 8; time-varying regimes in 5;
mechanism of action in 2.

## Four exploitable gaps

**G1 — Nobody validates a simulator against the thing it claims to simulate.**
26 papers simulate patient trajectories or trial arms; 1 compares its output to a real
trial arm. Validation is overwhelmingly an internal split (233 papers), i.e. the model is
scored on predicting held-out patients, not on reproducing a known causal contrast.

**G2 — Abstention and calibrated coverage are missing exactly where stakes are highest.**
Uncertainty appears as a scalar (78 papers) but almost never as a *decision*: one paper uses
conformal methods, one uses selective prediction. No paper in the set asks when a patient
world model should refuse to simulate.

**G3 — Protocol reasoning is disconnected from causal identification.**
13 papers read protocols, guidelines or eligibility criteria; 0 of them carry any causal
machinery, and 0 validate outside their own annotations. Protocol text is treated as an NLP
task, not as the specification of an estimand.

**G4 — Agentic clinical-research systems are unfalsified.**
22 agentic systems; 2 carry external or downstream validation. Demo-grade evidence dominates.

## Three candidate directions (provisional — to be fixed after full-text reading)

**D1. A trial-emulation stress test for patient world models.** Score simulators not on held-out
patient prediction but on reproducing published RCT contrasts under target-trial emulation, with
calibration and coverage as first-class metrics. Hits G1 + G2. Strongest claim; data access is the
risk (patient-level longitudinal data with a matching published trial).

**D2. Selective simulation: a world model that knows which patients it cannot simulate.**
Conformal risk control over trajectory-level error, reported as an accuracy-coverage curve under
protocol-defined distribution shift. Hits G2 + G1. Cleanest experiment for a six-day window;
runs on open longitudinal data or a mechanistic PK/PD simulator with known ground truth.

**D3. From protocol text to an identifiability-checked emulation.** An agent that compiles trial
eligibility, treatment strategy and endpoint text into a target-trial specification, then checks
which parts are identifiable from a given data schema and abstains on the rest. Hits G3 + G4.
Fully public data (ClinicalTrials.gov); most WMHS-native framing; needs a gold standard.

## Caveats

Scope filter is keyword + workshop based; papers whose abstracts avoid clinical vocabulary may be
missed. 30 of 1648 indexed papers had no retrievable abstract. Counts are per unique title
(17 papers appear in more than one workshop: 15 in two, 2 in three). OpenReview PDFs and the notes API are behind a
bot-verification gate, so no full texts were read for this pass.
