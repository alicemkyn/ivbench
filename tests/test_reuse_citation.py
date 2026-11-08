"""Lightweight checks for citation links and the released benchmark counts.

The SYNTHETIC fixture below is generated in a temporary file. It is not benchmark
data and is not a scientific result.
"""
import csv
import gzip
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
CITATION = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
TAB_MAIN = (ROOT / "scripts" / "cr" / "tables" / "tab_main.py").read_text(encoding="utf-8")

PAPER_URL = "https://arxiv.org/abs/2611.05907"
ARTIFACT_DOI = "10.5281/zenodo.23521618"
ARTIFACT_URL = "https://doi.org/10.5281/zenodo.23521618"
ZENODO_BADGE = (
    "[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23521618.svg)]"
    "(https://doi.org/10.5281/zenodo.23521618)"
)
TABLE_COMMAND = "python scripts/cr/tables/tab_main.py"
# GitHub About text confirmed on the repository settings page. It is not a file.
STALE_ABOUT = "11,274 within-trial contrasts from 8,011 ClinicalTrials.gov trials"
WORKSHOP = "NeurIPS 2026 Workshop on World Models for High-Stakes Health"


def contrast_counts(path):
    """Return (n_rows, n_trials) for a contrast table with an nct_id column."""
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        n_rows = 0
        trials = set()
        for row in rows:
            n_rows += 1
            trials.add(row["nct_id"])
    return n_rows, len(trials)


class ReuseCitationTests(unittest.TestCase):
    def test_citation_file_keeps_v21_software_record(self):
        self.assertIn('cff-version: 1.2.0', CITATION)
        self.assertIn('version: "2.1.0"', CITATION)
        self.assertIn("license: MIT", CITATION)
        self.assertIn(f"doi: {ARTIFACT_DOI}", CITATION)
        self.assertIn("family-names: Koyun", CITATION)
        self.assertIn("given-names: Alicem", CITATION)
        self.assertIn('date-released: "2026-10-20"', CITATION)

    def test_preferred_citation_is_the_workshop_preprint(self):
        preferred = CITATION.split("preferred-citation:", 1)[1]
        self.assertIn("type: conference-paper", preferred)
        self.assertIn(f'url: "{PAPER_URL}"', preferred)
        self.assertIn('value: "arXiv:2611.05907"', preferred)
        self.assertIn(f'collection-title: "{WORKSHOP} (WMHS)"', preferred)
        self.assertIn(f'name: "{WORKSHOP}"', preferred)
        self.assertIn(
            "Oral presentation at the NeurIPS 2026 Workshop on World Models for High-Stakes Health.",
            preferred,
        )
        self.assertNotIn("10.5281/zenodo", preferred.split("identifiers:", 1)[0])

    def test_readme_has_paper_artifact_and_badge(self):
        self.assertIn(ZENODO_BADGE, README)
        self.assertEqual(README.count(ZENODO_BADGE), 1)
        self.assertIn(PAPER_URL, README)
        self.assertIn(ARTIFACT_URL, README)
        self.assertIn("Oral presentation at the NeurIPS 2026 Workshop on World Models for High-Stakes Health (WMHS).", README)
        self.assertNotIn(STALE_ABOUT, README)
        self.assertNotIn(STALE_ABOUT, CITATION)

    def test_readme_keeps_v21_as_primary_and_v2_as_sensitivity(self):
        primary = README.split("The benchmark (v2.1) is ", 1)[1].split(".", 1)[0]
        self.assertIn("9,996 within-trial contrasts", primary)
        self.assertIn("7,207", primary)
        self.assertNotIn("11,274", primary)
        self.assertNotIn("8,011", primary)
        self.assertIn("v2: 11,274 contrasts from 8,011 trials", README)
        self.assertIn("sensitivity set", README)

    def test_readme_documents_the_existing_table_command(self):
        self.assertIn(TABLE_COMMAND, README)
        self.assertIn(TABLE_COMMAND, TAB_MAIN)
        self.assertTrue((ROOT / "scripts" / "cr" / "tables" / "tab_main.py").is_file())

    def test_counter_on_synthetic_fixture(self):
        """SYNTHETIC fixture: three rows and two trial ids, not a released table."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic_contrasts.csv.gz"
            with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["nct_id", "family"])
                writer.writeheader()
                writer.writerow({"nct_id": "SYNTHETIC_NCT_1", "family": "binary"})
                writer.writerow({"nct_id": "SYNTHETIC_NCT_1", "family": "continuous"})
                writer.writerow({"nct_id": "SYNTHETIC_NCT_2", "family": "binary"})
            self.assertEqual(contrast_counts(path), (3, 2))

    def test_released_tables_match_documented_counts(self):
        final = ROOT / "data" / "v2.1" / "benchmark_contrasts_v2.1_final.csv.gz"
        pre_finalize = ROOT / "data" / "v2.1" / "benchmark_contrasts_v2.1.csv.gz"
        v2 = ROOT / "data" / "benchmark_contrasts.csv.gz"
        self.assertEqual(contrast_counts(final), (9996, 7207))
        # Pre-finalize rebuild and the submitted v2 table. These counts are historical.
        self.assertEqual(contrast_counts(pre_finalize), (10004, 7210))
        self.assertEqual(contrast_counts(v2), (11274, 8011))


if __name__ == "__main__":
    unittest.main()
