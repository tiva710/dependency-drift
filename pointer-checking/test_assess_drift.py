import csv
from pathlib import Path
import tempfile
import unittest

from assess_drift import assess, combined, library_result, summarize, load_evidence, compact_notes


def declaration(version="1.0", kind="pinned"):
    return {"model_id": "org/model", "package": "example", "version_spec": version,
            "kind": kind, "file": "requirements.txt", "source_key": "1"}


def release(*versions, yanked=()):
    return {"status": "ok", "releases": {
        v: [{"yanked": v in yanked}] for v in versions}}


def pointer(kind="base_model", status="metadata_accessible", renamed=False):
    return {"model_id": "org/model", "dependency_type": kind, "dependency_id": "org/base",
            "status": status, "id_changed": str(renamed), "resolved_id": "org/renamed" if renamed else "org/base"}


class LabelTests(unittest.TestCase):
    def test_positive_dominates_unknown_but_negative_requires_complete_checks(self):
        self.assertEqual(combined(["Y", ""]), "Y")
        self.assertEqual(combined(["N", ""]), "")
        self.assertEqual(combined([]), "")
        self.assertEqual(combined(["N", "N"]), "N")

    def test_stale_threshold_and_breaking_evidence_are_independent(self):
        two = library_result(declaration(), release("1.0", "3.0"))
        three = library_result(declaration(), release("1.0", "4.0"))
        self.assertEqual(two["stale"], "N")
        self.assertEqual(three["stale"], "Y")
        self.assertEqual(three["rq1_drift"], "")

    def test_latest_ignores_yanked_and_prerelease_versions(self):
        result = library_result(declaration(), release("1.0", "2.0", "9.0rc1", "8.0", yanked=("8.0",)))
        self.assertEqual(result["latest_version"], "2.0")
        self.assertEqual(result["stale"], "N")

    def test_unknown_and_nonexact_declarations_remain_unresolved(self):
        cases = [("1.0.dev0", "pinned"), ("1.0+cu121", "pinned"),
                 (">=1.0", "ranged"), ("", "unpinned"), ("1.*", "pinned"), ("3.0", "pinned")]
        for value, kind in cases:
            with self.subTest(value=value):
                result = library_result(declaration(value, kind), release("1.0", "4.0"))
                self.assertEqual(result["stale"], "")
                self.assertEqual(result["rq1_drift"], "")

    def test_yanked_pin_is_flagged_without_claiming_deletion(self):
        result = library_result(declaration(), release("1.0", yanked=("1.0",)))
        self.assertTrue(result["release_yanked"])
        self.assertEqual(result["rq1_drift"], "")

    def test_rename_counts_once_at_model_level_even_with_unknown_library(self):
        libs = [library_result(declaration("1.0.dev0"), release("1.0"))]
        rows = assess([{"Model ID": "org/model"}], [pointer(renamed=True), pointer("dataset", "access_restricted")], libs, {})
        self.assertEqual(rows[0]["base_model_exists (Y/N)"], "Y")
        self.assertEqual(rows[0]["dataset_exists (Y/N)"], "")
        self.assertEqual(rows[0]["drift (Y/N)"], "Y")
        self.assertEqual(summarize(rows)["drift_yes"], 1)

    def test_404_and_gating_cannot_establish_deletion(self):
        libs = [library_result(declaration(), release("1.0"))]
        for status, exists, drift in [("not_found_or_private", "", ""), ("gated", "Y", "")]:
            with self.subTest(status=status):
                rows = assess([{"Model ID": "org/model"}], [pointer(status=status), pointer("dataset")], libs, {})
                self.assertEqual(rows[0]["base_model_exists (Y/N)"], exists)
                self.assertEqual(rows[0]["drift (Y/N)"], drift)

    def test_conflicting_source_pins_are_both_evaluated(self):
        old = library_result(declaration("1.0"), release("1.0", "4.0"))
        current = library_result({**declaration("4.0"), "file": "config.json"}, release("1.0", "4.0"))
        rows = assess([{"Model ID": "org/model"}], [pointer(), pointer("dataset")], [old, current], {})
        self.assertEqual(rows[0]["library_version_stale (Y/N)"], "Y")
        self.assertEqual(rows[0]["drift (Y/N)"], "Y")
        self.assertEqual(rows[0]["rq1_confirmed_drift (Y/N)"], "")

    def test_reviewed_breaking_change_is_scoped_to_exact_source(self):
        library = library_result(declaration(), release("1.0", "4.0"))
        key = ("org/model", "library", "example 1.0 @ requirements.txt #1")
        rows = assess([{"Model ID": "org/model"}], [pointer(), pointer("dataset")], [library],
                      {key: {"verdict": "breaking_change", "source_url": "https://example.org/changelog", "reason": "Removed API is used by the model"}})
        self.assertEqual(rows[0]["drift (Y/N)"], "Y")
        self.assertEqual(rows[0]["rq1_confirmed_drift (Y/N)"], "Y")

    def test_reviewed_no_breaking_change_does_not_override_another_source(self):
        old = library_result(declaration("1.0"), release("1.0", "4.0"))
        current = library_result({**declaration("4.0"), "file": "config.json"}, release("1.0", "4.0"))
        key = ("org/model", "library", "example 4.0 @ config.json #1")
        rows = assess([{"Model ID": "org/model"}], [pointer(), pointer("dataset")], [old, current],
                      {key: {"verdict": "no_breaking_change", "source_url": "https://example.org/changelog", "reason": "Current release"}})
        self.assertEqual(rows[0]["rq1_confirmed_drift (Y/N)"], "")


    def test_review_evidence_rejects_unmatched_source_and_missing_url_host(self):
        fields = ["model_id", "dependency_type", "dependency_id", "verdict", "source_url", "reason"]
        row = {"model_id": "org/model", "dependency_type": "library", "dependency_id": "example 1.0 @ requirements.txt #1",
               "verdict": "breaking_change", "source_url": "https://example.org/changelog", "reason": "Removed API is used"}
        key = (row["model_id"], row["dependency_type"], row["dependency_id"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"evidence.csv"
            def save(data):
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fields)
                    writer.writeheader()
                    writer.writerow(data)
            save(row)
            self.assertEqual(load_evidence(path, {"org/model"}, {key})[key]["verdict"], "breaking_change")
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_evidence(path, {"org/model"}, set())
            save({**row, "source_url": "https:/changelog"})
            with self.assertRaisesRegex(ValueError, "source URL"):
                load_evidence(path, {"org/model"}, {key})


    def test_large_model_notes_fit_normal_csv_readers(self):
        notes = compact_notes(["Status counts", "x" * 150000, "Another dependency"])
        self.assertLess(len(notes), 12000)
        self.assertIn("Status counts", notes)
        self.assertIn("2 further entries", notes)
        self.assertIn("library_checks.csv", notes)


    def test_verified_gated_access_can_resolve_pointer_drift(self):
        base = {**pointer(status="gated"), "access_status": "accessible"}
        library = library_result(declaration(), release("1.0"))
        rows = assess([{"Model ID": "org/model"}], [base, pointer("dataset")], [library], {})
        self.assertEqual(rows[0]["drift (Y/N)"], "N")
        self.assertEqual(rows[0]["rq1_confirmed_drift (Y/N)"], "N")


if __name__ == "__main__":
    unittest.main()
