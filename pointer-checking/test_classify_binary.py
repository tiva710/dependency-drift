import unittest

from packaging.version import Version

from assess_drift import library_result
from classify_binary import classify_library, classify_models, classify_pointers


def library(version="1.0", kind="pinned"):
    declaration = {"model_id": "org/model", "package": "example", "version_spec": version,
                   "kind": kind, "file": "config.json", "source_key": "example_version"}
    release = {"status": "ok", "releases": {v: [{"yanked": False}] for v in ("1.0", "2.8.1", "4.0")}}
    return library_result(declaration, release)


def pointer(status="metadata_accessible", access=""):
    return {"model_id": "org/model", "dependency_type": "base_model", "dependency_id": "org/base",
            "status": status, "access_status": access, "id_changed": "False"}


class BinaryTests(unittest.TestCase):
    def test_development_version_distance_is_classified_and_qualified(self):
        row = classify_library(library("1.2.dev0"), [Version("4.0")])
        self.assertEqual(row["binary_stale"], "Y")
        self.assertEqual(row["review_required"], "Y")
        self.assertIn("publication/build provenance needs review", row["classification_reason"])
        current = classify_library(library("4.0.dev0"), [Version("4.0")])
        self.assertEqual(current["binary_stale"], "N")

    def test_ranges_use_newest_allowed_release(self):
        versions = [Version("1.0"), Version("4.0")]
        old = classify_library(library("<2", "ranged"), versions)
        current = classify_library(library(">=1", "ranged"), versions)
        self.assertEqual(old["binary_stale"], "Y")
        self.assertEqual(current["binary_stale"], "N")

    def test_wildcard_and_revision_suffix_are_supported(self):
        wildcard = classify_library(library("==2.8.*"), [Version("2.8.1"), Version("5.0")])
        self.assertEqual(wildcard["binary_stale"], "Y")
        revision = classify_library(library("1.0.dev0@abcdef0123456789"), [Version("4.0")])
        self.assertEqual(revision["binary_stale"], "Y")

    def test_unpinned_has_no_old_version_restriction(self):
        row = classify_library(library("", "unpinned"), [Version("4.0")])
        self.assertEqual(row["binary_stale"], "N")
        self.assertIn("no old-version restriction", row["classification_reason"])

    def test_failed_lookup_and_denied_gating_flag_availability_candidates(self):
        exists, flagged, review, notes = classify_pointers([pointer("not_found_or_private")], "base_model")
        self.assertEqual(exists, "N")
        self.assertTrue(flagged)
        self.assertTrue(review)
        exists, flagged, _, _ = classify_pointers([pointer("gated", "denied")], "base_model")
        self.assertEqual(exists, "Y")
        self.assertTrue(flagged)
        _, allowed, _, _ = classify_pointers([pointer("gated", "accessible")], "base_model")
        self.assertFalse(allowed)

    def test_transient_failures_require_retry(self):
        with self.assertRaisesRegex(ValueError, "Retry incomplete"):
            classify_pointers([pointer("rate_limited")], "base_model")

    def test_unverifiable_library_is_binary_with_explicit_review_flag(self):
        row = classify_library(library("nonsense"), [Version("4.0")])
        previous = [{"Model ID": "org/model", "drift (Y/N)": ""}]
        pointers = [pointer(), {**pointer(), "dependency_type": "dataset"}]
        result = classify_models(previous, pointers, [row])[0]
        self.assertEqual(result["drift (Y/N)"], "N")
        self.assertEqual(result["review_required (Y/N)"], "Y")
        self.assertIn("N means no qualifying signal", result["notes"])

    def test_existing_positive_review_is_preserved(self):
        row = classify_library(library("1.0"), [Version("1.0")])
        previous = [{"Model ID": "org/model", "drift (Y/N)": "Y"}]
        result = classify_models(previous, [pointer(), {**pointer(), "dependency_type": "dataset"}], [row])[0]
        self.assertEqual(result["drift (Y/N)"], "Y")


if __name__ == "__main__":
    unittest.main()
