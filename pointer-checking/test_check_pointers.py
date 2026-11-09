import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

import requests

from check_pointers import PointerChecker, extract_pointers, load_cache, load_models, run, select_sample
from verify_pages import PageEvidence


def response(code=200, **data):
    result = Mock(status_code=code, url="https://huggingface.co/api/models/org/base", headers={})
    result.json.return_value = data
    return result


class ExtractionTests(unittest.TestCase):
    def test_sample_csv_uses_exact_ids_in_sheet_order(self):
        models = [{"id": "org/a"}, {"id": "org/b"}, {"id": "org/c"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("Model ID,notes\norg/c,\norg/a,\n", encoding="utf-8-sig")
            self.assertEqual(select_sample(models, path), [models[2], models[0]])
            for content, error in [
                ("Model ID\norg/c\norg/c\n", "Duplicate"),
                ("Model ID\norg/missing\n", "absent"),
                ("id\norg/c\n", "requires a Model ID"),
                ("Model ID,notes\n,review\n", "nonempty"),
            ]:
                path.write_text(content)
                with self.assertRaisesRegex(ValueError, error):
                    select_sample(models, path)

    def test_page_title_ignores_later_svg_titles(self):
        page = PageEvidence()
        page.feed('<head><title>org/model</title><link rel="canonical" href="https://huggingface.co/org/model"></head>'
                  '<svg><title>Icon</title></svg>')
        self.assertEqual(page.title, "org/model")
        self.assertEqual(page.canonical, "https://huggingface.co/org/model")

    def test_relations_are_deduplicated_and_evidence_retained(self):
        pointers = extract_pointers({"id": "org/model", "tags": [
            "base_model:org/base", "base_model:quantized:org/base",
            "base_model:merge:org/second", "dataset:snli", "dataset:snli", "text-generation"]})
        self.assertEqual(len(pointers), 3)
        self.assertEqual(len(pointers[0]["source_tags"]), 2)
        self.assertEqual(pointers[1]["dependency_id"], "org/second")
        self.assertEqual(pointers[2]["dependency_id"], "snli")

    def test_read_shared_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("metadata/models.json", '[{"id":"org/model","tags":[]}]')
                archive.writestr("commits/commits.json", "[]")
            self.assertEqual(load_models(path)[0]["id"], "org/model")

    def test_latest_failed_refresh_does_not_reuse_an_old_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checks.jsonl"
            base = {"dependency_type": "dataset", "dependency_id": "snli"}
            path.write_text("\n".join(json.dumps({**base, "status": status}) for status in
                                      ["metadata_accessible", "server_error"]) + "\n")
            self.assertEqual(load_cache(path), {})


@patch("check_pointers.time.sleep")
class CheckerTests(unittest.TestCase):
    def check(self, *responses):
        session = Mock()
        session.get.side_effect = responses
        return PointerChecker(session, interval=0), session

    def test_gated_metadata_is_not_unrestricted_access(self, sleep):
        checker, _ = self.check(response(id="org/base", gated="manual"))
        self.assertEqual(checker.check("base_model", "org/base")["status"], "gated")

    def test_rename_evidence_is_retained(self, sleep):
        checker, _ = self.check(response(id="org/renamed", gated=False))
        result = checker.check("base_model", "org/base")
        self.assertTrue(result["id_changed"])
        self.assertEqual(result["resolved_id"], "org/renamed")

    def test_case_only_change_is_not_rename(self, sleep):
        checker, _ = self.check(response(id="ORG/base", gated=False))
        self.assertFalse(checker.check("base_model", "org/base")["id_changed"])

    def test_ambiguous_errors_do_not_mean_deleted(self, sleep):
        for code, expected in [(401, "access_restricted"), (403, "access_restricted"),
                               (404, "not_found_or_private")]:
            checker, _ = self.check(response(code))
            self.assertEqual(checker.check("dataset", "snli")["status"], expected)

    def test_retry_timeout_and_server_error_then_success(self, sleep):
        checker, session = self.check(requests.Timeout(), response(503), response(id="org/base"))
        self.assertEqual(checker.check("base_model", "org/base")["status"], "metadata_accessible")
        self.assertEqual(session.get.call_count, 3)

    def test_long_retry_after_defers_until_next_run(self, sleep):
        limited = response(429)
        limited.headers = {"Retry-After": "300"}
        checker, session = self.check(limited)
        self.assertEqual(checker.check("dataset", "snli")["status"], "rate_limited")
        self.assertEqual(session.get.call_count, 1)

    def test_invalid_reference_does_not_trigger_request(self, sleep):
        checker, session = self.check()
        result = checker.check("dataset", "https://elsewhere.example/data")
        self.assertEqual(result["status"], "invalid_reference")
        session.get.assert_not_called()

    def test_successful_html_is_not_accessible(self, sleep):
        bad = response()
        bad.json.side_effect = ValueError()
        checker, _ = self.check(bad, bad, bad)
        self.assertEqual(checker.check("dataset", "snli")["status"], "unexpected_response")

    def test_resume_shares_checks_and_retries_transient_failures(self, sleep):
        models = [{"id": "org/a", "tags": ["base_model:org/base", "dataset:snli"]},
                  {"id": "org/b", "tags": ["base_model:org/base"]}]
        with tempfile.TemporaryDirectory() as directory:
            checker, session = self.check(response(id="org/base"), response(503))
            checker.attempts = 1
            first = run(models, directory, checker)
            self.assertEqual(session.get.call_count, 2)
            self.assertEqual(first["pointer_count"], 3)
            checker, session = self.check(response(id="stanfordnlp/snli"))
            second = run(models, directory, checker)
            self.assertEqual(session.get.call_count, 1)
            self.assertEqual(second["statuses_by_pointer"], {"metadata_accessible": 3})
            self.assertEqual(len(second["missing_reference_types"]), 1)
            self.assertTrue(json.loads((Path(directory) / "summary.json").read_text()))


if __name__ == "__main__":
    unittest.main()
