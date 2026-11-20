import json
from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

import requests

from check_pointers import PointerChecker, extract_pointers, load_cache, load_models, run, select_sample, owner_matches, main
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

    def test_long_retry_after_waits_before_retrying_same_reference(self, sleep):
        limited = response(429)
        limited.headers = {"Retry-After": "300"}
        checker, session = self.check(limited, response(id="stanfordnlp/snli"))
        self.assertEqual(checker.check("dataset", "snli")["status"], "metadata_accessible")
        sleep.assert_any_call(301)
        self.assertEqual(session.get.call_count, 2)
        self.assertEqual(session.get.call_args_list[0], session.get.call_args_list[1])

    def test_hub_reset_header_controls_wait(self, sleep):
        limited = response(429)
        limited.headers = {"RateLimit": '"api";r=0;t=125'}
        checker, _ = self.check(limited, response(id="stanfordnlp/snli"))
        self.assertEqual(checker.check("dataset", "snli")["status"], "metadata_accessible")
        sleep.assert_any_call(126)

    def test_persistent_rate_limit_saves_progress_and_stops(self, sleep):
        models = [{"id": "org/a", "tags": ["base_model:org/base", "dataset:snli", "dataset:other"]}]
        with tempfile.TemporaryDirectory() as directory:
            checker, session = self.check(response(id="org/base"), *[response(429)] * 3)
            with self.assertRaisesRegex(SystemExit, "Progress saved"):
                run(models, directory, checker)
            self.assertEqual(session.get.call_count, 4)
            sleep.assert_any_call(301)
            journal = Path(directory) / "checks.jsonl"
            self.assertEqual(set(load_cache(journal)), {("base_model", "org/base")})
            self.assertEqual(json.loads(journal.read_text().splitlines()[-1])["status"], "rate_limited")

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


class AuthenticationTests(unittest.TestCase):
    def test_authentication_header_is_not_part_of_saved_observations(self):
        session = Mock(headers={})
        session.get.return_value = response(id="org/base")
        checker = PointerChecker(session=session, token="test-secret", interval=0)
        with patch("check_pointers.time.sleep"):
            result = checker.check("base_model", "org/base")
        self.assertEqual(session.headers["Authorization"], "Bearer test-secret")
        self.assertNotIn("test-secret", json.dumps(result))

    def test_authenticated_dry_run_binds_account_on_live_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "models.json"
            source.write_text('[{"id":"org/model","tags":[]}]')
            args = ["check_pointers.py", "--input", str(source), "--output-dir", str(root/"out"), "--authenticated"]
            checker = Mock()
            checker.session.get.return_value = response(name="reviewer")
            with patch("check_pointers.hub_token", return_value="test-secret"), patch("check_pointers.PointerChecker", return_value=checker), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with patch("sys.argv", args + ["--dry-run"]):
                    main()
                checker.session.get.assert_not_called()
                with patch("sys.argv", args):
                    main()
                manifest = json.loads((root/"out"/"manifest.json").read_text())
                self.assertEqual(manifest["account"], "reviewer")
                self.assertNotIn("test-secret", json.dumps(manifest))
                checker.session.get.return_value = response(name="another-reviewer")
                with patch("sys.argv", args), self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)

    @patch("check_pointers.time.sleep")
    def test_single_worker_authenticates_gated_access_and_resume_reuses_it(self, sleep):
        models = [{"id": "org/model", "tags": ["base_model:org/base"]}]
        session = Mock(headers={})
        session.get.side_effect = [response(id="org/base", gated="manual"), response()]
        checker = PointerChecker(session=session, token="test-secret", interval=0)
        with tempfile.TemporaryDirectory() as directory:
            run(models, directory, checker, token="test-secret")
            first = load_cache(Path(directory)/"checks.jsonl")[("base_model", "org/base")]
            self.assertEqual(first["status"], "gated")
            self.assertEqual(first["access_status"], "accessible")
            self.assertTrue(first["access_request_url"].endswith("/auth-check"))
            run(models, directory, checker, token="test-secret")
            self.assertEqual(session.get.call_count, 2)

    @patch("check_pointers.check_batch")
    def test_parallel_resume_preserves_deduplication_and_terminal_evidence(self, batch):
        models = [{"id": "org/a", "tags": ["base_model:org/base", "dataset:snli"]},
                  {"id": "org/b", "tags": ["base_model:org/base"]}]
        base = {"dependency_type": "base_model", "dependency_id": "org/base", "status": "metadata_accessible"}
        dataset = {"dependency_type": "dataset", "dependency_id": "snli", "status": "metadata_accessible"}
        batch.return_value = [dataset]
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/"checks.jsonl").write_text(json.dumps(base)+"\n")
            checker = Mock()
            result = run(models, directory, checker, workers=2, token="test-secret")
            batch.assert_called_once_with([("dataset", "snli")], "test-secret")
            checker.check.assert_not_called()
            self.assertEqual(result["pointer_count"], 3)
            self.assertEqual(result["statuses_by_pointer"], {"metadata_accessible": 3})


class OwnerListingTests(unittest.TestCase):
    @patch("check_pointers.time.sleep")
    def test_only_exact_ids_with_access_metadata_are_used(self, sleep):
        session = Mock()
        listed = response()
        listed.json.return_value = [
            {"id": "org/base", "gated": False, "private": False},
            {"id": "org/gated", "gated": "manual", "private": False},
            {"id": "org/renamed", "gated": False, "private": False},
            {"id": "org/no-flags"}]
        session.get.return_value = listed
        rows = owner_matches(PointerChecker(session=session), "base_model", "org",
                             {"org/base", "org/gated", "org/old-name", "org/no-flags"})
        self.assertEqual({r["dependency_id"] for r in rows}, {"org/base", "org/gated"})
        self.assertEqual(rows[1]["status"], "gated")
        self.assertTrue(all(r["lookup_method"] == "owner_listing" for r in rows))

    @patch("check_pointers.time.sleep")
    def test_missing_listing_match_falls_back_to_individual_checks(self, sleep):
        models = [{"id": "org/model", "tags": ["base_model:owner/" + str(i) for i in range(5)]}]
        checker = Mock()
        checker.check.side_effect = lambda kind, dependency: {
            "dependency_type": kind, "dependency_id": dependency, "status": "metadata_accessible"}
        with tempfile.TemporaryDirectory() as directory, patch("check_pointers.owner_matches", return_value=[]):
            result = run(models, directory, checker, prefetch_owners=True)
        self.assertEqual(checker.check.call_count, 5)
        self.assertEqual(result["pointer_count"], 5)


if __name__ == "__main__":
    unittest.main()
