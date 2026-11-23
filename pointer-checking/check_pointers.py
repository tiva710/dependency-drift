"""Check current Hub metadata availability for declared model dependencies."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from urllib.parse import quote
import zipfile

import requests


RELATIONS = {"adapter", "finetune", "merge", "quantized"}
RETRY_STATUSES = {"request_error", "rate_limited", "server_error", "unexpected_response"}
FIELDS = ["model_id", "dependency_type", "dependency_id", "source_tags", "status",
          "http_status", "resolved_id", "gated", "private", "sha", "checked_at",
          "request_url", "response_url", "attempts", "lookup_method",
          "access_status", "access_http_status", "access_checked_at", "access_request_url"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def hub_token():
    """Read the environment token or the current local Hub login."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    home = Path(os.environ.get("HF_HOME", Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "huggingface"))
    path = Path(os.environ.get("HF_TOKEN_PATH", home / "token"))
    return path.read_text().strip() if path.is_file() else None


def load_models(path):
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist()
                     if n.startswith("metadata/") and n.endswith(".json")]
            if len(names) != 1:
                raise ValueError("ZIP must contain exactly one metadata/*.json file")
            with archive.open(names[0]) as stream:
                models = json.load(stream)
    else:
        with path.open(encoding="utf-8") as stream:
            models = json.load(stream)
    if not isinstance(models, list) or any(
        not isinstance(m, dict) or not isinstance(m.get("id"), str) or not m["id"]
        for m in models
    ):
        raise ValueError("Expected a JSON array of model metadata objects with string ids")
    # Match the collection pipeline's first-record-wins deduplication.
    unique = {}
    for model in models:
        unique.setdefault(model["id"], model)
    return list(unique.values())


def extract_pointers(model):
    references = {}
    for tag in model.get("tags", []) or []:
        if not isinstance(tag, str):
            continue
        if tag.startswith("base_model:"):
            kind, value = "base_model", tag[len("base_model:"):]
            relation, separator, remainder = value.partition(":")
            if separator and relation in RELATIONS:
                value = remainder
        elif tag.startswith("dataset:"):
            kind, value = "dataset", tag[len("dataset:"):]
        else:
            continue
        key = (kind, value.strip())
        references.setdefault(key, [])
        if tag not in references[key]:
            references[key].append(tag)
    return [{"model_id": model["id"], "dependency_type": kind,
             "dependency_id": dependency, "source_tags": tags}
            for (kind, dependency), tags in references.items()]


def select_sample(models, path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if "Model ID" not in (reader.fieldnames or []):
            raise ValueError("Sample CSV requires a Model ID column")
        ids = [(row["Model ID"] or "").strip() for row in reader]
    if not ids or not all(ids):
        raise ValueError("Sample CSV must contain nonempty model IDs")
    duplicates = [mid for mid, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate sample IDs: {', '.join(duplicates[:10])}")
    by_id = {model["id"]: model for model in models}
    missing = [mid for mid in ids if mid not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} sample IDs absent from metadata: {', '.join(missing[:10])}")
    return [by_id[mid] for mid in ids]


def valid_id(value):
    # Legacy dataset IDs can be a single component (e.g. snli).
    parts = value.split("/")
    return (1 <= len(parts) <= 2 and all(len(p) <= 96 for p in parts) and
            all(re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", p) and
                not p.endswith((".", "-")) for p in parts) and
            ".." not in value and "--" not in value)


def rate_limit_delay(headers):
    reset = re.search(r"(?:^|;)\s*t=(\d+)", headers.get("RateLimit", ""))
    delays = [int(reset[1])] if reset else []
    retry_after = headers.get("Retry-After", "").strip()
    if retry_after.isdecimal():
        delays.append(int(retry_after))
    elif retry_after:
        try:
            delays.append(parsedate_to_datetime(retry_after).timestamp() - time.time())
        except (ValueError, TypeError, OverflowError):
            pass
    return max(1, max(delays, default=300) + 1)


class PointerChecker:
    def __init__(self, session=None, attempts=3, interval=0.25, token=None):
        self.session = session or requests.Session()
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        self.attempts = attempts
        self.interval = interval

    def check(self, kind, dependency):
        result = {"dependency_type": kind, "dependency_id": dependency,
                  "checked_at": utc_now(), "http_status": None, "resolved_id": None,
                  "gated": None, "private": None, "sha": None,
                  "request_url": None, "response_url": None, "attempts": 0,
                  "lookup_method": "repository"}
        if not valid_id(dependency):
            return {**result, "status": "invalid_reference"}
        category = "models" if kind == "base_model" else "datasets"
        url = f"https://huggingface.co/api/{category}/{quote(dependency, safe='/')}"
        result["request_url"] = url
        for attempt in range(1, self.attempts + 1):
            result.update(attempts=attempt, checked_at=utc_now(), http_status=None,
                          response_url=None)
            time.sleep(self.interval)
            response = None
            try:
                response = self.session.get(url, timeout=30)
                result.update(http_status=response.status_code, response_url=response.url)
                code = response.status_code
                if code == 200:
                    data = response.json()
                    if not isinstance(data, dict) or not isinstance(data.get("id"), str) or not valid_id(data["id"]):
                        raise ValueError("Missing repository id")
                    result.update(resolved_id=data["id"], gated=data.get("gated"),
                                  private=data.get("private"), sha=data.get("sha"))
                    if data.get("private") is True:
                        status = "private"
                    elif data.get("gated") in (True, "auto", "manual"):
                        status = "gated"
                    else:
                        status = "metadata_accessible"
                    # Keep rename evidence independent of gating/access status.
                    result["id_changed"] = data["id"].casefold() != dependency.casefold()
                elif code in (401, 403):
                    status = "access_restricted"
                elif code == 404:
                    status = "not_found_or_private"
                elif code == 429:
                    status = "rate_limited"
                elif code >= 500:
                    status = "server_error"
                else:
                    status = "unexpected_response"
            except requests.RequestException:
                status = "request_error"
            except ValueError:
                status = "unexpected_response"
            result["status"] = status
            if status not in RETRY_STATUSES or attempt == self.attempts:
                break
            if status == "rate_limited":
                delay = rate_limit_delay(response.headers)
                print(f"Hub rate limit reached. Waiting {delay:.0f}s before retrying {dependency}.",
                      file=sys.stderr, flush=True)
                time.sleep(delay)
                continue
            delay = 2 ** attempt
            if response is not None:
                try:
                    delay = max(delay, float(response.headers.get("Retry-After", 0)))
                except ValueError:
                    # A date-form Retry-After is left for the next invocation.
                    break
            if delay > 60:
                break
            time.sleep(delay)
        return result

    def check_access(self, result):
        """Check read permission for a known gated/private repository."""
        category = "models" if result["dependency_type"] == "base_model" else "datasets"
        dependency = result.get("resolved_id") or result["dependency_id"]
        url = f"https://huggingface.co/api/{category}/{quote(dependency, safe='/')}/auth-check"
        result = {**result, "access_request_url": url}
        for attempt in range(1, self.attempts + 1):
            result.update(access_checked_at=utc_now(), access_http_status=None)
            time.sleep(self.interval)
            response = None
            try:
                response = self.session.get(url, timeout=30)
                code = response.status_code
                result["access_http_status"] = code
                if code == 200:
                    status = "accessible"
                elif code in (401, 403):
                    status = "denied"
                elif code == 404:
                    status = "not_found_or_private"
                elif code == 429:
                    status = "rate_limited"
                elif code >= 500:
                    status = "server_error"
                else:
                    status = "unexpected_response"
            except requests.RequestException:
                status = "request_error"
            result["access_status"] = status
            if status not in RETRY_STATUSES or attempt == self.attempts:
                break
            delay = rate_limit_delay(response.headers) if status == "rate_limited" else 2 ** attempt
            if status == "rate_limited":
                print(f"Hub access-check quota reached. Waiting {delay:.0f}s.", file=sys.stderr, flush=True)
            time.sleep(delay)
        return result


def load_cache(path):
    cache = {}
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    result = json.loads(line)
                    key = (result["dependency_type"], result["dependency_id"])
                    if result["status"] in RETRY_STATUSES:
                        cache.pop(key, None)
                    else:
                        cache[key] = result
                except (ValueError, KeyError):
                    continue
    return cache


def check_batch(keys, token, access=False):
    """Use one session per worker; the caller owns all journal writes."""
    def check(key):
        checker = PointerChecker(token=token)
        try:
            if access:
                return checker.check_access(key)
            return checker.check(*key)
        finally:
            checker.session.close()
    with ThreadPoolExecutor(max_workers=len(keys)) as pool:
        return list(pool.map(check, keys))


def owner_matches(checker, kind, owner, dependencies):
    """Only exact positive listing matches can replace individual lookups."""
    category = "models" if kind == "base_model" else "datasets"
    url = f"https://huggingface.co/api/{category}"
    for attempt in range(1, checker.attempts + 1):
        try:
            time.sleep(checker.interval)
            response = checker.session.get(url, params={"author": owner, "limit": 1000, "full": "true"}, timeout=60)
            if response.status_code == 429 and attempt < checker.attempts:
                time.sleep(rate_limit_delay(response.headers))
                continue
            if response.status_code != 200:
                return []
            data = response.json()
            if not isinstance(data, list):
                return []
            wanted = {dependency.casefold(): dependency for dependency in dependencies}
            matches = []
            for item in data:
                if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                    continue
                dependency = wanted.get(item["id"].casefold())
                if not dependency or not isinstance(item.get("private"), bool) or "gated" not in item:
                    continue
                gated = item["gated"]
                if gated not in (False, True, "auto", "manual"):
                    continue
                status = "private" if item["private"] else "gated" if gated else "metadata_accessible"
                matches.append({"dependency_type": kind, "dependency_id": dependency,
                                "checked_at": utc_now(), "http_status": 200,
                                "resolved_id": item["id"], "gated": gated,
                                "private": item["private"], "sha": item.get("sha"),
                                "request_url": response.url, "response_url": response.url,
                                "attempts": attempt, "status": status, "id_changed": False,
                                "lookup_method": "owner_listing"})
            return matches
        except (requests.RequestException, ValueError):
            continue
    return []


def run(models, output, checker, refresh=False, workers=1, token=None, prefetch_owners=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    journal = output / "checks.jsonl"
    cache = {} if refresh else load_cache(journal)
    counts = Counter()
    missing = []
    if prefetch_owners:
        groups = defaultdict(set)
        for model in models:
            for pointer in extract_pointers(model):
                kind, dependency = pointer["dependency_type"], pointer["dependency_id"]
                if (kind, dependency) not in cache and valid_id(dependency) and "/" in dependency:
                    groups[(kind, dependency.split("/")[0])].add(dependency)
        with journal.open("a", encoding="utf-8") as progress:
            progress.write("\n")
            for (kind, owner), dependencies in groups.items():
                if len(dependencies) < 5:
                    continue
                results = owner_matches(checker, kind, owner, dependencies)
                for result in results:
                    cache[(kind, result["dependency_id"])] = result
                    progress.write(json.dumps(result) + "\n")
                    progress.flush()
                print(f"Owner listing {kind} {owner}: verified {len(results)}/{len(dependencies)} references", flush=True)
    if workers > 1:
        keys = list(dict.fromkeys((p["dependency_type"], p["dependency_id"])
                                 for model in models for p in extract_pointers(model)))
        pending = [key for key in keys if key not in cache]
        with journal.open("a", encoding="utf-8") as progress:
            progress.write("\n")
            for start in range(0, len(pending), workers):
                batch = pending[start:start + workers]
                results = check_batch(batch, token)
                for key, result in zip(batch, results):
                    progress.write(json.dumps(result) + "\n")
                    progress.flush()
                    cache[key] = result
                if start % 100 < workers or start + workers >= len(pending):
                    print(f"Hub references checked {len(cache)}/{len(keys)}", flush=True)
                if any(result["status"] == "rate_limited" for result in results):
                    raise SystemExit("Hub rate limit persisted. Progress saved; rerun after quota reset.")
    if token:
        pending_access = [key for key, result in cache.items()
                          if result["status"] in {"gated", "private"}
                          and (not result.get("access_status") or result["access_status"] in RETRY_STATUSES)]
        with journal.open("a", encoding="utf-8") as progress:
            progress.write("\n")
            for start in range(0, len(pending_access), workers):
                batch = pending_access[start:start + workers]
                if workers == 1:
                    results = [checker.check_access(cache[batch[0]])]
                else:
                    results = check_batch([cache[key] for key in batch], token, access=True)
                for key, result in zip(batch, results):
                    cache[key] = result
                    progress.write(json.dumps(result) + "\n")
                    progress.flush()
                completed = start + len(batch)
                if start % 25 < workers or completed == len(pending_access):
                    print(f"Gated/private access checks {completed}/{len(pending_access)}", flush=True)
                if any(result["access_status"] == "rate_limited" for result in results):
                    raise SystemExit("Hub access-check rate limit persisted. Progress saved; rerun after quota reset.")
    with journal.open("a", encoding="utf-8") as progress, \
            (output / "pointers.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS + ["id_changed"], lineterminator="\n")
        writer.writeheader()
        # Isolate a partial last line left by an interrupted journal write.
        progress.write("\n")
        for model in models:
            pointers = extract_pointers(model)
            present = {p["dependency_type"] for p in pointers}
            for kind in ("base_model", "dataset"):
                if kind not in present:
                    missing.append({"model_id": model["id"], "dependency_type": kind})
            for pointer in pointers:
                key = (pointer["dependency_type"], pointer["dependency_id"])
                if key not in cache:
                    cache[key] = checker.check(*key)
                    if token and cache[key]["status"] in {"gated", "private"}:
                        cache[key] = checker.check_access(cache[key])
                    progress.write(json.dumps(cache[key]) + "\n")
                    progress.flush()
                    if cache[key]["status"] == "rate_limited":
                        raise SystemExit("Hub rate limit persisted. Progress saved; wait for the quota "
                                         "to reset, then rerun the same command.")
                    if cache[key].get("access_status") == "rate_limited":
                        raise SystemExit("Hub access-check rate limit persisted. Progress saved; rerun after quota reset.")
                result = cache[key]
                writer.writerow({**pointer, **result,
                                 "source_tags": json.dumps(pointer["source_tags"])})
                counts[result["status"]] += 1
    summary = {"generated_at": utc_now(), "models_checked": len(models),
               "pointer_count": sum(counts.values()), "statuses_by_pointer": dict(counts),
               "missing_reference_types": missing,
               "note": "Current Hub metadata checks; see manifest for authentication mode."}
    summary["access_checks_by_unique_reference"] = dict(Counter(
        result["access_status"] for result in cache.values() if result.get("access_status")))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Metadata JSON or shared dataset ZIP")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Use the same directory to resume; a new directory for a new observation")
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--sample-size", type=int, help="Uniform random sample, not stratified")
    sampling.add_argument("--sample-csv", type=Path, help="Use exact IDs from a CSV with a Model ID column")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Extract references without network calls")
    parser.add_argument("--refresh", action="store_true", help="Recheck cached references")
    parser.add_argument("--authenticated", action="store_true", help="Require HF_TOKEN or a cached Hub login")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent checks, from 1 to 8")
    parser.add_argument("--prefetch-owners", action="store_true", help="Use exact owner-list matches before individual checks")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    token = hub_token() if args.authenticated else None
    if args.authenticated and not token:
        parser.error("Authentication requires HF_TOKEN or a cached Hub login")
    checker = PointerChecker(token=token)
    identity = None
    if args.authenticated and not args.dry_run:
        try:
            response = checker.session.get("https://huggingface.co/api/whoami-v2", timeout=30)
        except requests.RequestException:
            parser.error("Unable to validate Hub authentication; check network access")
        if response.status_code != 200:
            parser.error(f"Hub authentication validation failed (HTTP {response.status_code})")
        identity = response.json()["name"]
    models = load_models(args.input)
    if args.sample_csv is not None:
        try:
            models = select_sample(models, args.sample_csv)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    elif args.sample_size is not None:
        if not 1 <= args.sample_size <= len(models):
            parser.error("sample-size must be between 1 and the number of models")
        models = random.Random(args.seed).sample(models, args.sample_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"input": str(args.input.resolve()), "model_ids": [m["id"] for m in models],
                "sample_size": args.sample_size, "seed": args.seed,
                "auth": "authenticated" if args.authenticated else "anonymous"}
    if identity:
        manifest["account"] = identity
    if args.sample_csv is not None:
        manifest["sample_csv"] = str(args.sample_csv.resolve())
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if args.authenticated and args.dry_run and previous.get("account"):
            manifest["account"] = previous["account"]
        # Bind an offline extraction to the validated account on its first live run.
        if identity and not previous.get("account") and not (args.output_dir / "checks.jsonl").exists():
            previous["account"] = identity
        if previous != manifest:
            parser.error("Output directory belongs to a different input/sample/account; choose a new directory")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        pointers = [p for m in models for p in extract_pointers(m)]
        (args.output_dir / "references.json").write_text(json.dumps(pointers, indent=2) + "\n")
        print(f"Extracted {len(pointers)} pointers from {len(models)} models; no network requests")
    else:
        summary = run(models, args.output_dir, checker, args.refresh, args.workers, token, args.prefetch_owners)
        print(json.dumps({k: v for k, v in summary.items() if k != "missing_reference_types"}, indent=2))


if __name__ == "__main__":
    main()
