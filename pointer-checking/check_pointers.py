"""Check current Hub metadata availability for declared model dependencies."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import time
from urllib.parse import quote
import zipfile

import requests


RELATIONS = {"adapter", "finetune", "merge", "quantized"}
RETRY_STATUSES = {"request_error", "rate_limited", "server_error", "unexpected_response"}
FIELDS = ["model_id", "dependency_type", "dependency_id", "source_tags", "status",
          "http_status", "resolved_id", "gated", "private", "sha", "checked_at",
          "request_url", "response_url", "attempts"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


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


class PointerChecker:
    def __init__(self, session=None, attempts=3, interval=0.25):
        self.session = session or requests.Session()
        self.attempts = attempts
        self.interval = interval

    def check(self, kind, dependency):
        result = {"dependency_type": kind, "dependency_id": dependency,
                  "checked_at": utc_now(), "http_status": None, "resolved_id": None,
                  "gated": None, "private": None, "sha": None,
                  "request_url": None, "response_url": None, "attempts": 0}
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
                    if not isinstance(data, dict) or not isinstance(data.get("id"), str):
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


def run(models, output, checker, refresh=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    journal = output / "checks.jsonl"
    cache = {} if refresh else load_cache(journal)
    counts = Counter()
    missing = []
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
                    progress.write(json.dumps(cache[key]) + "\n")
                    progress.flush()
                result = cache[key]
                writer.writerow({**pointer, **result,
                                 "source_tags": json.dumps(pointer["source_tags"])})
                counts[result["status"]] += 1
    summary = {"generated_at": utc_now(), "models_checked": len(models),
               "pointer_count": sum(counts.values()), "statuses_by_pointer": dict(counts),
               "missing_reference_types": missing,
               "note": "Current anonymous metadata checks; not final drift labels."}
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
    args = parser.parse_args()
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
                "sample_size": args.sample_size, "seed": args.seed, "auth": "anonymous"}
    if args.sample_csv is not None:
        manifest["sample_csv"] = str(args.sample_csv.resolve())
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        parser.error("Output directory belongs to a different input/sample; choose a new directory")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        pointers = [p for m in models for p in extract_pointers(m)]
        (args.output_dir / "references.json").write_text(json.dumps(pointers, indent=2) + "\n")
        print(f"Extracted {len(pointers)} pointers from {len(models)} models; no network requests")
    else:
        summary = run(models, args.output_dir, PointerChecker(), args.refresh)
        print(json.dumps({k: v for k, v in summary.items() if k != "missing_reference_types"}, indent=2))


if __name__ == "__main__":
    main()
