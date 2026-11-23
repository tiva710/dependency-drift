"""Evaluate dependency evidence and report model-level RQ1 counts."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import time
from urllib.parse import quote, urlparse
import zipfile

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version
import requests

from check_pointers import utc_now

RULE_VERSION = "rq1-v1"
TRANSIENT = {"request_error", "rate_limited", "server_error", "unexpected_response"}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def combined(values):
    """Combine drift signals: Y dominates uncertainty, N requires all N."""
    values = list(values)
    if "Y" in values:
        return "Y"
    return "N" if values and all(v == "N" for v in values) else ""


def compact_notes(notes):
    """Keep model summaries readable; detailed evidence remains in its own tables."""
    selected = []
    length = 0
    for note in notes:
        if length + len(note) + 3 > 11000:
            break
        selected.append(note)
        length += len(note) + 3
    omitted = len(notes) - len(selected)
    if omitted:
        selected.append(f"{omitted} further entries in pointers.csv and library_checks.csv; match by model ID")
    return " | ".join(selected)


def declarations(archive_path, model_ids):
    rows = []
    with zipfile.ZipFile(archive_path) as archive:
        for folder in ("configs/", "requirements/"):
            names = [n for n in archive.namelist() if n.startswith(folder) and n.endswith(".csv")]
            if len(names) != 1:
                raise ValueError(f"Expected one {folder} CSV in archive")
            with io.TextIOWrapper(archive.open(names[0]), encoding="utf-8-sig") as stream:
                for row in csv.DictReader(stream):
                    if row["Model ID"] not in model_ids:
                        continue
                    rows.append({"model_id": row["Model ID"], "file": row["file"],
                                 "source": names[0], "source_key": row.get("key", row.get("line_no", "")),
                                 "package": canonicalize_name(row["package"]),
                                 "version_spec": row.get("specifier", row.get("version", "")),
                                 "kind": row.get("kind", "pinned"), "raw": row.get("raw", "")})
    return rows


def library_id(row):
    """Scope review evidence to a single version declaration and source."""
    return f"{row['package']} {row['version_spec']} @ {row['file']} #{row['source_key']}"


class ReleaseChecker:
    def __init__(self, session=None):
        # A separate session keeps Hub credentials off PyPI requests.
        self.session = session or requests.Session()

    def check(self, package):
        url = f"https://pypi.org/pypi/{quote(package, safe='')}/json"
        result = {"package": package, "request_url": url, "checked_at": utc_now()}
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", package):
            return {**result, "status": "invalid_package_name"}
        for attempt in range(3):
            try:
                response = self.session.get(url, timeout=30)
                result["http_status"] = response.status_code
                if response.status_code == 404:
                    return {**result, "status": "project_not_found"}
                if response.status_code == 429:
                    return {**result, "status": "rate_limited"}
                if response.status_code >= 500:
                    result["status"] = "server_error"
                elif response.status_code == 200:
                    data = response.json()
                    releases = data.get("releases")
                    if not isinstance(releases, dict):
                        raise ValueError("Missing release history")
                    compact = {}
                    for version, files in releases.items():
                        if not isinstance(files, list):
                            raise ValueError("Invalid release history")
                        compact[version] = [{"filename": f["filename"],
                                             "yanked": bool(f.get("yanked", False)),
                                             "yanked_reason": f.get("yanked_reason"),
                                             "upload_time": f.get("upload_time_iso_8601")}
                                            for f in files]
                    return {**result, "status": "ok", "releases": compact}
                else:
                    result["status"] = "unexpected_response"
            except requests.RequestException:
                result["status"] = "request_error"
            except (ValueError, KeyError, TypeError):
                result["status"] = "unexpected_response"
            if attempt < 2:
                time.sleep(2 ** attempt)
        return result


def release_cache(path, packages, offline=False):
    cache = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
                cache[row["package"]] = row
            except (ValueError, KeyError):
                continue
    checker = ReleaseChecker()
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n")
        for index, package in enumerate(sorted(packages), 1):
            if package not in cache or cache[package]["status"] in TRANSIENT:
                if offline:
                    continue
                row = checker.check(package)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                cache[package] = row
                print(f"PyPI {index}/{len(packages)} {package}: {row['status']}", flush=True)
                if row["status"] == "rate_limited":
                    raise SystemExit("PyPI rate limited; progress saved. Rerun after the quota resets.")
    return cache


def library_result(declaration, release):
    result = {**declaration, "status": "unresolved", "stale": "", "rq1_drift": "",
              "pinned_version": "", "latest_version": "", "major_gap": "",
              "release_yanked": "", "checked_at": release.get("checked_at", ""),
              "request_url": release.get("request_url", "")}
    if declaration["kind"] != "pinned":
        return {**result, "status": declaration["kind"] + "_requires_review"}
    if release.get("status") != "ok":
        return {**result, "status": release.get("status", "not_checked")}
    spec = declaration["version_spec"].strip()
    try:
        if spec.startswith(("==", "===")):
            terms = list(SpecifierSet(spec))
            if len(terms) != 1 or terms[0].operator not in {"==", "==="} or "*" in terms[0].version:
                return {**result, "status": "nonexact_pin"}
            spec = terms[0].version
        pin = Version(spec)
    except (InvalidVersion, InvalidSpecifier):
        return {**result, "status": "invalid_pin"}
    result["pinned_version"] = str(pin)
    versions = {}
    for name, files in release["releases"].items():
        try:
            version = Version(name)
        except InvalidVersion:
            continue
        versions.setdefault(version, []).extend(files)
    files = versions.get(pin, [])
    if not files:
        return {**result, "status": "release_not_verified"}
    result["release_yanked"] = all(f["yanked"] for f in files)
    stable = [v for v, artifacts in versions.items()
              if not v.is_prerelease and not v.is_devrelease and not v.local
              and any(not f["yanked"] for f in artifacts)]
    if not stable:
        return {**result, "status": "no_stable_comparison"}
    latest = max(stable)
    result["latest_version"] = str(latest)
    if pin.is_prerelease or pin.is_devrelease or pin.local or pin.epoch != latest.epoch:
        return {**result, "status": "nonstandard_pin_requires_review"}
    gap = max(0, latest.major - pin.major)
    result.update(major_gap=gap, stale="Y" if gap > 2 else "N", status="verified_pin")
    # PyPI availability and version distance do not establish breaking changes.
    newer = any(v > pin and any(not f["yanked"] for f in artifacts)
                for v, artifacts in versions.items() if not v.is_devrelease and not v.is_prerelease)
    if not newer and not result["release_yanked"]:
        result["rq1_drift"] = "N"
    return result


def load_evidence(path, model_ids, dependency_keys):
    result = {}
    if path is None:
        return result
    for row in read_csv(path):
        key = (row["model_id"], row["dependency_type"], row["dependency_id"])
        if key[0] not in model_ids or key not in dependency_keys:
            raise ValueError(f"Evidence does not match a declared dependency: {key}")
        if key in result:
            raise ValueError(f"Duplicate evidence: {key}")
        if row["verdict"] not in {"available", "unavailable", "breaking_change", "no_breaking_change"}:
            raise ValueError(f"Unsupported verdict: {row['verdict']}")
        if key[1] != "library" and row["verdict"] in {"breaking_change", "no_breaking_change"}:
            raise ValueError("Breaking-change verdicts apply to libraries")
        source = urlparse(row["source_url"])
        if source.scheme not in {"http", "https"} or not source.netloc or not row["reason"].strip():
            raise ValueError("Evidence needs a source URL and explanation")
        result[key] = row
    return result


def pointer_labels(pointers, overrides):
    existence = []
    drift = []
    notes = []
    for pointer in pointers:
        key = (pointer["model_id"], pointer["dependency_type"], pointer["dependency_id"])
        evidence = overrides.get(key, {})
        verdict = evidence.get("verdict")
        known = pointer["status"] in {"metadata_accessible", "gated", "private"}
        existence.append("N" if verdict == "unavailable" else "Y" if known or verdict == "available" else "")
        renamed = str(pointer.get("id_changed", "")).lower() == "true"
        accessible = (pointer["status"] == "metadata_accessible"
                      or pointer.get("access_status") == "accessible" or verdict == "available")
        label = "Y" if renamed or verdict == "unavailable" else "N" if accessible else ""
        drift.append(label)
        notes.append(f"{pointer['dependency_type']} {pointer['dependency_id']}: {pointer['status']}"
                     + (f" -> {pointer['resolved_id']}" if renamed else "")
                     + (f"; read access={pointer['access_status']}" if pointer.get("access_status") else "")
                     + (f"; {evidence['source_url']} {evidence['reason']}" if evidence else ""))
    exists = "N" if "N" in existence else "Y" if existence and all(v == "Y" for v in existence) else ""
    return exists, combined(drift), notes


def assess(models, pointers, libraries, evidence):
    pointer_groups = defaultdict(list)
    library_groups = defaultdict(list)
    for pointer in pointers:
        pointer_groups[(pointer["model_id"], pointer["dependency_type"])].append(pointer)
    for row in libraries:
        row["dependency_id"] = library_id(row)
        key = (row["model_id"], "library", row["dependency_id"])
        reviewed = evidence.get(key, {})
        if reviewed.get("verdict") in {"unavailable", "breaking_change"}:
            row["rq1_drift"] = "Y"
        elif reviewed.get("verdict") == "no_breaking_change" and row["status"] == "verified_pin" and not row["release_yanked"]:
            row["rq1_drift"] = "N"
        row["review_evidence"] = json.dumps(reviewed) if reviewed else ""
        library_groups[row["model_id"]].append(row)
    output = []
    for model in models:
        mid = model["Model ID"]
        base, base_drift, base_notes = pointer_labels(pointer_groups[(mid, "base_model")], evidence)
        dataset, dataset_drift, dataset_notes = pointer_labels(pointer_groups[(mid, "dataset")], evidence)
        libs = library_groups[mid]
        stale = combined(r["stale"] for r in libs)
        lib_drift = combined(r["rq1_drift"] for r in libs)
        library_candidates = combined(
            "Y" if r["stale"] == "Y" or r["rq1_drift"] == "Y"
            else "N" if r["status"] == "verified_pin" and r["stale"] == "N" and not r["release_yanked"]
            else "" for r in libs)
        drift = combined([base_drift, dataset_drift, lib_drift])
        candidate = combined([base_drift, dataset_drift, library_candidates])
        notes = (base_notes or ["base_model: no declared reference in snapshot"])
        notes += dataset_notes or ["dataset: no declared reference in snapshot"]
        notes += [f"{r['file']} {r['package']} {r['version_spec']}: {r['status']}; latest={r['latest_version']}; gap={r['major_gap']}; yanked={r['release_yanked']}"
                  + (f"; raw={r['raw']}" if r.get("raw") and r["kind"] not in {"pinned", "ranged", "unpinned"} else "")
                  for r in libs]
        if not libs:
            notes.append("No library declaration found in archive CSVs")
        if not drift:
            notes.insert(0, "RQ1 unresolved; review missing or breaking-change evidence")
        status_summary = [
            "Base lookup statuses=" + json.dumps(dict(Counter(p["status"] for p in pointer_groups[(mid, "base_model")]))),
            "Dataset lookup statuses=" + json.dumps(dict(Counter(p["status"] for p in pointer_groups[(mid, "dataset")]))),
            "Library statuses=" + json.dumps(dict(Counter(r["status"] for r in libs)))]
        output.append({"Model ID": mid, "base_model_exists (Y/N)": base,
                       "dataset_exists (Y/N)": dataset, "library_version_stale (Y/N)": stale,
                       "drift (Y/N)": candidate,
                       "rq1_confirmed_drift (Y/N)": drift,
                       "base_model_drift": base_drift, "dataset_drift": dataset_drift,
                       "library_drift": lib_drift, "notes": compact_notes(status_summary + notes)})
    return output


def summarize(rows):
    counts = Counter(row["drift (Y/N)"] for row in rows)
    confirmed = Counter(row["rq1_confirmed_drift (Y/N)"] for row in rows)
    total = len(rows)
    return {"generated_at": utc_now(), "rule_version": RULE_VERSION, "models_total": total,
            "drift_yes": counts["Y"], "drift_no": counts["N"], "unresolved": counts[""],
            "candidate_fraction_of_all_models": counts["Y"] / total,
            "candidate_fraction_bounds_with_unresolved": [counts["Y"] / total, (counts["Y"] + counts[""]) / total],
            "rq1_confirmed_yes": confirmed["Y"], "rq1_confirmed_no": confirmed["N"],
            "rq1_unresolved": confirmed[""],
            "confirmed_fraction_of_all_models": confirmed["Y"] / total,
            "fraction_bounds_with_unresolved": [confirmed["Y"] / total, (confirmed["Y"] + confirmed[""]) / total],
            "resolved_only_fraction": confirmed["Y"] / (confirmed["Y"] + confirmed["N"]) if confirmed["Y"] + confirmed["N"] else None,
            "models_by_dependency_type": {kind: dict(Counter(r[field] or "unresolved" for r in rows))
                                          for kind, field in [("base_model", "base_model_drift"),
                                                              ("dataset", "dataset_drift"),
                                                              ("library", "library_drift")]},
            "label_counts": {field: dict(Counter(r[field] or "unresolved" for r in rows))
                             for field in ["base_model_exists (Y/N)", "dataset_exists (Y/N)",
                                           "library_version_stale (Y/N)"]},
            "categories_models": {kind: sum(r[kind] == "Y" for r in rows)
                                  for kind in ("base_model_drift", "dataset_drift", "library_drift")},
            "note": "Bounds reflect unresolved classifications, not a statistical confidence interval. Category counts may overlap. Observations concern declared dependencies; runtime compatibility and historical replacement are not exhaustively tested."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="v2 dataset ZIP")
    parser.add_argument("--model-csv", type=Path, required=True)
    parser.add_argument("--pointer-dir", type=Path, required=True, help="Completed pointer-check output")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-csv", type=Path)
    parser.add_argument("--offline", action="store_true", help="Use cached PyPI observations only")
    parser.add_argument("--expected-models", type=int)
    args = parser.parse_args()
    models = read_csv(args.model_csv)
    ids = [r["Model ID"] for r in models]
    if not ids or not all(ids) or len(set(ids)) != len(ids):
        parser.error("Model IDs must be nonempty and unique")
    if args.expected_models and len(ids) != args.expected_models:
        parser.error("Model count does not match expected-models")
    manifest = json.loads((args.pointer_dir / "manifest.json").read_text())
    if manifest["model_ids"] != ids:
        parser.error("Pointer manifest must match the model CSV in order")
    summary = json.loads((args.pointer_dir / "summary.json").read_text())
    pointers = read_csv(args.pointer_dir / "pointers.csv")
    if summary["models_checked"] != len(ids) or summary["pointer_count"] != len(pointers):
        parser.error("Pointer run is incomplete")
    # Check the pointer identities against their original snapshot declarations.
    from check_pointers import extract_pointers, load_models, select_sample
    metadata = select_sample(load_models(args.input), args.model_csv)
    expected = {(p["model_id"], p["dependency_type"], p["dependency_id"])
                for m in metadata for p in extract_pointers(m)}
    actual = [(p["model_id"], p["dependency_type"], p["dependency_id"]) for p in pointers]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        parser.error("Pointer rows do not match archive references")
    declared = declarations(args.input, set(ids))
    keys = expected | {(r["model_id"], "library", library_id(r)) for r in declared}
    evidence = load_evidence(args.evidence_csv, set(ids), keys)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bound = {"rule_version": RULE_VERSION, "input": str(args.input.resolve()),
             "model_ids": ids, "declarations_sha256": hashlib.sha256(json.dumps(declared, sort_keys=True).encode()).hexdigest()}
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != bound:
        parser.error("Assessment directory belongs to different inputs or rules")
    manifest_path.write_text(json.dumps(bound, indent=2) + "\n")
    releases = release_cache(args.output_dir / "releases.jsonl", {r["package"] for r in declared if r["package"]}, args.offline)
    libraries = [library_result(row, releases.get(row["package"], {})) for row in declared]
    labels = assess(models, pointers, libraries, evidence)
    write_csv(args.output_dir / "library_checks.csv", libraries,
              list(libraries[0]) if libraries else ["model_id", "package", "status"])
    write_csv(args.output_dir / "model_labels.csv", labels, list(labels[0]))
    result = summarize(labels)
    result["pointer_auth"] = manifest.get("auth", "unknown")
    result["pointer_dir"] = str(args.pointer_dir.resolve())
    result["pointer_manifest_sha256"] = hashlib.sha256((args.pointer_dir / "manifest.json").read_bytes()).hexdigest()
    result["pointers_sha256"] = hashlib.sha256((args.pointer_dir / "pointers.csv").read_bytes()).hexdigest()
    result["pointer_observation_range"] = [min((r["checked_at"] for r in pointers), default=None),
                                           max((r["checked_at"] for r in pointers), default=None)]
    result["release_observation_range"] = [min((r["checked_at"] for r in libraries if r["checked_at"]), default=None),
                                           max((r["checked_at"] for r in libraries if r["checked_at"]), default=None)]
    result["access_observation_range"] = [min((r["access_checked_at"] for r in pointers if r.get("access_checked_at")), default=None),
                                          max((r["access_checked_at"] for r in pointers if r.get("access_checked_at")), default=None)]
    if args.evidence_csv:
        result["evidence_sha256"] = hashlib.sha256(args.evidence_csv.read_bytes()).hexdigest()
        write_csv(args.output_dir / "review_evidence.csv", list(evidence.values()),
                  ["model_id", "dependency_type", "dependency_id", "verdict", "source_url", "reason"])
    result["model_labels_sha256"] = hashlib.sha256((args.output_dir / "model_labels.csv").read_bytes()).hexdigest()
    result["library_checks_sha256"] = hashlib.sha256((args.output_dir / "library_checks.csv").read_bytes()).hexdigest()
    temporary = args.output_dir / "summary.json.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output_dir / "summary.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
