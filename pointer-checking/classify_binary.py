"""Assign binary dependency-screening labels from saved authenticated observations."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from assess_drift import compact_notes, read_csv, write_csv
from check_pointers import RETRY_STATUSES, utc_now

RULE_VERSION = "binary-screening-v2"
LABEL_FIELDS = ["base_model_exists (Y/N)", "dataset_exists (Y/N)",
                "library_version_stale (Y/N)", "drift (Y/N)"]
EXISTS = {"metadata_accessible", "gated", "private"}


def published_versions(release):
    versions = []
    for text, files in release.get("releases", {}).items():
        try:
            version = Version(text)
        except InvalidVersion:
            continue
        if not version.is_prerelease and not version.is_devrelease and not version.local and any(
                not f["yanked"] for f in files):
            versions.append(version)
    return versions


def classify_library(row, versions):
    result = {**row, "binary_stale": "N", "binary_drift": "N", "review_required": "N",
              "comparison_version": "", "binary_major_gap": "", "classification_reason": ""}
    kind = row["kind"]
    if kind == "unpinned":
        result["classification_reason"] = "Unpinned declaration permits current releases; no old-version restriction"
        return result
    if kind not in {"pinned", "ranged"}:
        result.update(review_required="Y", classification_reason=f"{kind} entry supplies no comparable version; no staleness signal")
        return result
    if not versions:
        result.update(review_required="Y", classification_reason="No stable comparison available from PyPI")
        if row["status"] == "project_not_found":
            result.update(binary_drift="Y", classification_reason="Declared package was not found on PyPI; verify its name and index")
        return result
    latest = max(versions)
    result["comparison_version"] = str(latest)
    spec = row["version_spec"].strip()
    try:
        if kind == "ranged" or "*" in spec:
            allowed = SpecifierSet(spec)
            matching = [v for v in versions if v in allowed]
            if not matching:
                result.update(binary_drift="Y", review_required="Y",
                              classification_reason="No stable non-yanked PyPI release satisfies the declared range")
                return result
            declared = max(matching)
            basis = f"Newest release allowed by {spec} is {declared}"
        else:
            if spec.startswith(("==", "===")):
                terms = list(SpecifierSet(spec))
                if len(terms) != 1 or terms[0].operator not in {"==", "==="}:
                    raise InvalidSpecifier(spec)
                spec = terms[0].version
            # Config crawls can append the source revision to a development version.
            version_text = re.sub(r"@[0-9a-fA-F]{7,40}$", "", spec)
            declared = Version(version_text)
            basis = f"Declared version {declared}"
            if row["status"] != "verified_pin":
                basis += "; version-distance classification complete; exact build publication unverified"
    except (InvalidVersion, InvalidSpecifier):
        result.update(review_required="Y", classification_reason="Version expression could not be compared; no staleness signal")
        return result
    if declared.epoch != latest.epoch:
        result.update(review_required="Y", classification_reason="Version epochs differ; major-number distance is not comparable")
        return result
    gap = max(0, latest.major - declared.major)
    result.update(binary_major_gap=gap, binary_stale="Y" if gap > 2 else "N",
                  classification_reason=f"{basis}; latest stable {latest}; major gap {gap}")
    if gap > 2 or row["rq1_drift"] == "Y":
        result["binary_drift"] = "Y"
    if str(row["release_yanked"]).lower() == "true":
        result.update(binary_drift="Y")
        result["classification_reason"] += "; declared release is fully yanked"
    return result


def classify_pointers(rows, kind):
    notes = []
    exists = bool(rows) and all(r["status"] in EXISTS for r in rows)
    flagged = not rows
    review = False
    for row in rows:
        if row["status"] in RETRY_STATUSES or row.get("access_status") in RETRY_STATUSES:
            raise ValueError("Retry incomplete network checks before binary classification")
        if row["status"] not in EXISTS | {"not_found_or_private", "invalid_reference", "access_restricted"}:
            raise ValueError(f"Unsupported pointer observation: {row['status']}")
        if row["status"] in {"gated", "private"} and row.get("access_status") not in {"accessible", "denied"}:
            raise ValueError("Complete the gated read-permission check before classification")
        renamed = str(row.get("id_changed", "")).lower() == "true"
        unavailable = row["status"] not in EXISTS
        blocked = row["status"] in {"gated", "private"} and row.get("access_status") != "accessible"
        flagged = flagged or renamed or unavailable or blocked
        if renamed or unavailable or blocked:
            notes.append(f"{kind} {row['dependency_id']}: {row['status']}"
                         + (f" -> {row['resolved_id']}" if renamed else "")
                         + (f"; read access={row.get('access_status') or 'unverified'}" if blocked else ""))
    if not rows:
        notes.append(f"{kind}: no declared reference; availability not demonstrated")
    return "Y" if exists else "N", flagged, review, notes


def apply_resolutions(libraries, path):
    """Apply source-backed findings to exact declarations without broad overrides."""
    if path is None:
        return []
    by_key = defaultdict(list)
    for row in libraries:
        by_key[(row["model_id"], row["dependency_id"])].append(row)
    seen = set()
    resolutions = read_csv(path)
    for resolution in resolutions:
        key = (resolution["model_id"], resolution["dependency_id"])
        if key not in by_key or key in seen:
            raise ValueError(f"Unmatched or duplicate review resolution: {key}")
        seen.add(key)
        if any(resolution[field] not in {"Y", "N"} for field in ("binary_stale", "binary_drift")):
            raise ValueError("Reviewed classifications require Y/N values")
        if not resolution["source_url"].startswith("https://") or not resolution["reason"].strip():
            raise ValueError("Review resolution requires an evidence URL and reason")
        for row in by_key[key]:
            row.update(binary_stale=resolution["binary_stale"], binary_drift=resolution["binary_drift"],
                       review_required="N", classification_reason=resolution["reason"],
                       review_resolution=resolution["resolution"], resolution_source=resolution["source_url"])
            for field in ("comparison_version", "binary_major_gap"):
                if field in resolution:
                    row[field] = resolution[field]
    return resolutions


def classify_models(previous, pointers, libraries):
    by_pointer = defaultdict(list)
    by_library = defaultdict(list)
    for row in pointers:
        by_pointer[(row["model_id"], row["dependency_type"])].append(row)
    for row in libraries:
        by_library[row["model_id"]].append(row)
    output = []
    for old in previous:
        mid = old["Model ID"]
        base, base_flag, base_review, base_notes = classify_pointers(by_pointer[(mid, "base_model")], "base_model")
        dataset, dataset_flag, dataset_review, dataset_notes = classify_pointers(by_pointer[(mid, "dataset")], "dataset")
        libs = by_library[mid]
        stale = any(r["binary_stale"] == "Y" for r in libs)
        flagged = base_flag or dataset_flag or stale or any(r["binary_drift"] == "Y" for r in libs)
        # Retain any earlier positive supported by separately reviewed evidence.
        flagged = flagged or old["drift (Y/N)"] == "Y"
        review = base_review or dataset_review or not libs or any(r["review_required"] == "Y" for r in libs)
        notes = ["Binary screening: N means no qualifying signal under these rules",
                 f"Base references={len(by_pointer[(mid, 'base_model')])}; dataset references={len(by_pointer[(mid, 'dataset')])}; library declarations={len(libs)}"]
        numbered_builds = sum(r["binary_major_gap"] != "" and r["status"] != "verified_pin" for r in libs)
        if numbered_builds:
            notes.append(f"Compared {numbered_builds} nonstandard saved version numbers by major distance; exact build publication is recorded separately")
        notes += base_notes + dataset_notes
        notes += [f"{r['file']} {r['package']} {r['version_spec']}: {r['classification_reason']}"
                  + (f"; evidence={r.get('resolution_source')}" if r.get("resolution_source") else "")
                  for r in libs if r["binary_drift"] == "Y" or r["review_required"] == "Y" or r.get("review_resolution") not in {None, "", "version_distance_evaluated"}]
        output.append({"Model ID": mid, "base_model_exists (Y/N)": base,
                       "dataset_exists (Y/N)": dataset, "library_version_stale (Y/N)": "Y" if stale else "N",
                       "drift (Y/N)": "Y" if flagged else "N", "review_required (Y/N)": "Y" if review else "N",
                       "notes": compact_notes(notes)})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resolutions-csv", type=Path, help="Completed source-backed declaration reviews")
    args = parser.parse_args()
    if args.assessment_dir.resolve() == args.output_dir.resolve():
        parser.error("Use a separate output directory for the binary rule version")
    summary = json.loads((args.assessment_dir / "summary.json").read_text())
    if summary["pointer_auth"] != "authenticated":
        parser.error("Binary classification requires authenticated pointer observations")
    pointers_path = Path(summary["pointer_dir"]) / "pointers.csv"
    inputs = {"model_labels.csv": args.assessment_dir / "model_labels.csv",
              "library_checks.csv": args.assessment_dir / "library_checks.csv", "pointers.csv": pointers_path}
    for name, expected in [("model_labels.csv", "model_labels_sha256"),
                           ("library_checks.csv", "library_checks_sha256"), ("pointers.csv", "pointers_sha256")]:
        if hashlib.sha256(inputs[name].read_bytes()).hexdigest() != summary[expected]:
            parser.error(f"Source {name} changed since the completed assessment")
    previous = read_csv(inputs["model_labels.csv"])
    pointers = read_csv(pointers_path)
    libraries = read_csv(inputs["library_checks.csv"])
    release_versions = {}
    for line in (args.assessment_dir / "releases.jsonl").read_text().splitlines():
        try:
            release = json.loads(line)
            release_versions[release["package"]] = published_versions(release)
        except (ValueError, KeyError):
            continue
    evaluated = [classify_library(r, release_versions.get(r["package"], [])) for r in libraries]
    for row in evaluated:
        row["review_resolution"] = "version_distance_evaluated" if row["binary_major_gap"] != "" else ""
        row["resolution_source"] = row.get("request_url", "")
        if str(row["release_yanked"]).lower() == "true":
            row["review_resolution"] = "yank_status_verified"
    resolutions = apply_resolutions(evaluated, args.resolutions_csv)
    labels = classify_models(previous, pointers, evaluated)
    ids = [r["Model ID"] for r in labels]
    if len(ids) != summary["models_total"] or len(set(ids)) != len(ids):
        parser.error("Binary classification did not preserve source model coverage")
    assert all(r[field] in {"Y", "N"} for r in labels for field in LABEL_FIELDS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "model_labels.csv", labels, list(labels[0]))
    library_fields = list(evaluated[0]) if evaluated else ["model_id", "binary_stale", "binary_drift", "review_required", "classification_reason"]
    write_csv(args.output_dir / "library_checks.csv", evaluated, library_fields)
    if resolutions:
        write_csv(args.output_dir / "review_resolutions.csv", resolutions, list(resolutions[0]))
    counts = {field: dict(Counter(r[field] for r in labels)) for field in LABEL_FIELDS + ["review_required (Y/N)"]}
    result = {"generated_at": utc_now(), "rule_version": RULE_VERSION, "models_total": len(labels),
              "label_counts": counts, "candidate_fraction": counts["drift (Y/N)"].get("Y", 0) / len(labels),
              "source_assessment": str(args.assessment_dir.resolve()),
              "pointer_observation_range": summary["pointer_observation_range"],
              "release_observation_range": summary["release_observation_range"],
              "source_hashes": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in inputs.items()},
              "note": "Binary screening labels include current lookup/access failures and version-distance signals. N means no qualifying signal. These labels do not establish historical deletion or breaking-change incompatibility."}
    result["source_backed_declaration_reviews"] = len(resolutions)
    if args.resolutions_csv:
        result["resolutions_sha256"] = hashlib.sha256(args.resolutions_csv.read_bytes()).hexdigest()
    result["model_labels_sha256"] = hashlib.sha256((args.output_dir / "model_labels.csv").read_bytes()).hexdigest()
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
