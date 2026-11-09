# Base-model and dataset pointer checks

Checks current anonymous Hub API metadata for references declared in model tags.
Accepts the filtered metadata JSON or the team's combined dataset ZIP directly.
Only the metadata member is read from the ZIP. Requirements files, model weights,
and commit histories are not needed for these checks.

From the repository root, first extract a reproducible pilot sample offline:

```bash
python3 pointer-checking/check_pointers.py \
  --input /path/to/dependency-drift_dataset_v2_2026-11-03_111878.zip \
  --output-dir pointer-checking/output/pilot --sample-size 50 --dry-run
```

Remove `--dry-run` to make the API requests. The same sample size and default seed
42 select the same models. This is a uniform pilot sample, not the final stratified
study sample. Omit `--sample-size` only when ready to process the full input set.
No additional filtering criteria are applied by this script.

For the team's Pass 1 sample, pass the exported tab instead of drawing a new
sample:

```bash
python3 pointer-checking/check_pointers.py \
  --input /path/to/dependency-drift_dataset_v2_2026-11-03_111878.zip \
  --sample-csv /path/to/pass1-samples.csv \
  --output-dir pointer-checking/output/pass1
```

The CSV must have a `Model ID` column. Rows keep the sheet's order; duplicate,
blank, and unmatched IDs stop the run. The input sheet's review and label
columns are not read or changed.

## Outputs

- `manifest.json`: input path, selected model IDs, sampling settings and anonymous access mode.
- `references.json`: extracted references and original tags (offline mode).
- `checks.jsonl`: flushed progress log, one record per distinct typed dependency lookup.
- `pointers.csv`: one row per model/dependency, retaining original tags and lookup evidence.
- `summary.json`: counts by pointer status and model/type pairs without declared references.

The output directory is bound to an input and sample. Reuse it to resume; terminal
results are reused with their original timestamps, while temporary errors are retried.
Shared dependencies are checked once per invocation. `--refresh` rechecks every
dependency and appends new observations to the journal. Use a new output directory
for a separate observation date; results otherwise can mix check dates after resume.

## Interpretation

| Status/field | Interpretation |
| --- | --- |
| `metadata_accessible` | Repository metadata returned successfully; file access and usability were not tested. |
| `gated` | Metadata reports gating; this does not test whether any user can obtain access. |
| `private` | Metadata explicitly reports a private repository. |
| `access_restricted` | HTTP 401/403; no claim that the dependency was deleted. |
| `not_found_or_private` | HTTP 404; deletion and hidden/private repositories cannot be distinguished. |
| `invalid_reference` | Cannot interpret the tag as a Hub repository ID; manual review needed. |
| `request_error`, `rate_limited`, `server_error`, `unexpected_response` | Inconclusive observation; retry later. |
| `id_changed` | Response reports a different canonical ID, ignoring case; possible move/rename evidence. |

Legacy one-component dataset IDs are sent unchanged to the dataset endpoint.
Plain and relation-qualified base-model tags are merged without discarding the
original tag evidence. Unknown tag formats are retained for review rather than
silently corrected. Both model and dataset IDs are queried only on huggingface.co.

These observations identify candidates for review, not final drift labels. A current
404 does not establish that a reference was once valid. Successful metadata does
not establish that weights or dataset files work. Detecting silent replacements,
reference changes across commits, or actual library incompatibility requires later
historical/version checks. The v2 archive supplies root config contents and
extracted library versions, but those are not evaluated by this checker. The
commit records do not include file diffs or historical file contents.

Manual validation: inspect 10 sampled models, compare the exact declared reference
with the returned canonical ID/status and linked repository, and record reviewer
judgment separately. Resolve ambiguous cases with the team before counting drift.

For supplementary page-level evidence, run:

```bash
python3 pointer-checking/verify_pages.py --pilot-dir pointer-checking/output/pilot
```

This fetches repository page responses for 10 models, prioritizing ambiguous or
changed references. It saves HTTP statuses, redirects, canonical URLs, and page
titles to `page_verification.json`. This cross-check does not replace independent
human coding or establish inter-rater agreement.

API references: [Hub endpoints](https://huggingface.co/docs/hub/api),
[gated models](https://huggingface.co/docs/hub/models-gated).

## Tests

The 50-model pilot checked 130 references and cross-checked repository pages
for ten models. The official 260-model Pass 1 run matched all sample IDs and
checked 776 references. These runs record current pointer status, not final
drift labels. Raw evidence and review materials stay outside Git.

```bash
python3 -m unittest discover -s pointer-checking -p 'test_*.py' -v
```

Tests use mocked responses and temporary files; they do not contact the Hub.
