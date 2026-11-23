# Base-model and dataset pointer checks

Checks current Hub API metadata for references declared in model tags.
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

- `manifest.json`: input path, selected model IDs, sampling settings, access mode and authenticated account when applicable.
- `references.json`: extracted references and original tags (offline mode).
- `checks.jsonl`: flushed progress log, one record per distinct typed dependency lookup.
- `pointers.csv`: one row per model/dependency, retaining original tags and lookup evidence.
- `summary.json`: counts by pointer status and model/type pairs without declared references.

The output directory is bound to an input and sample. Reuse it to resume; terminal
results are reused with their original timestamps, while temporary errors are retried.
Shared dependencies are checked once per invocation. `--refresh` rechecks every
dependency and appends new observations to the journal. Use a new output directory
for a separate observation date; results otherwise can mix check dates after resume.

On HTTP 429, the checker waits for the Hub's reset time before retrying the
same reference. If throttling persists, it saves progress and stops. Wait for
the quota to reset and rerun the same command to resume.

## Interpretation

| Status/field | Interpretation |
| --- | --- |
| `metadata_accessible` | Repository metadata returned successfully; file access and usability were not tested. |
| `gated` | Metadata reports gating; authenticated runs record a separate read-permission check. |
| `private` | Metadata explicitly reports a private repository. |
| `access_restricted` | HTTP 401/403; no claim that the dependency was deleted. |
| `not_found_or_private` | HTTP 404; deletion and hidden/private repositories cannot be distinguished. |
| `invalid_reference` | Cannot interpret the tag as a Hub repository ID; manual review needed. |
| `request_error`, `rate_limited`, `server_error`, `unexpected_response` | Inconclusive observation; retry later. |
| `id_changed` | Response reports a different canonical ID, ignoring case; possible move/rename evidence. |
| `access_status` | For gated/private references, `accessible` means the account passed the read-permission check. `denied` or an error requires review. |

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

## Authenticated full-dataset assessment

The final CSV selects the 10,656 study models from the larger archive. Supply
`HF_TOKEN` through the environment or use an existing local Hub login. The
`--authenticated` option validates the token before checking references and checks
read permissions for known gated/private dependencies.

```bash
python3 pointer-checking/check_pointers.py \
  --input /path/to/dependency-drift_dataset_v2_2026-11-03_111878.zip \
  --sample-csv /path/to/final_models_10656.csv \
  --output-dir pointer-checking/output/authenticated-study \
  --authenticated --workers 4 --prefetch-owners

python3 pointer-checking/assess_drift.py \
  --input /path/to/dependency-drift_dataset_v2_2026-11-03_111878.zip \
  --model-csv /path/to/final_models_10656.csv \
  --pointer-dir pointer-checking/output/authenticated-study \
  --output-dir pointer-checking/output/rq1-study \
  --expected-models 10656
```

Use the same commands and directories to resume. Concurrent checks share cached
references and save completed batches to the journal. Hub authentication can still
return access errors for resources outside the account's permissions. The
assessment requires a completed pointer output matching the archive and model IDs.

The assessment saves `library_checks.csv`, `model_labels.csv`, `releases.jsonl`,
`manifest.json`, and `summary.json`. It reads requirements and config declarations
separately and retains both when versions differ. `--offline` uses previously
collected PyPI observations. `--evidence-csv /path/to/reviews.csv` applies reviewed
dependency findings with source links.

The model output includes existence labels, library staleness, Pass 1 candidate drift,
confirmed RQ1 drift, per-type drift labels, and notes. Empty labels require review.
The summary reports confirmed counts, unresolved cases, per-type counts, and
the resulting fraction bounds across the full model denominator.

See [assessment methods](METHODS.md) for the decision table, review evidence format,
staleness cutoff, and interpretation of the RQ1 and candidate labels. A major-version
gap greater than two triggers a stale candidate. Assigning library breaking-change
drift requires a reviewed source explaining how the change affects the model.

Release metadata is retrieved from the [PyPI JSON API](https://docs.pypi.org/api/json/).

## Binary classification

After assessment, classify every model into Y/N values using the saved
authenticated observations and release histories.

```bash
python3 pointer-checking/classify_binary.py \
  --assessment-dir pointer-checking/output/rq1-study \
  --output-dir pointer-checking/output/binary-study
```

Outputs include `model_labels.csv`, detailed `library_checks.csv`, and
`summary.json`. The four requested columns are fully binary. A review flag and
notes retain evidence limitations. N means no qualifying screening signal.
Failed lookups, blocked gated access, renames, and library availability or staleness
signals produce candidate Y labels according to the [binary rules](METHODS.md#binary-screening-labels).
This step uses the source observation dates and checks source hashes before reading
the saved evidence.

To replay the completed exceptional-case reviews, supply the provided resolutions
file. Resolutions match exact model IDs and version declarations and retain source
links, reasons, and corrected comparison values.

```bash
python3 pointer-checking/classify_binary.py \
  --assessment-dir pointer-checking/output/rq1-2026-11-20 \
  --resolutions-csv pointer-checking/output/review-sources-2026-11-21/library_review_resolutions.csv \
  --output-dir pointer-checking/output/reviewed-2026-11-21
```

Completed HTTP observations and numeric comparisons retain their evidence
limitations in the status fields and notes. `review_required` indicates an action
still needed to finish classification. The reviewed report contains no open
classification actions under the binary screening rules.

## Tests

The 50-model pilot checked 130 references and cross-checked repository pages
for ten models. The official 260-model Pass 1 run matched all sample IDs and
checked 776 references. These runs record current pointer status, not final
drift labels. Raw evidence and review materials stay outside Git.

```bash
python3 -m unittest discover -s pointer-checking -p 'test_*.py' -v
```

Tests use mocked responses and temporary files; they do not contact the Hub.
