# Dependency drift assessment

RQ1 measures the fraction of the fixed 10,656-model dataset with at least one
unavailable or renamed dependency, or a library dependency superseded by a
breaking-change release. Each model contributes at most once to the numerator.
These results describe this filtered dataset at the recorded checking dates.

## Binary screening labels

The fully binary output uses `binary-screening-v1` in `classify_binary.py`. It
reuses the saved authenticated observations and PyPI histories. All four requested
columns contain Y or N. N means no qualifying screening signal under these rules.
The `review_required (Y/N)` column records evidence limitations.

| Column | Y | N |
| --- | --- | --- |
| Base model exists | Every declared base reference returned metadata | A reference failed to resolve, was invalid, or was missing |
| Dataset exists | Every declared dataset reference returned metadata | A reference failed to resolve, was invalid, or was missing |
| Library version stale | A comparable declaration is more than two major versions behind | No declaration establishes that distance signal |
| Drift | A reference fails lookup, is invalid or missing, is renamed, or has denied/unverified gated access; or a library has a stale declaration, fully yanked pin, missing PyPI project, unsatisfied release range, or earlier reviewed positive | No qualifying signal was found |

An existence N records failed resolution in the observation. Deletion, privacy,
and an originally invalid reference can explain it. Successful gated metadata
retains existence Y. A denied or unverified read-permission check flags a current
access problem. Establishing when access changed requires historical evidence.

For a development, prerelease, local-build, or revision-suffixed pin, compare its
parseable major version with the latest stable release. Retain a provenance review
flag when the exact declaration was not verified as published. For a range or
wildcard, compare the newest stable non-yanked release allowed by that expression
with the latest stable release. An unpinned declaration permits current versions
and receives stale N because it imposes no old-version restriction.

Unparseable declarations, external URLs, options, missing comparison releases,
and differing version epochs receive stale N with a review flag. These N labels
mean the cutoff could not be established. A project absent from PyPI, or a range
matching no stable non-yanked release there, separately flags a possible
availability problem. Verify package aliases and alternative indexes during review.
Fully yanked pins also flag candidates for review.

These rules measure current availability and version-distance candidates. Report
the binary candidate fraction with this operational definition. Historical
deletion and applicable breaking changes retain separate evidence requirements
in the assessment below.

## Inputs and evidence

Use the final model CSV to select exact IDs from the v2 metadata snapshot. Read
base-model and dataset tags, preserving relation tags and original IDs. Read both
extracted library CSVs in the same archive. Retain each config or requirements
file, key or line number, package, version expression, and declaration kind.
Different versions of the same package remain separate declarations. Config
versions describe the producer's saved environment, while requirements describe
installation instructions. Their implications can differ during review.
Model notes include lookup-status counts and a bounded set of details. Large
reference lists remain complete in `pointers.csv` and `library_checks.csv` and can
be joined by model ID. An omission count in the notes identifies shortened summaries.

Run fresh Hub checks with a token validated through `whoami-v2`. Record the account
and access mode in the pointer manifest. Keep tokens in the environment or the
existing local Hub login. Use a fresh output directory when switching between
anonymous and authenticated observations. Resume uses previously recorded checks
with their timestamps. Changing the account requires a separate directory.

PyPI uses a separate unauthenticated session. Save release files, yank flags,
release timestamps, lookup URLs, and observation times in the release journal.
Select the greatest stable, non-local, non-yanked release with a downloadable
artifact as the comparison version. Retain unresolved package lookups.

For owners with several referenced repositories, an authenticated full metadata
listing can establish exact positive ID matches. Save the listing URL and the
`owner_listing` lookup method for those observations. Missing matches always fall
back to individual repository lookups, including entries beyond the first listing
page. Absence from a listing supplies no negative evidence.

## Decision rules

An empty label means unresolved. Every unresolved result needs supporting status
information in the evidence output. Model-level aggregation uses these rules.

| Column | Y | N | Unresolved |
| --- | --- | --- | --- |
| Base model exists | Every declared base model returns repository metadata, including gated/private metadata | At least one reference has reviewed evidence establishing unavailability | Missing declaration, invalid reference, failed lookup, or ambiguous 401/403/404 |
| Dataset exists | Every declared dataset returns repository metadata, including gated/private metadata | At least one reference has reviewed evidence establishing unavailability | Same uncertainty rules as base models |
| Library version stale | At least one verified exact stable pin has a major-version gap greater than 2 | Every declaration is a verified comparable pin with a gap at most 2 | A declaration needs review and none already establishes Y |
| Confirmed RQ1 drift | At least one dependency is verifiably renamed, reviewed as unavailable, or has reviewed applicable breaking-change evidence | All applicable dependency checks establish absence of RQ1 drift | At least one check is inconclusive and none establishes Y |

A successful canonical ID change, ignoring letter case, counts as renamed-reference
drift, including working redirects and legacy namespace migrations. The existence
label can therefore be Y while overall drift is also Y. Keep the returned ID and
response URL so a reviewer can verify the move.

A 401/403/404 after authentication remains ambiguous. Review the exact ID,
permissions, owner announcements, and historical evidence before assigning N to
existence. A successful response with a gated/private flag establishes existence.
Authenticated runs query the repository's read-only `auth-check` endpoint. A
successful check establishes read permission for the recorded account. A denied
or inconclusive check remains unresolved for drift classification. Access status,
HTTP code, endpoint, and timestamp are recorded separately from metadata status.
Invalid references,
external data descriptions such as `custom`, and missing declarations require
review. Rate limits and connection/server errors are retried on resume.

For library staleness, compute `max(0, latest.major - pinned.major) > 2`, using
PEP 440 parsing. For example, 1.0 to 4.0 qualifies and 1.0 to 3.0 does not. This
cutoff describes version distance. Some packages use calendar versions or do not
follow semantic versioning, so the distance needs contextual interpretation.

Ranged, unpinned, wildcard, invalid, development, prerelease, local-build, epoch
mismatch, and unverified release declarations remain unresolved for automatic
staleness decisions. A missing release in PyPI may have another index or a Git
source. Check the original declaration before claiming it disappeared. A yanked
pin retains its yank flag for review. Yanking alone does not establish deletion or
an applicable breaking change.

Overall RQ1 library drift requires evidence from upstream release notes or a
migration guide and an explanation connecting the change to the declared version
and the model's use. When a newer release exists, version distance alone leaves
that part of RQ1 unresolved. With no newer stable non-yanked release and an
available non-yanked exact stable pin, the library check is N. A reviewed finding
that newer releases introduce no relevant breaking change can also establish N.

`drift (Y/N)` follows the earlier Pass 1 candidate convention. Y means an
observed rename, reviewed unavailability or breaking change, or a verified pin
exceeding the agreed staleness cutoff. N requires complete pointer checks and
verified non-yanked pins within the cutoff. Otherwise it remains unresolved.
`rq1_confirmed_drift (Y/N)` uses the RQ1 evidence rule in the table above. Both
fields are exported so candidate counts and confirmed RQ1 counts can be reported
with explicit definitions.

## Historical checks

Current pointer lookups do not test silent replacement or changing base-model
fields over time. Review archived card declarations and commits separately. A
changed declaration records a history candidate. Establish whether it reflects
an intentional update, a corrected declaration, a rename, or an unavailable
reference before assigning an RQ1 category. Review the earlier Pass 1 sheet labels
against these definitions, particularly its stale-library and gated cases.

## Recording reviewed evidence

Supply `--evidence-csv` with columns `model_id`, `dependency_type`, `dependency_id`,
`verdict`, `source_url`, and `reason`. Dependency types are `base_model`, `dataset`,
and `library`. Copy the library `dependency_id` from `library_checks.csv` so the
review applies to the exact version and source declaration. Pointer IDs use the
original declared Hub ID. Duplicate or unmatched evidence stops assessment.

Verdicts are `available`, `unavailable`, `breaking_change`, and
`no_breaking_change`. The last two apply to libraries. Provide a source URL and an
explanation for every verdict. For pointers, `available` must document usable
access at the declared ID or a reviewed redirect. For libraries, `available`
establishes availability alone. A `no_breaking_change` verdict requires a verified
non-yanked pin as well as a review of relevant newer releases. Record reviewed
sources as evidence, and independently review a portion for coding consistency.

## RQ1 calculation

Using `rq1_confirmed_drift (Y/N)`, let D be models with confirmed Y, C be models
with N, and U be unresolved models.
Check that `D + C + U = 10656`. Report `D / 10656` as the confirmed observed
fraction and `[D / 10656, (D + U) / 10656]` as the bounds while cases remain
unresolved. The resolved-only rate `D / (D + C)` is supplementary and has its own
denominator. It can differ systematically when unresolved cases are excluded.

Report the Pass 1 candidate fraction from `drift (Y/N)` separately, using the
selected staleness proxy and its own unresolved-case bounds. Also report per-type model counts, unresolved counts, and
observation dates. Categories can overlap, so their counts need not sum to D.
These bounds describe unresolved labels. Statistical confidence intervals for a
sample require a separate analysis respecting the stratified sampling design.

API behavior is documented in the [Hub API reference](https://huggingface.co/docs/hub/api),
[Hub token documentation](https://huggingface.co/docs/hub/security-tokens), and
[PyPI JSON API reference](https://docs.pypi.org/api/json/).
