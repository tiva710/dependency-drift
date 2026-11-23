# Dependency assessment status

- [x] Complete evidence review of all 4,996 previously flagged models. Save 150 declaration-level source-backed resolutions and correct registry aliases, native/source-distributed components, and parser false positives.
- [x] Validate reviewed results across all 10,656 models. Current totals are 4,671 candidate Y and 5,985 N, with 224 stale-version Y labels and zero open binary-classification actions. Results are in `pointer-checking/output/reviewed-2026-11-21/`.
- [x] Produce fully binary screening labels for all 10,656 models under `binary-screening-v1`, with reasons and evidence review flags. Results are in `pointer-checking/output/binary-2026-11-21/`.
- [x] Verify all 42,624 requested cells contain Y/N. Binary candidate totals are 4,677 Y and 5,979 N, using the November 20 observations. The assigned 53-model export contains 19 Y and 34 N.
- [x] Document and implement reproducible labeling rules, evidence requirements, examples, and unresolved-case handling in [METHODS.md](pointer-checking/METHODS.md).
- [x] Run authenticated metadata checks across all 10,656 models and 34,775 pointers, including references previously returning HTTP 401. Complete read-permission checks for all 480 distinct gated references.
- [x] Collect PyPI release histories and assess requirements and config version declarations separately. Retain declaration kind, source file, release availability, yank flags, and major-version distance.
- [x] Export per-model existence, staleness, candidate drift, confirmed RQ1 drift, and evidence notes. Report counts, denominators, observation dates, and unresolved-case bounds.
- [x] Compare the earlier saved sheet snapshot for assigned rows 209–261 against the new results. Save the 53-row comparison in the local assessment output.
- [ ] Review the 11 differences in that comparison, including differing pin-verification and historical-review coverage, before changing the shared sheet. The comparison uses the saved November 15 export.
- [x] Resolve the screening review queue using saved authenticated outcomes, version-distance calculations, upstream registry/source checks, and archived file contents. Preserve temporal-deletion and compatibility limitations in the evidence.
- [ ] Review relevant upstream breaking changes and historical base-model replacements. Resolve the 15 historical-review notes in the assigned-row audit separately from current pointer observations.
- [ ] Review the operational definitions with the team and finalize the research report. State explicitly whether each reported rate describes candidates under the selected staleness proxy or confirmed RQ1 evidence.

The November 20 automated run produced 1,410 candidate Y labels, 4,958 N labels,
and 4,288 unresolved labels across 10,656 models. The confirmed RQ1 evidence
identifies 1,212 positive models, with the remaining 9,444 unresolved. Detailed
results and the assigned-row audit are in `pointer-checking/output/rq1-2026-11-20/`.

RQ1 asks, “What fraction of sampled models reference a base model, dataset, or
library version that is no longer available, has been renamed, or has been
superseded by a breaking-change release?”
