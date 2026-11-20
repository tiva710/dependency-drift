# Dependency assessment status

- [x] Document and implement reproducible labeling rules, evidence requirements, examples, and unresolved-case handling in [METHODS.md](pointer-checking/METHODS.md).
- [x] Run authenticated metadata checks across all 10,656 models and 34,775 pointers, including references previously returning HTTP 401. Complete read-permission checks for all 480 distinct gated references.
- [x] Collect PyPI release histories and assess requirements and config version declarations separately. Retain declaration kind, source file, release availability, yank flags, and major-version distance.
- [x] Export per-model existence, staleness, candidate drift, confirmed RQ1 drift, and evidence notes. Report counts, denominators, observation dates, and unresolved-case bounds.
- [x] Compare the earlier saved sheet snapshot for assigned rows 209–261 against the new results. Save the 53-row comparison in the local assessment output.
- [ ] Review the 11 differences in that comparison, including differing pin-verification and historical-review coverage, before changing the shared sheet. The comparison uses the saved November 15 export.
- [ ] Review unresolved references, gated-access denials, missing or nonstandard releases, and ranged/unpinned declarations. Include source links and reasons in reviewed evidence.
- [ ] Review relevant upstream breaking changes and historical base-model replacements. Resolve the 15 historical-review notes in the assigned-row audit separately from current pointer observations.
- [ ] Review the operational definitions with the team and finalize the research report. State explicitly whether each reported rate describes candidates under the selected staleness proxy or confirmed RQ1 evidence.

The November 20 automated run produced 1,410 candidate Y labels, 4,958 N labels,
and 4,288 unresolved labels across 10,656 models. The confirmed RQ1 evidence
identifies 1,212 positive models, with the remaining 9,444 unresolved. Detailed
results and the assigned-row audit are in `pointer-checking/output/rq1-2026-11-20/`.

RQ1 asks, “What fraction of sampled models reference a base model, dataset, or
library version that is no longer available, has been renamed, or has been
superseded by a breaking-change release?”
