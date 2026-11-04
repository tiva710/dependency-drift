# Dependency Drift in Open-Source ML Models on Hugging Face

Measures how often Hugging Face model repositories reference a base model,
dataset, or library version that has since become invalid or stale, and what
forms that dependency drift takes.

## Progress so far (Yinan, 2026-11-03)

Data collection and the first round of filtering are done. Everything below
comes from one snapshot of the Hugging Face model list taken on **2026-11-03**.

1. **Collected metadata for every public model on the Hub**: 3,209,401 models (~1.9 GB).
2. **Filtered them down to 8,106 models** (see the table below).
3. **Collected the full commit history** of the filtered models: 129,053 commits, no errors.
4. **Collected the file list and file sizes** of the filtered models (for model size), no errors.

The data files are not in git because they are too big (`output/` folders are gitignored).

## How the 8,106 models were picked

We start from every model and keep only the ones that pass each check, in this order.
The criteria come from the project notes and the proposal slides.

| # | Filter | What it means | Why | Models left |
|---|---|---|---|---:|
| 0 | All models | Every public model in the listing (27 duplicates removed) | Starting point | 3,209,374 |
| 1 | Has task | The model says what it does (`pipeline_tag`, e.g. text-generation) | We group models by task | 1,116,003 |
| 2 | Has base model | The model says which model it was built from (`base_model:` tag) | The base model is the main dependency we track | 497,144 |
| 3 | Has dataset | The model says which dataset it used (`dataset:` tag) | The dataset is another dependency we track | 103,906 |
| 4 | At least 6 months old | Created on or before 2026-05-03 | Gives its dependencies time to drift | 79,174 |
| 5 | Likes ≥ 2 | At least 2 likes | Drops throwaway and test repos | 8,454 |
| 6 | Updated after upload | `lastModified` is later than `createdAt` | We only want repos that changed | 8,432 |
| 7 | Not gated | No "accept the terms first" wall | Gated repos block reading files and commits | 8,199 |
| 8 | Commits ≥ 2 | At least 2 commits in its history | We need a history to compare versions | **8,106** |

Filters 1–7 only need the metadata. Filter 8 needs the commit history, so it runs after the commits are collected.

Not applied yet (on hold): has `requirements.txt` (only 63 of the 8,199 models have one),
permissive license, and minimum model size (no threshold decided).

## Pipeline

| Step | Script | Run from | Output |
|---|---|---|---|
| 1. Metadata | `data-collection/metadata-collection/crawl-metadata.py` | that folder | `output/<date>_all_huggingface_models_<n>.json/.csv` |
| 2. Filter 1–7 | `model-filtering/filter-models.py` | anywhere | `model-filtering/output/<date>_filtered_models_<n>.json/.csv` |
| 3. Commits | `data-collection/model-commits-collection/crawl-commits.py` | anywhere | `output/<date>_model_commits_<n>.json/.csv` |
| 4. Filter 8 | `model-filtering/filter-models.py --commits <commits json>` | anywhere | `model-filtering/output/<date>_filtered_models_<n>.json/.csv` |
| 5. Model size | `data-collection/model-size-collection/crawl-model-size.py` | repo root | `output/model_size_tree_final.json` |

## Setup

```bash
pip install -r requirements.txt
```

Each script reads a `config.yaml` in its own folder (gitignored, never commit it):

```yaml
huggingface_keys:
- <your-hf-key>
- <another-hf-key>   # optional, one thread per key
```
