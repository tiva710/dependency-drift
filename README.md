# Dependency Drift in Open-Source ML Models on Hugging Face

Measures how often Hugging Face model repositories reference a base model,
dataset, or library version that has since become invalid or stale, and what
forms that dependency drift takes.

## Progress so far (Yinan, 2026-11-05)

Everything below comes from one snapshot of the Hugging Face model list taken on **2026-11-03**.

1. **Collected metadata for every public model on the Hub**: 3,209,401 models (~1.9 GB).
2. **Filtered them down to 111,878 models** using only the must-have filters (see the table below).
   - Discord link: [metadata](https://discord.com/channels/@me/1539731118272684204/1567914782378434600)
3. **Collected the full commit history** of those models: 1,283,235 commits, no errors.
   - Discord link: [commits](https://discord.com/channels/@me/1539731118272684204/1567915082254655570)
4. **Collected the file list and file sizes** of those models (for model size).
   - Discord link: [model size](https://discord.com/channels/@me/1539731118272684204/1567915287314047208)
5. **Collected their requirements files**: only 445 of the models have one.
   - Discord link: [requirements](https://discord.com/channels/@me/1539731118272684204/1567915562909311026)

The data files are not in git because they are too big (`output/` folders are gitignored).
They are posted in our Discord, so the links open for team members only. More details are in [Data files](#data-files).

## How the 111,878 models were picked

Some filters may still change, and we don't want to re-crawl every time one does.
So we only applied the filters we all agree on and collected everything for those models once.

| # | Filter | What it means | Why | Models left |
|---|---|---|---|---:|
| 0 | All models | Every public model in the listing (duplicates removed) | Starting point | 3,209,374 |
| 1 | Has base model | The model says which model it was built from (`base_model:` tag) | The base model is the main dependency we track | 921,478 |
| 2 | Has dataset | The model says which dataset it used (`dataset:` tag) | The dataset is another dependency we track | 143,307 |
| 3 | Not gated | No "accept the terms first" wall | Gated repos block reading files and commits | 140,958 |
| 4 | At least 6 months old | Created on or before 2026-05-03 | Gives its dependencies time to drift | **111,878** |

Everything else is kept as a column to filter on later: task, likes, `lastModified` > `createdAt`,
number of commits, model size, license and whether the repo has a `requirements.txt`.
When everyone agrees on the final criteria, we filter the tables to get the final dataset. No re-crawling needed.

## Data files

Posted in our Discord, so the links open for team members only:

| File | What's in it | Format | Link |
|---|---|---|---|
| Metadata | Listing fields and tags for the 111,878 models (task, likes, dates, license, ...) | JSON + CSV | [Discord](https://discord.com/channels/@me/1539731118272684204/1567914782378434600) |
| Commits | Full commit history of each model (date, author, title, message) | JSON + CSV | [Discord](https://discord.com/channels/@me/1539731118272684204/1567915082254655570) |
| Model size | Every file path and its size in each repo | JSON | [Discord](https://discord.com/channels/@me/1539731118272684204/1567915287314047208) |
| Requirements | The raw requirements files, plus one row per package and version | JSON + CSV | [Discord](https://discord.com/channels/@me/1539731118272684204/1567915562909311026) |

## Pipeline

| Step | Script | Run from | Output |
|---|---|---|---|
| 1. Metadata | `data-collection/metadata-collection/crawl-metadata.py` | that folder | `output/<date>_all_huggingface_models_<n>.json/.csv` |
| 2. Filter | `model-filtering/filter-models.py` | anywhere | `model-filtering/output/<date>_filtered_models_<n>.json/.csv` |
| 3. Commits | `data-collection/model-commits-collection/crawl-commits.py` | anywhere | `output/<date>_model_commits_<n>.json/.csv` |
| 4. Model size | `data-collection/model-size-collection/crawl-model-size.py` | repo root | `output/model_size_tree_final.json` |
| 5. Requirements | `data-collection/requirements-collection/crawl-requirements.py` | anywhere | `output/requirements_<n>.json/.csv` |

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
