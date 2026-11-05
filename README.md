# Dependency Drift in Open-Source ML Models on Hugging Face

Measures how often Hugging Face model repositories reference a base model,
dataset, or library version that has since become invalid or stale, and what
forms that dependency drift takes.

## Progress so far (Yinan, 2026-11-05)

Everything below comes from one snapshot of the Hugging Face model list taken on **2026-11-03**.

1. **Collected metadata for every public model on the Hub**: 3,209,401 models (~1.9 GB).
2. **Filtered them down to 111,878 models** using only the must-have filters (see the table below).
3. **Collected the full commit history** of those models: 1,283,235 commits, no errors.
4. **Collected the file list and file sizes** of those models (for model size).
5. **Collected their requirements files**: only 445 of the models have one.
6. **Collected their config files** (`config.json` etc.) and the library versions recorded in them:
   72,562 models (64.9%) record at least one, mostly `transformers_version`.

All of this data is in one zip in our Discord:
**[dependency-drift_dataset_v2_2026-11-03_111878.zip](https://discord.com/channels/@me/1539731118272684204/1568006333775609958)** (191 MB).
The link opens for team members only. The data is not in git because it is too big (`output/` folders are gitignored).
See [Data files](#data-files) for which folder in the zip holds what.

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

Everything is in one zip: **[dependency-drift_dataset_v2_2026-11-03_111878.zip](https://discord.com/channels/@me/1539731118272684204/1568006333775609958)**
(Discord, team members only). Start from `metadata/`. Every file joins on the model id (`id` in JSON, `Model ID` in CSV).
The `README.txt` in the zip lists every column.

| Folder in the zip | What's in it | Format |
|---|---|---|
| `metadata/` | Listing fields and tags for the 111,878 models (task, likes, dates, license, ...) | JSON + CSV |
| `commits/` | Full commit history of each model (date, author, title, message) | JSON + CSV |
| `model-size/` | Every file path and its size in each repo | JSON |
| `requirements/` | The raw requirements files, plus one row per package and version | JSON + CSV |
| `configs/` | The raw root config files, plus one row per library version found in them | JSON + CSV |

## Pipeline

| Step | Script | Run from | Output |
|---|---|---|---|
| 1. Metadata | `data-collection/metadata-collection/crawl-metadata.py` | that folder | `output/<date>_all_huggingface_models_<n>.json/.csv` |
| 2. Filter | `model-filtering/filter-models.py` | anywhere | `model-filtering/output/<date>_filtered_models_<n>.json/.csv` |
| 3. Commits | `data-collection/model-commits-collection/crawl-commits.py` | anywhere | `output/<date>_model_commits_<n>.json/.csv` |
| 4. Model size | `data-collection/model-size-collection/crawl-model-size.py` | repo root | `output/model_size_tree_final.json` |
| 5. Requirements | `data-collection/requirements-collection/crawl-requirements.py` | anywhere | `output/requirements_<n>.json/.csv` |
| 6. Configs | `data-collection/model-config-collection/crawl-configs.py` | anywhere | `output/model_configs_<n>.json`, `output/model_config_versions_<n>.csv` |

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
