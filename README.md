# Dependency Drift in Open-Source ML Models on Hugging Face

Measures how often Hugging Face model repositories reference a base model,
dataset, or library version that has since become invalid or stale, and what
forms that dependency drift takes.

## Initial FIltering (Yinan)

Everything below comes from one snapshot of the Hugging Face model list taken on **2026-11-03**.

1. Collected metadata for every public model on the Hub and filtered down (see table below).
3. Collected full commit history, file list,  and file sizes of those models.
5. Collected requirements.txt files: only 445 of the models have one.
6. Collected their config files and the library versions recorded in them

## How the models were filtered

| # | Filter | What it means | Why | Models left |
|---|---|---|---|---:|
| 0 | All models | Every unique public model in the listing | Starting point | 3,209,374 |
| 1 | Has base model | The model says which model it was built from (`base_model:` tag) | The base model is the main dependency we track | 921,478 |
| 2 | Has dataset | The model says which dataset it used (`dataset:` tag) | The dataset is another dependency we track | 143,307 |
| 3 | Not gated | No "accept the terms first" wall | Gated repos block reading files and commits | 140,958 |
| 4 | At least 6 months old | Created on or before 2026-05-03 | Gives its dependencies time to drift | **111,878** |


## Final dataset: 10,656 models (Tiva)

`model-filtering/finalize-dataset.py` applies the remaining filters to the zip from Yinan's intiial filtering:

| Filter | Models left |
|---|---:|
| Initial set | 111,878 |
| Likes >= 1 | 22,617 |
| Commits >= 2 | 22,331 |
| Has a task (`pipeline_tag`) | 14,395 |
| Records a library version (in config or reqs) | **10,656** |

License is is kept as a column with an `is_permissive` flag.

### Pass 1 sample: 260 models

50 random models per task bucket, with the 10 in the "Other" category.


### Pass 2 subsample: 50 models

Once Pass 1 is labeled, run `model-filtering/sample-pass2.py`. 

It draws 50 models with `drift = yes`,
and adds the needed columns for categorical analysis.

Outputs (in our google drive):

| File | What it is |
|---|---|
| `final_models_10656.csv` | One row per final model |
| `final_library_versions_10656.csv` | One row per (model, file, package, version) and whether they're pinned, ranged, or unpinned |
| `pass1_sample_260.csv` | Pass 1 sheet |

## Pipeline

| Step | Script | Run from | Output |
|---|---|---|---|
| 1. Metadata | `data-collection/metadata-collection/crawl-metadata.py` | that folder | `output/<date>_all_huggingface_models_<n>.json/.csv` |
| 2. Filter | `model-filtering/filter-models.py` | anywhere | `model-filtering/output/<date>_filtered_models_<n>.json/.csv` |
| 3. Commits | `data-collection/model-commits-collection/crawl-commits.py` | anywhere | `output/<date>_model_commits_<n>.json/.csv` |
| 4. Model size | `data-collection/model-size-collection/crawl-model-size.py` | repo root | `output/model_size_tree_final.json` |
| 5. Requirements | `data-collection/requirements-collection/crawl-requirements.py` | anywhere | `output/requirements_<n>.json/.csv` |
| 6. Configs | `data-collection/model-config-collection/crawl-configs.py` | anywhere | `output/model_configs_<n>.json`, `output/model_config_versions_<n>.csv` |
| 7. Final filter + Pass 1 sample | `model-filtering/finalize-dataset.py` | anywhere | `model-filtering/output/final_*.csv`, `pass1_sample_260.csv` |
| 8. Pass 2 subsample | `model-filtering/sample-pass2.py` | anywhere | `model-filtering/output/pass2_coding_<n>.csv` |

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
