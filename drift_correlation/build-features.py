import os
import argparse
import numpy as np
import pandas as pd

# Pass 3, Step 2a: build one row per model with every characteristic we test against drift.
# Inputs: the Step 1 labels, two tabs of the final dataset, and the commit history from the team zip.
# No Hugging Face crawling.

SNAPSHOT = pd.Timestamp('2026-11-03', tz='UTC')   # date the team's data was collected
BIG_ORG_TOP_N = 20                                 # "big org" = one of the 20 most-used base-model owners

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_FILE = os.path.join(SCRIPT_DIR, 'pass3_model_labels.csv')
OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'pass3_features.csv')
COMMITS_FILE = os.path.join('commits', '2026-11-03_model_commits_111878.csv')


def split_ids(cell):
    # base_models / datasets cells hold ';'-separated ids
    if pd.isna(cell):
        return []
    return [part.strip() for part in str(cell).split(';') if part.strip()]


def owner(repo_id):
    return repo_id.split('/')[0].lower() if '/' in repo_id else None


def load_inputs(data_path):
    labels = pd.read_csv(LABELS_FILE)
    models = pd.read_excel(data_path, sheet_name='final_models_10656')
    libraries = pd.read_excel(data_path, sheet_name='final_library_versions_10656')
    return labels, models, libraries


def commit_features(dataset_dir, model_ids):
    path = os.path.join(dataset_dir, COMMITS_FILE)
    commits = pd.read_csv(path, usecols=['Model ID', 'commit_count', 'date'])
    commits = commits[commits['Model ID'].isin(model_ids)]
    commits['date'] = pd.to_datetime(commits['date'], utc=True)
    per_model = commits.groupby('Model ID').agg(n_commits=('commit_count', 'first'), last_commit=('date', 'max'))
    per_model['months_since_last_commit'] = (SNAPSHOT - per_model['last_commit']).dt.days / 30.44
    return per_model[['n_commits', 'months_since_last_commit']]


def version_source(libraries):
    # 1 if any library version came from a requirements.txt, 0 if only from config files
    from_reqs = libraries['file'].astype(str).str.contains('requirements', case=False)
    return from_reqs.groupby(libraries['Model ID']).any().astype(int).rename('has_requirements_file')


def ownership_features(models):
    base_owners = models['base_models'].map(lambda cell: [owner(b) for b in split_ids(cell) if owner(b)])
    data_owners = models['datasets'].map(lambda cell: [owner(d) for d in split_ids(cell) if owner(d)])
    model_owner = models['Model ID'].map(owner)

    # author also owns one of its own base models or datasets
    owns = [int(o in set(b) | set(d)) for o, b, d in zip(model_owner, base_owners, data_owners)]

    # base model comes from one of the most-used base-model owners in our data (meta-llama, Qwen, google, ...)
    counts = pd.Series([o for owners in base_owners for o in set(owners)]).value_counts()
    big_orgs = set(counts.head(BIG_ORG_TOP_N).index)
    from_big = [int(bool(set(b) & big_orgs)) for b in base_owners]

    print(f'Big orgs (top {BIG_ORG_TOP_N} base-model owners): {", ".join(counts.head(BIG_ORG_TOP_N).index)}')
    return pd.DataFrame({'Model ID': models['Model ID'],
                         'author_owns_dependency': owns,
                         'base_from_big_org': from_big})


def build(labels, models, libraries, commits):
    features = labels[['Model ID', 'drift', 'dataset_drift', 'base_drift', 'library_stale', 'drift_evidence',
                       'n_base_models', 'n_datasets', 'n_pinned_libs']].copy()
    for column in ['drift', 'dataset_drift', 'base_drift', 'library_stale']:
        features[column] = (features[column] == 'Y').astype(int)

    first_commit = pd.to_datetime(models['first_commit_date'], utc=True)
    characteristics = pd.DataFrame({
        'Model ID': models['Model ID'],
        'task_bucket': models['task_bucket'],
        'author_type': models['author_type'],
        'downloads_bucket': models['downloads_bucket'],
        'upload_age': models['upload_age'],
        'likes': models['likes'],
        'is_permissive': models['is_permissive'].astype(int),
        'age_months': (SNAPSHOT - first_commit).dt.days / 30.44,
    })

    features = (features.merge(characteristics, on='Model ID', how='left')
                        .merge(version_source(libraries), left_on='Model ID', right_index=True, how='left')
                        .merge(ownership_features(models), on='Model ID', how='left')
                        .merge(commits, left_on='Model ID', right_index=True, how='left'))
    features['has_requirements_file'] = features['has_requirements_file'].fillna(0).astype(int)

    # flag models whose drift comes from the removed Common Voice datasets (used for the re-run without them)
    evidence = features['drift_evidence'].fillna('').str.lower()
    features['common_voice_drift'] = evidence.str.contains('common_voice').astype(int)
    return features.drop(columns='drift_evidence')


def report(features):
    print(f'{len(features)} models, {features.isna().any(axis=1).sum()} with a missing value')
    print(f'drift = Y: {features["drift"].sum()}  dataset_drift = Y: {features["dataset_drift"].sum()}  '
          f'Common Voice drift: {features["common_voice_drift"].sum()}')
    print(features.describe().T[['mean', '50%', 'min', 'max']].round(2).to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build the per-model feature table for the drift correlation analysis')
    parser.add_argument('--data', required=True, help='Final Dataset.xlsx')
    parser.add_argument('--dataset-dir', required=True, help='unzipped dependency-drift_dataset_v2_2026-11-03_111878 folder')
    args = parser.parse_args()

    labels, models, libraries = load_inputs(args.data)
    commits = commit_features(args.dataset_dir, set(labels['Model ID']))
    features = build(labels, models, libraries, commits)
    report(features)
    features.to_csv(OUTPUT_FILE, index=False)
    print(f'Saved {OUTPUT_FILE}')
