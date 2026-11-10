import os
import time
import argparse
import requests
import pandas as pd

# Filters the 111,878 models into the final dataset and the Pass 1 sample. 
#   Remaining filters: likes >= 1, commits >= 2, has a task (pipeline_tag), records a library version
SNAPSHOT = '2026-11-03'
MIN_LIKES = 1
MIN_COMMITS = 2
PER_BUCKET = 50  # Pass1 sample 
SEED = 42

# tasks from Yinan's earlier script 
TASK_BUCKETS = {
    'Multimodal': ['audio-text-to-text', 'image-text-to-text', 'image-text-to-image', 'image-text-to-video', 'visual-question-answering', 'document-question-answering', 'video-text-to-text', 'visual-document-retrieval', 'any-to-any'],
    'Computer Vision': ['depth-estimation', 'image-classification', 'object-detection', 'image-segmentation', 'text-to-image', 'image-to-text', 'image-to-image', 'image-to-video', 'unconditional-image-generation', 'video-classification', 'text-to-video', 'zero-shot-image-classification', 'mask-generation', 'zero-shot-object-detection', 'text-to-3d', 'image-to-3d', 'image-feature-extraction', 'keypoint-detection', 'video-to-video'],
    'NLP': ['text-classification', 'token-classification', 'table-question-answering', 'question-answering', 'zero-shot-classification', 'translation', 'summarization', 'feature-extraction', 'text-generation', 'fill-mask', 'sentence-similarity', 'text-ranking'],
    'Audio': ['text-to-speech', 'text-to-audio', 'automatic-speech-recognition', 'audio-to-audio', 'audio-classification', 'voice-activity-detection'],
    'Reinforcement Learning': ['reinforcement-learning', 'robotics'],
}
TASK_TO_BUCKET = {task: bucket for bucket, tasks in TASK_BUCKETS.items() for task in tasks}

# Bins are upper-inclusive: 9 downloads is '0-9', 10 is '10-99'
DOWNLOAD_BINS = [-1, 9, 99, 999, 9999, float('inf')]
DOWNLOAD_LABELS = ['0-9dl', '10-99dl', '100-999dl', '1000-9999dl', '10k+dl']

# Age in whole months at SNAPSHOT: 6 months and 20 days is '6-12'
AGE_BINS = [5, 12, 24, 36, float('inf')]
AGE_LABELS = ['6-12mo', '13-24mo', '25-36mo', '37+mo']

PERMISSIVE_LICENSES = {
    'apache-2.0', 'mit', 'bsd', 'bsd-2-clause', 'bsd-3-clause', 'bsd-3-clause-clear', 'isc', 'unlicense',
    'cc0-1.0', 'cc-by-2.0', 'cc-by-2.5', 'cc-by-3.0', 'cc-by-4.0', 'cc-by-sa-3.0', 'cc-by-sa-4.0',
    'artistic-2.0', 'bsl-1.0', 'zlib', 'mpl-2.0', 'afl-3.0', 'ecl-2.0', 'wtfpl', 'postgresql', 'odc-by',
    'pddl', 'odbl', 'lgpl-2.1', 'lgpl-3.0', 'gpl-2.0', 'gpl-3.0', 'agpl-3.0', 'epl-2.0', 'openrail',
}

# added download buckets and upload age cols
MODEL_COLUMNS = ['Model ID', 'hf_url', 'task_bucket', 'likes', 'first_commit_date', 'license', 'is_permissive', 'base_models', 'datasets', 'library_versions', 'upload_age', 'downloads_bucket']

# Empty cols so we can fill it in
PASS1_COLUMNS = ['base_model_exists', 'dataset_exists', 'library_version_stale', 'drift(yes/no)', 'checked_by', 'notes']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')

def load_commits(data_dir):
    commits = pd.read_csv(os.path.join(data_dir, 'commits', f'{SNAPSHOT}_model_commits_111878.csv'), usecols=['Model ID', 'commit_count', 'commit_index', 'date'])
    commits = commits.sort_values(['Model ID', 'commit_index'])
    return commits.groupby('Model ID').agg(commit_count=('commit_count', 'first'), first_commit_date=('date', 'first'))


def load_library_versions(data_dir):
    configs = pd.read_csv(os.path.join(data_dir, 'configs', 'model_config_versions_87969.csv'))
    configs = configs.rename(columns={'version': 'version_spec'})
    configs['kind'] = 'pinned'  

    reqs = pd.read_csv(os.path.join(data_dir, 'requirements', 'requirements_445.csv'))
    reqs = reqs[reqs['kind'].isin(['pinned', 'ranged', 'unpinned'])]
    reqs = reqs.rename(columns={'specifier': 'version_spec'})

    versions = pd.concat([configs, reqs], ignore_index=True)
    versions['version_spec'] = versions['version_spec'].fillna('').str.lstrip('=')
    return versions[['Model ID', 'file', 'package', 'version_spec', 'kind']].drop_duplicates()


def author_type(author):
    for kind, path in [('organization', 'organizations'), ('individual', 'users')]:
        for attempt in range(10):
            response = requests.get(f'https://huggingface.co/api/{path}/{author}/overview', timeout=30)
            if response.status_code != 429:  
                break
            time.sleep(30)
        if response.status_code == 200:
            return kind
    return 'unknown'


def months_at_snapshot(created_at):
    created = pd.to_datetime(created_at, utc=True)
    snapshot = pd.Timestamp(SNAPSHOT, tz='UTC')
    months = (snapshot.year - created.dt.year) * 12 + (snapshot.month - created.dt.month)
    # don't count the current month until it has fully elapsed
    return months - (created.dt.day > snapshot.day).astype(int)


def finalize(data_dir):
    models = pd.read_csv(os.path.join(data_dir, 'metadata', f'{SNAPSHOT}_filtered_models_111878.csv'))
    models = models.join(load_commits(data_dir), on='Model ID')
    versions = load_library_versions(data_dir)

    filters = [
        (f'likes >= {MIN_LIKES}', lambda m: m['likes'] >= MIN_LIKES),
        (f'commits >= {MIN_COMMITS}', lambda m: m['commit_count'] >= MIN_COMMITS),
        ('has task', lambda m: m['pipeline_tag'].notna()),
        ('records a library version', lambda m: m['Model ID'].isin(versions['Model ID'])),
    ]
    for name, check in filters:
        models = models[check(models)]
        print(f"{name:>30}: {len(models):,}")

    models = models.copy()
    models['upload_age'] = pd.cut(months_at_snapshot(models['createdAt']), bins=AGE_BINS, labels=AGE_LABELS)
    models['task_bucket'] = models['pipeline_tag'].map(TASK_TO_BUCKET).fillna('Other')
    models['is_permissive'] = models['license'].isin(PERMISSIVE_LICENSES)
    models['downloads_bucket'] = pd.cut(models['downloads'], bins=DOWNLOAD_BINS, labels=DOWNLOAD_LABELS)

    versions = versions[versions['Model ID'].isin(models['Model ID'])]
    summary = (versions.assign(v=versions['package'] + ' ' + versions['version_spec']).groupby('Model ID')['v'].apply(lambda v: '; '.join(dict.fromkeys(v))))
    models['library_versions'] = models['Model ID'].map(summary)
    models['hf_url'] = 'https://huggingface.co/' + models['Model ID']
    return models[MODEL_COLUMNS], versions


def pass1_sample(models):
    # shuffle, then take 1st 50 models of each bucket
    sample = models.sample(frac=1, random_state=SEED).groupby('task_bucket').head(PER_BUCKET)
    sample = sample.sort_values(['task_bucket', 'Model ID'])
    authors = sample['Model ID'].str.split('/').str[0]
    print('Looking up author type for', authors.nunique(), 'authors')
    types = {a: author_type(a) for a in authors.unique()}
    sample.insert(3, 'author_type', authors.map(types))
    for column in PASS1_COLUMNS:
        sample[column] = ''
    return sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply the remaining filters & draw the Pass 1 sample')
    parser.add_argument('--data', required=True)
    args = parser.parse_args()

    models, versions = finalize(args.data)
    print('\nTask buckets:\n' + models['task_bucket'].value_counts().to_string())

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n = len(models)
    models.to_csv(os.path.join(OUTPUT_DIR, f'final_models_{n}.csv'), index=False)
    versions.to_csv(os.path.join(OUTPUT_DIR, f'final_library_versions_{n}.csv'), index=False)

    sample = pass1_sample(models)
    sample.to_csv(os.path.join(OUTPUT_DIR, f'pass1_sample_{len(sample)}.csv'), index=False)
    print(f'\nSaved {n:,} final models, their library versions, & Pass 1 sample to {OUTPUT_DIR}')
