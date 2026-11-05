import os
import csv
import glob
import json
import calendar
import argparse
from datetime import datetime, timezone

# Stage-1 filters, only the must-have ones the team agrees on, all answerable from the listing metadata (applied in this order):
#   has base model, has dataset, not gated, age since upload >= 6 months
# The soft filters (task, likes, lastModified > createdAt) are left commented out in FILTERS below. They are still
# being discussed, so those models stay in the shared data and can be filtered by hand from the CSV columns.
# Stage 2 (--commits, needs crawl-commits.py output for the stage-1 models): commits >= 2
MIN_AGE_MONTHS = 6
MIN_LIKES = 2  # only used by the commented-out likes filter
MIN_COMMITS = 2

# base_model:<relation>:<id> tags say how a model derives from its base
BASE_MODEL_RELATIONS = {"adapter", "finetune", "merge", "quantized"}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data-collection', 'metadata-collection', 'output')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')


def parse_time(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def months_before(date, months):
    month_index = date.year * 12 + date.month - 1 - months
    year, month = divmod(month_index, 12)
    day = min(date.day, calendar.monthrange(year, month + 1)[1])
    return date.replace(year=year, month=month + 1, day=day)


def tag_values(model, prefix):
    return [t[len(prefix):] for t in model.get('tags', []) if t.startswith(prefix)]


def base_models(model):
    # plain base_model:<id> tags only, the relation tags repeat the same ids
    return [v for v in tag_values(model, 'base_model:') if v.split(':')[0] not in BASE_MODEL_RELATIONS]


def base_model_relations(model):
    return sorted({v.split(':')[0] for v in tag_values(model, 'base_model:')} & BASE_MODEL_RELATIONS)


FILTERS = [
    ('has base model', lambda m, cutoff: bool(tag_values(m, 'base_model:'))),
    ('has dataset', lambda m, cutoff: bool(tag_values(m, 'dataset:'))),
    # gated is False, "auto" or "manual"; gated repos need a manual accept before their files and commits can be read
    ('not gated', lambda m, cutoff: m['gated'] is False),
    (f'age >= {MIN_AGE_MONTHS} months', lambda m, cutoff: parse_time(m['createdAt']) <= cutoff),

    # Soft filters, NOT applied for now: the team hasn't agreed on them yet. Uncomment a line to apply it here.
    # ('has task', lambda m, cutoff: bool(m.get('pipeline_tag'))),
    # (f'likes >= {MIN_LIKES}', lambda m, cutoff: m.get('likes', 0) >= MIN_LIKES),
    # ('lastModified > createdAt', lambda m, cutoff: parse_time(m['lastModified']) > parse_time(m['createdAt'])),

    # Other soft criteria need data from the later crawls, so they are columns to filter on instead:
    #   commits >= 2 (or use --commits below), model size, license, has requirements.txt
]


def filter_models(models, cutoff):
    counts = {name: 0 for name, _ in FILTERS}
    seen, kept = set(), []

    for model in models:
        # the listing is paged by trendingScore, so a few models show up twice
        if model['id'] in seen:
            continue
        seen.add(model['id'])

        for name, check in FILTERS:
            if not check(model, cutoff):
                break
            counts[name] += 1
        else:
            # later scripts look models up by modelId, which the expanded listing no longer returns
            model['modelId'] = model['id']
            kept.append(model)

    return kept, len(seen), counts


def filter_by_commits(models, commits_file):
    # models whose commits could not be crawled have no commit_count and are dropped
    with open(commits_file, 'r', encoding='utf-8') as f:
        commit_counts = {r['id']: r.get('commit_count', 0) for r in json.load(f)}
    return [m for m in models if commit_counts.get(m['id'], 0) >= MIN_COMMITS]


def save_csv(models, csv_file):
    with open(csv_file, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        writer.writerow(['Model ID', 'pipeline_tag', 'library_name', 'likes', 'downloads', 'createdAt', 'lastModified',
                         'base_models', 'base_model_relation', 'datasets', 'license'])
        for m in models:
            writer.writerow([m['id'], m.get('pipeline_tag', ''), m.get('library_name', ''), m.get('likes', 0),
                             m.get('downloads', 0), m['createdAt'], m['lastModified'],
                             ';'.join(base_models(m)), ';'.join(base_model_relations(m)),
                             ';'.join(tag_values(m, 'dataset:')), ';'.join(tag_values(m, 'license:'))])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage-1 filter over the Hugging Face model listing')
    parser.add_argument('--input', help='metadata JSON from crawl-metadata.py (default: newest in metadata-collection/output)')
    parser.add_argument('--commits', help=f'commits JSON from crawl-commits.py, also applies commits >= {MIN_COMMITS}')
    args = parser.parse_args()

    input_file = args.input or max(glob.glob(os.path.join(METADATA_DIR, '*_all_huggingface_models_*.json')),
                                   key=os.path.getmtime)
    # age is measured against the snapshot date in the file name, not today, so reruns give the same set
    snapshot_date = os.path.basename(input_file)[:10]
    cutoff = months_before(datetime.strptime(snapshot_date, '%Y-%m-%d').replace(tzinfo=timezone.utc), MIN_AGE_MONTHS)

    print(f"Loading {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        all_models = json.load(f)
    missing = [k for k in ('lastModified', 'gated') if all_models and k not in all_models[0]]
    if missing:
        raise SystemExit(f"Input has no {', '.join(missing)} field, re-crawl it with the current crawl-metadata.py")

    kept, total, counts = filter_models(all_models, cutoff)
    if args.commits:
        kept = filter_by_commits(kept, args.commits)
        counts[f'commits >= {MIN_COMMITS}'] = len(kept)
    print(f"Snapshot {snapshot_date}, age cutoff: created on or before {cutoff.date()}")
    print(f"{'unique models':>26}: {total:,}")
    for name, count in counts.items():
        print(f"{name:>26}: {count:,}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_file_name = os.path.join(OUTPUT_DIR, f'{snapshot_date}_filtered_models_{len(kept)}.json')
    csv_file_name = os.path.join(OUTPUT_DIR, f'{snapshot_date}_filtered_models_{len(kept)}.csv')

    with open(json_file_name, 'w', encoding='utf-8') as f:
        json.dump(kept, f, indent=4)
    print(f"Filtered models saved in JSON file: {json_file_name}")

    save_csv(kept, csv_file_name)
    print(f"Filtered models saved in CSV file: {csv_file_name}")
