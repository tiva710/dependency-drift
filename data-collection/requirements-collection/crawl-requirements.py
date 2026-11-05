import os
import re
import csv
import json
import time
import argparse
from collections import Counter, defaultdict
from urllib.parse import quote

import requests
from packaging.requirements import Requirement, InvalidRequirement

# Finds requirements files in the file lists from crawl-model-size.py, downloads them and splits them into packages.
# Every requirement line gets a kind:
#   pinned   (==, ===)   ranged (>=, <, ~=, ...)   unpinned (no version)
#   url      (git+..., pkg @ file:///...)   option (-r, -e, --index-url ...)   unparsed (not a valid requirement)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TREE_FILE = os.path.join(SCRIPT_DIR, '..', 'model-size-collection', 'output', 'model_size_tree_final.json')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')


def is_requirements_file(path):
    # requirements.txt anywhere in the repo, plus variants like requirements-dev.txt or inference_requirements.txt
    name = os.path.basename(path).lower()
    return name.endswith('.txt') and 'requirements' in name


def decode(data):
    # "pip freeze > requirements.txt" in Windows PowerShell writes UTF-16
    if data.startswith((b'\xff\xfe', b'\xfe\xff')):
        return data.decode('utf-16')
    return data.decode('utf-8-sig', errors='replace')


def download(session, model_id, path):
    url = f'https://huggingface.co/{model_id}/resolve/main/{quote(path)}'
    for attempt in range(5):
        try:
            response = session.get(url, timeout=60)
        except requests.RequestException as e:
            error = str(e)
            time.sleep(5)
            continue
        if response.status_code == 200:
            return decode(response.content), None
        error = f'Status {response.status_code}'
        if response.status_code == 429:
            # ratelimit header looks like "resolvers";r=0;t=123, t is the seconds until the window resets
            reset = re.search(r't=(\d+)', response.headers.get('ratelimit', ''))
            time.sleep(int(reset.group(1)) + 1 if reset else 30)
        elif response.status_code >= 500:
            time.sleep(5 * (attempt + 1))
        else:
            break
    return None, error


def parse_line(line):
    line = re.split(r'\s+#', line, maxsplit=1)[0].strip()
    if not line or line.startswith('#'):
        return None
    if line.startswith('-'):
        return {'package': '', 'specifier': '', 'kind': 'option'}
    if line.startswith(('git+', 'http://', 'https://')):
        return {'package': '', 'specifier': '', 'kind': 'url'}
    try:
        req = Requirement(line)
    except InvalidRequirement:
        return {'package': '', 'specifier': '', 'kind': 'unparsed'}

    package = req.name.lower().replace('_', '-')
    specifier = str(req.specifier)
    if req.url:
        kind = 'url'
    elif any(s.operator in ('==', '===') for s in req.specifier):
        kind = 'pinned'
    elif specifier:
        kind = 'ranged'
    else:
        kind = 'unpinned'
    return {'package': package, 'specifier': specifier, 'kind': kind}


def print_summary(n_models, results, rows):
    files = [f for r in results for f in r['files']]
    errors = [f for f in files if f['error']]
    print(f"\nModels scanned: {n_models:,}")
    print(f"Models with a requirements file: {len(results):,} ({len(results) / n_models:.1%})")
    print(f"Requirements files: {len(files):,} (download errors: {len(errors)})")

    kinds = Counter(r['kind'] for r in rows)
    print(f"\nRequirement lines: {len(rows):,}")
    for kind, count in kinds.most_common():
        print(f"{kind:>10}: {count:>5,} ({count / len(rows):.1%})")

    # count each package once per model
    per_package = defaultdict(dict)
    for r in rows:
        if r['package']:
            per_package[r['package']].setdefault(r['Model ID'], r['kind'])
    print("\nTop packages (models listing it, share pinned with ==):")
    for package, models in sorted(per_package.items(), key=lambda kv: -len(kv[1]))[:15]:
        pinned = sum(k == 'pinned' for k in models.values())
        print(f"{package:>24}: {len(models):>3} models, {pinned / len(models):.0%} pinned")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and parse requirements files of the crawled models')
    parser.add_argument('--tree', default=TREE_FILE, help='file list JSON from crawl-model-size.py ({model id: {path: size}})')
    args = parser.parse_args()

    with open(args.tree, 'r', encoding='utf-8') as f:
        tree = json.load(f)

    targets = {mid: sorted(p for p in files if is_requirements_file(p))
               for mid, files in tree.items() if 'error' not in files}
    targets = {mid: paths for mid, paths in targets.items() if paths}
    print(f"{len(targets):,} of {len(tree):,} models have a requirements file, downloading {sum(map(len, targets.values()))} files")

    session = requests.Session()
    results, rows = [], []
    for i, (model_id, paths) in enumerate(targets.items(), 1):
        files = []
        for path in paths:
            content, error = download(session, model_id, path)
            files.append({'path': path, 'content': content, 'error': error})
            for line_no, line in enumerate((content or '').splitlines(), 1):
                parsed = parse_line(line)
                if parsed:
                    rows.append({'Model ID': model_id, 'file': path, 'line_no': line_no, **parsed, 'raw': line.strip()})
        results.append({'id': model_id, 'files': files})
        print(f"\r{i}/{len(targets)}", end='', flush=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_file = os.path.join(OUTPUT_DIR, f'requirements_{len(results)}.json')
    csv_file = os.path.join(OUTPUT_DIR, f'requirements_{len(results)}.csv')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Model ID', 'file', 'line_no', 'package', 'specifier', 'kind', 'raw'])
        writer.writeheader()
        writer.writerows(rows)

    print_summary(len(tree), results, rows)
    print(f"\nRaw files saved in JSON file: {json_file}")
    print(f"One row per requirement saved in CSV file: {csv_file}")
