import os
import re
import csv
import json
import time
import argparse
import threading
from collections import Counter, defaultdict
from queue import Queue, Empty
from urllib.parse import quote

import requests
import yaml
from tqdm import tqdm

# Downloads the config files at the root of each model repo (file lists come from crawl-model-size.py) and pulls out
# the library versions recorded in them, e.g. transformers_version in config.json, peft_version in adapter_config.json,
# __version__ in config_sentence_transformers.json, _diffusers_version in model_index.json.
# Only root-level files: config.json files in subfolders are mostly training checkpoints (checkpoint-500/config.json).
CONFIG_FILES = ['config.json', 'generation_config.json', 'adapter_config.json',
                'config_sentence_transformers.json', 'model_index.json']
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TREE_FILE = os.path.join(SCRIPT_DIR, '..', 'model-size-collection', 'output', 'model_size_tree_final.json')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')

# not retried on resume: models deleted or made private since the snapshot give 401 / 404
PERMANENT_ERRORS = {'Status 401', 'Status 403', 'Status 404'}
# config_sentence_transformers.json calls torch "pytorch"; use the pip name so it matches the requirements CSV
PACKAGE_NAMES = {'pytorch': 'torch'}
# a value must look like a version (4.57.1, 2025.10.4, 0.19.0.dev0); this drops model settings that happen to be
# named *_version, like ps_version "v2" or arch_version "applio"
VERSION_RE = re.compile(r'^v?\d+(\.\d+)+')


def wait_for_rate_limit(response):
    # header looks like: "resolvers";r=<requests left in window>;t=<seconds until window resets>
    fields = dict(p.split('=', 1) for p in response.headers.get('ratelimit', '').split(';') if '=' in p)
    if response.status_code == 429 or fields.get('r') == '0':
        wait_time = int(fields.get('t', 60)) + 1
        tqdm.write(f"Rate limit reached, waiting {wait_time}s")
        time.sleep(wait_time)


def download(session, model_id, path, max_retries=5):
    # returns (parsed JSON, error); a file that isn't valid JSON is kept as raw text
    url = f'https://huggingface.co/{model_id}/resolve/main/{quote(path)}'
    error = 'Max retries exceeded'
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, timeout=120)
        except requests.RequestException as e:
            error = str(e)
            time.sleep(10 * attempt)
            continue
        wait_for_rate_limit(response)
        if response.status_code == 200:
            text = response.content.decode('utf-8-sig', errors='replace')
            try:
                return json.loads(text), None
            except json.JSONDecodeError as e:
                return text, f'Invalid JSON: {e}'
        error = f'Status {response.status_code}'
        if response.status_code >= 500:
            time.sleep(10 * attempt)
        elif response.status_code != 429:
            break
    return None, error


def find_versions(value, path=''):
    # yields (key path, version) for every "*_version" or "__version__" string, at any depth
    if isinstance(value, dict):
        for k, v in value.items():
            key = f'{path}.{k}' if path else k
            if isinstance(v, str) and (k.endswith('_version') or k == '__version__'):
                if VERSION_RE.match(v):
                    yield key, v
            elif k == '__version__' and isinstance(v, dict):
                # config_sentence_transformers.json: {"sentence_transformers": "2.2.2", "transformers": ..., "pytorch": ...}
                for lib, version in v.items():
                    if isinstance(version, str) and VERSION_RE.match(version):
                        yield f'{key}.{lib}', version
            else:
                yield from find_versions(v, key)
    elif isinstance(value, list):
        for item in value:
            yield from find_versions(item, f'{path}[]')


def package_of(key):
    # transformers_version -> transformers, _diffusers_version -> diffusers, __version__.sentence_transformers -> sentence-transformers
    parts = key.split('.')
    name = parts[-1] if len(parts) > 1 and parts[-2] == '__version__' else re.sub(r'_version$', '', parts[-1].strip('_'))
    name = name.lower().replace('_', '-')
    return PACKAGE_NAMES.get(name, name)


def worker(api_key, model_queue, out, write_lock, progress_bar):
    # one thread per key, each key has its own rate-limit window
    session = requests.Session()
    session.headers.update({'Authorization': f'Bearer {api_key}'})
    while True:
        try:
            model_id, paths = model_queue.get_nowait()
        except Empty:
            return
        files = []
        for path in paths:
            content, error = download(session, model_id, path)
            if error:
                tqdm.write(f"{model_id}/{path}: {error}")
            files.append({'path': path, 'content': content, 'error': error})
        with write_lock:
            out.write(json.dumps({'id': model_id, 'files': files}, ensure_ascii=False) + '\n')
            out.flush()
            progress_bar.update(1)


def is_done(record):
    # transient errors (429, 5xx, timeouts) get retried on resume
    return all(f['error'] is None or f['error'] in PERMANENT_ERRORS or f['error'].startswith('Invalid JSON')
               for f in record['files'])


def read_jsonl(jsonl_path):
    records = {}
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # half-written last line from an interrupted run
                records[record['id']] = record  # a retried model keeps its latest record
    return records


def merge_results(jsonl_path, model_ids, json_file, csv_file):
    records = read_jsonl(jsonl_path)
    results = [records[mid] for mid in model_ids if mid in records]

    # no indent: a few quantized models have config.json files of 10+ MB that indenting would blow up
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)

    rows = []
    for record in results:
        for file in record['files']:
            for key, version in find_versions(file['content']):
                rows.append({'Model ID': record['id'], 'file': file['path'], 'key': key,
                             'package': package_of(key), 'version': version})
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Model ID', 'file', 'key', 'package', 'version'])
        writer.writeheader()
        writer.writerows(rows)
    return results, rows


def print_summary(n_models, results, rows):
    files = [f for r in results for f in r['files']]
    errors = Counter(f['error'].split(':')[0] for f in files if f['error'])
    print(f"\nModels scanned: {n_models:,}")
    print(f"Models with a root config file: {len(results):,} ({len(results) / n_models:.1%})")
    print(f"Config files: {len(files):,} (errors: {sum(errors.values())})")
    for error, count in errors.most_common():
        print(f"  {error}: {count}")

    with_version = {r['Model ID'] for r in rows}
    print(f"\nModels with at least one recorded version: {len(with_version):,} ({len(with_version) / n_models:.1%})")

    # count each package once per model, show its most common versions
    per_package = defaultdict(dict)
    for r in rows:
        per_package[r['package']].setdefault(r['Model ID'], r['version'])
    print("Top packages (models recording it, most common versions):")
    for package, models in sorted(per_package.items(), key=lambda kv: -len(kv[1]))[:15]:
        top = ', '.join(v for v, _ in Counter(models.values()).most_common(3))
        print(f"{package:>24}: {len(models):>6,} models  ({top})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download the root config files of the crawled models and extract library versions')
    parser.add_argument('--tree', default=TREE_FILE, help='file list JSON from crawl-model-size.py ({model id: {path: size}})')
    parser.add_argument('--limit', type=int, help='only crawl the first N models that have a config file (for testing)')
    args = parser.parse_args()

    with open(os.path.join(SCRIPT_DIR, 'config.yaml'), 'r') as f:
        api_keys = yaml.safe_load(f)['huggingface_keys']

    with open(args.tree, 'r', encoding='utf-8') as f:
        tree = json.load(f)
    targets = {mid: [p for p in CONFIG_FILES if p in files] for mid, files in tree.items() if 'error' not in files}
    targets = {mid: paths for mid, paths in targets.items() if paths}
    model_ids = list(targets)[:args.limit]

    # progress is appended per model, so an interrupted run picks up where it stopped
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    jsonl_path = os.path.join(OUTPUT_DIR, 'model_configs.jsonl')
    done = {mid for mid, record in read_jsonl(jsonl_path).items() if is_done(record)}
    todo = [mid for mid in model_ids if mid not in done]
    print(f"{len(targets):,} of {len(tree):,} models have a root config file")
    print(f"{len(model_ids):,} models, {len(model_ids) - len(todo):,} already done, {len(todo):,} to crawl "
          f"({sum(len(targets[mid]) for mid in todo):,} files)")
    print(f"Using {len(api_keys)} keys, one thread each")

    model_queue = Queue()
    for model_id in todo:
        model_queue.put((model_id, targets[model_id]))
    write_lock = threading.Lock()

    with open(jsonl_path, 'a', encoding='utf-8') as out, tqdm(total=len(todo), unit='model') as progress_bar:
        # daemon threads so Ctrl-C stops the run; finished models are already flushed to the jsonl
        threads = [threading.Thread(target=worker, args=(key, model_queue, out, write_lock, progress_bar), daemon=True)
                   for key in api_keys]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    json_file = os.path.join(OUTPUT_DIR, f'model_configs_{len(model_ids)}.json')
    csv_file = os.path.join(OUTPUT_DIR, f'model_config_versions_{len(model_ids)}.csv')
    results, rows = merge_results(jsonl_path, model_ids, json_file, csv_file)

    print_summary(len(tree) if args.limit is None else len(model_ids), results, rows)
    print(f"\nRaw config files saved in JSON file: {json_file}")
    print(f"One row per recorded version saved in CSV file: {csv_file}")
