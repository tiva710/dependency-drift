import os
import csv
import glob
import json
import time
import argparse
import threading
from queue import Queue, Empty
import requests
import yaml
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILTERED_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'model-filtering', 'output')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')

# not retried on resume: gated models give 403, models deleted or renamed since the snapshot give 404
PERMANENT_ERRORS = {"Status 401", "Status 403", "Status 404"}


class HuggingFaceAPI:
    def __init__(self, api_key=None):
        self.base_url = 'https://huggingface.co'
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def get_commits(self, model_id, max_retries=5):
        # returns (commits, error); limit=1000 fits almost every repo in one page, next links cover the rest
        commits = []
        url = f"{self.base_url}/api/models/{model_id}/commits/main?limit=1000"
        while url:
            response = self.get(url, max_retries)
            if response is None:
                return None, "Max retries exceeded"
            if response.status_code != 200:
                return None, f"Status {response.status_code}"
            commits.extend(response.json())
            url = response.links.get("next", {}).get("url")
        return commits, None

    def get(self, url, max_retries):
        # retry on rate limits (429) / server errors; HF rate-limit window is 5 min
        for attempt in range(1, max_retries + 1):
            try:
                response = self.session.get(url, timeout=60)
            except requests.RequestException as e:
                tqdm.write(f"Request error: {e}, retrying ({attempt}/{max_retries})")
                time.sleep(10 * attempt)
                continue
            self.wait_for_rate_limit(response)
            if response.status_code == 429 or response.status_code >= 500:
                tqdm.write(f"Request failed: Status {response.status_code}, retrying ({attempt}/{max_retries})")
                if response.status_code >= 500:
                    time.sleep(10 * attempt)
                continue
            return response
        return None

    def wait_for_rate_limit(self, response):
        # header looks like: "api";r=<requests left in window>;t=<seconds until window resets>
        fields = dict(p.split('=', 1) for p in response.headers.get('ratelimit', '').split(';') if '=' in p)
        if response.status_code == 429 or fields.get('r') == '0':
            wait_time = int(fields.get('t', 60)) + 1
            tqdm.write(f"Rate limit reached, waiting {wait_time}s")
            time.sleep(wait_time)


def slim_commits(commits):
    # API returns newest first; store oldest first so commits[0] is the initial commit and commits[-1] the latest
    return [{
        "id": c.get("id"),
        "date": c.get("date"),
        "authors": [a.get("user") for a in c.get("authors", [])],
        "title": c.get("title"),
        "message": c.get("message"),
    } for c in reversed(commits)]


def worker(api_key, model_queue, out, write_lock, progress_bar):
    # one thread per key, each key has its own rate-limit window
    hf_api = HuggingFaceAPI(api_key)
    while True:
        try:
            model_id = model_queue.get_nowait()
        except Empty:
            return
        commits, error = hf_api.get_commits(model_id)
        if error:
            tqdm.write(f"{model_id}: {error}")
            record = {"id": model_id, "error": error}
        else:
            record = {"id": model_id, "commit_count": len(commits), "commits": slim_commits(commits)}
        with write_lock:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            progress_bar.update(1)


def load_done_ids(jsonl_path):
    # models with commits or a permanent error are skipped on resume, transient errors get retried
    done = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # half-written last line from an interrupted run
                if 'commits' in record or record.get('error') in PERMANENT_ERRORS:
                    done.add(record['id'])
    return done


def merge_results(jsonl_path, model_ids, json_file, csv_file):
    records = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records[record['id']] = record  # a retried model keeps its latest record
    results = [records[mid] for mid in model_ids if mid in records]

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # one row per commit, commit_index 0 is the initial commit
    with open(csv_file, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        writer.writerow(['Model ID', 'commit_count', 'commit_index', 'commit_id', 'date', 'authors', 'title', 'message'])
        for record in results:
            commits = record.get('commits')
            if commits is None:
                continue
            for i, c in enumerate(commits):
                writer.writerow([record['id'], len(commits), i, c['id'], c['date'], ';'.join(filter(None, c['authors'])),
                                 c['title'], c['message']])
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crawl the commit history of each filtered model')
    parser.add_argument('--input', help='filtered JSON from filter-models.py (default: newest in model-filtering/output)')
    parser.add_argument('--limit', type=int, help='only crawl the first N models (for testing)')
    args = parser.parse_args()

    with open(os.path.join(SCRIPT_DIR, 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    api_keys = config['huggingface_keys']

    input_file = args.input or max(glob.glob(os.path.join(FILTERED_DIR, '*_filtered_models_*.json')),
                                   key=os.path.getmtime)
    snapshot_date = os.path.basename(input_file)[:10]
    with open(input_file, 'r', encoding='utf-8') as f:
        model_ids = [m['id'] for m in json.load(f)][:args.limit]

    # progress is appended per model, so an interrupted run picks up where it stopped
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    jsonl_path = os.path.join(OUTPUT_DIR, f'{snapshot_date}_model_commits.jsonl')
    done = load_done_ids(jsonl_path)
    todo = [mid for mid in model_ids if mid not in done]
    print(f"Input: {input_file}")
    print(f"{len(model_ids)} models, {len(model_ids) - len(todo)} already done, {len(todo)} to crawl")
    print(f"Using {len(api_keys)} keys, one thread each")

    model_queue = Queue()
    for model_id in todo:
        model_queue.put(model_id)
    write_lock = threading.Lock()

    with open(jsonl_path, 'a', encoding='utf-8') as out, tqdm(total=len(todo), unit='model') as progress_bar:
        # daemon threads so Ctrl-C stops the run; finished models are already flushed to the jsonl
        threads = [threading.Thread(target=worker, args=(key, model_queue, out, write_lock, progress_bar), daemon=True)
                   for key in api_keys]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    json_file_name = os.path.join(OUTPUT_DIR, f'{snapshot_date}_model_commits_{len(model_ids)}.json')
    csv_file_name = os.path.join(OUTPUT_DIR, f'{snapshot_date}_model_commits_{len(model_ids)}.csv')
    results = merge_results(jsonl_path, model_ids, json_file_name, csv_file_name)

    errors = [r['error'] for r in results if 'error' in r]
    print(f"Commits obtained for {len(results) - len(errors)} models, {len(errors)} errors")
    for error in sorted(set(errors)):
        print(f"  {error}: {errors.count(error)}")
    print(f"Commits saved in JSON file: {json_file_name}")
    print(f"Commits saved in CSV file: {csv_file_name}")
