import os
import csv
import time
import argparse
import requests
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = os.path.join(SCRIPT_DIR, 'output', 'author_types.csv')


def author_type(author):
    # 200 for orgs, 404 for users
    for kind, path in [('organization', 'organizations'), ('individual', 'users')]:
        for attempt in range(10):
            response = requests.get(f'https://huggingface.co/api/{path}/{author}/overview', timeout=30)
            if response.status_code != 429:  
                break
            time.sleep(30)
        if response.status_code == 200:
            return kind
    return 'unknown'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Look up author types')
    parser.add_argument('--models', required=True)
    args = parser.parse_args()

    authors = sorted(set(pd.read_csv(args.models, usecols=['Model ID'])['Model ID'].str.split('/').str[0]))
    done = set()
    if os.path.exists(OUT_FILE):
        done = set(pd.read_csv(OUT_FILE, dtype=str, keep_default_na=False)['author'])
    todo = [a for a in authors if a not in done]

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    new_file = not os.path.exists(OUT_FILE)
    with open(OUT_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(['author', 'author_type'])
        for i, author in enumerate(todo, 1):
            writer.writerow([author, author_type(author)])
            f.flush()  
            if i % 250 == 0:
                print(f'{i:,} / {len(todo):,}')
    print(f'Saved author types to {OUT_FILE}')
