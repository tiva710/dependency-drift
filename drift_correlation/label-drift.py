"""Pass 3, step 1: automatically label dependency drift for every final model.

Reads the shared Final Dataset.xlsx and applies the team's Pass 1 rules:
  base model / dataset (from the PR #3 `pointers` tab)
    resolves, gated, or renamed with a working redirect  -> no drift
    malformed tag or placeholder name (custom, imagefolder, train.jsonl, ...) -> invalid, no drift
    single-name dataset that 401/404s, but a copy exists under an organization -> no drift
    anything else that 401/404s (deleted, made private, gone everywhere) -> drift
  library (from the `final_library_versions_10656` tab)
    pinned version >= 2 major versions behind the latest PyPI release -> stale (drift)
    .dev0 builds are compared by their major version; calendar versions are never stale;
    ranged/unpinned requirements are never stale

Outputs (written next to this script, in drift_correlation/):
  pass3_dependency_labels.csv  one row per dependency, with the evidence and rule used
  pass3_model_labels.csv       one row per model, with the drift labels
  pass3_validation.txt         agreement with the human Pass 1 labels
Network lookups (PyPI, Hugging Face search) are cached in drift_correlation/cache/, so reruns are fast.
"""

import os
import re
import json
import time
import argparse
from datetime import date
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from packaging.version import Version, InvalidVersion

MAJOR_GAP = 2
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = SCRIPT_DIR
CACHE_DIR = os.path.join(SCRIPT_DIR, 'cache')

# Dataset tags that name a loader, a local file, or a placeholder instead of a Hub dataset
PLACEHOLDER_NAMES = {
    'imagefolder', 'image_folder', 'audiofolder', 'videofolder', 'arrow', 'json', 'csv', 'parquet', 'text',
    'custom', 'customized', 'generator', 'generated', 'private', 'proprietary', 'internal', 'internal-curated',
    'classified', 'nan', 'none', 'null', 'a', 'load_data', 'local', 'your-dataset-id', 'dataset', 'data',
    'community-data',
}
PLACEHOLDER_PATTERN = re.compile(r'^(custom|synthetic)[-_ a-z0-9]*$|\.(jsonl?|csv|parquet|txt|tsv)$', re.IGNORECASE)

# Owners that mark a placeholder rather than a real Hub account, e.g. custom/polly-generated-voice
PLACEHOLDER_OWNERS = {'custom', 'local', 'private', 'user', 'username', 'your-username', 'my-org'}

# Bare (no owner) dataset names that were official Hugging Face datasets, from the datasets library
# releases that still shipped them (the folder moved to the Hub after 2.5.0)
CANONICAL_RELEASES = ['2.0.0', '2.5.0']

# Calendar-versioned packages whose major number is a two-digit year (four-digit years are detected automatically)
TWO_DIGIT_CALVER = {'pip', 'attrs', 'packaging', 'pyopenssl', 'black', 'twisted', 'incremental', 'cattrs',
                    'flatbuffers', 'webcolors', 'argon2-cffi', 'argon2-cffi-bindings', 'service-identity'}

# Package names recorded in configs that differ from their PyPI name
PYPI_ALIASES = {'nemo': 'nemo-toolkit', 'autoround': 'auto-round'}

session = requests.Session()


def get_json(url, attempts=4):
    for attempt in range(attempts):
        try:
            response = session.get(url, timeout=30)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if response.status_code == 429:
            time.sleep(max(float(response.headers.get('Retry-After', 0) or 0), 2 ** (attempt + 2)))
            continue
        if response.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        return response.status_code, (response.json() if response.status_code == 200 else None)
    return None, None


def cached_lookup(name, keys, fetch, refresh, workers=8):
    path = os.path.join(CACHE_DIR, f'{name}.json')
    cache = {} if refresh or not os.path.exists(path) else json.load(open(path))
    todo = [k for k in dict.fromkeys(keys) if k not in cache]
    print(f'{name}: {len(keys) - len(todo)} cached, {len(todo)} to look up')
    with ThreadPoolExecutor(workers) as pool:
        for key, value in zip(todo, pool.map(fetch, todo)):
            if value is not None:  # don't cache failed requests; they're retried next run
                cache[key] = value
    os.makedirs(CACHE_DIR, exist_ok=True)
    json.dump(cache, open(path, 'w'), indent=1, sort_keys=True)
    return cache


# ---------- base models and datasets ----------

def is_placeholder(name):
    owner = name.split('/')[0].lower() if '/' in name else None
    return name.lower() in PLACEHOLDER_NAMES or bool(PLACEHOLDER_PATTERN.search(name)) or owner in PLACEHOLDER_OWNERS


def canonical_datasets(refresh):
    path = os.path.join(CACHE_DIR, 'hf_canonical_datasets.json')
    if os.path.exists(path) and not refresh:
        return set(json.load(open(path)))
    names = set()
    for release in CANONICAL_RELEASES:
        _, root = get_json(f'https://api.github.com/repos/huggingface/datasets/git/trees/{release}')
        folder = next(t['sha'] for t in root['tree'] if t['path'] == 'datasets')
        _, tree = get_json(f'https://api.github.com/repos/huggingface/datasets/git/trees/{folder}')
        names |= {t['path'].lower() for t in tree['tree'] if t['type'] == 'tree'}
    os.makedirs(CACHE_DIR, exist_ok=True)
    json.dump(sorted(names), open(path, 'w'), indent=1)
    return names


def search_copies(name):
    # exact-name matches under any owner, e.g. voxpopuli -> facebook/voxpopuli
    status, results = get_json(f'https://huggingface.co/api/datasets?search={requests.utils.quote(name)}&limit=1000')
    if status != 200:
        return None
    return [r['id'] for r in results if r['id'].split('/')[-1].lower() == name.lower()]


def is_org(owner):
    status, _ = get_json(f'https://huggingface.co/api/organizations/{requests.utils.quote(owner)}/overview')
    return None if status is None else status == 200


def label_pointers(pointers, refresh):
    pointers = pointers.copy()
    pointers['dependency_id'] = pointers['dependency_id'].astype(str)
    broken = pointers['status'].isin(['access_restricted', 'not_found_or_private'])
    single_name = ~pointers['dependency_id'].str.contains('/', regex=False)
    canonical = canonical_datasets(refresh)
    to_search = sorted(n for n in set(pointers.loc[broken & single_name, 'dependency_id'])
                       if n.lower() in canonical and not is_placeholder(n))

    copies = cached_lookup('hf_copy_search', to_search, search_copies, refresh, workers=4)
    owners = sorted({c.split('/')[0] for found in copies.values() for c in found})
    orgs = cached_lookup('hf_is_org', owners, is_org, refresh, workers=4)

    def rule(row):
        dep, status = row['dependency_id'], row['status']
        if status == 'invalid_reference':
            return 'invalid', 'N', 'malformed tag'
        if status in ('metadata_accessible', 'gated'):
            if row.get('id_changed') == 1:  # read from the sheet as 1.0 / 0.0
                return 'renamed_redirect', 'N', f"redirects to {row['resolved_id']}"
            return ('gated', 'N', '') if status == 'gated' else ('exists', 'N', '')
        if status == 'private':
            return 'gone', 'Y', 'repository is private'
        if status not in ('access_restricted', 'not_found_or_private'):
            return 'unresolved', '', f'checker status {status}'
        if is_placeholder(dep):
            return 'invalid', 'N', 'placeholder or local name, not a Hub dataset'
        if '/' not in dep:
            if dep.lower() not in canonical:
                return 'invalid', 'N', 'bare name that was never an official Hub dataset (outside dataset or missing owner)'
            found = copies.get(dep)
            if found is None:
                return 'unresolved', '', 'copy search failed'
            org_copies = [c for c in found if orgs.get(c.split('/')[0])]
            if org_copies:
                return 'moved_copy_exists', 'N', f"copy under org: {', '.join(org_copies[:3])}"
            if found:
                return 'gone', 'Y', f"no org copy; user copies only: {', '.join(found[:3])}"
            return 'gone', 'Y', 'no copy found on the Hub'
        return 'gone', 'Y', f'HTTP {row["http_status"]:.0f} (deleted or private)'

    labels = pointers.apply(rule, axis=1, result_type='expand')
    pointers['rule'], pointers['drifted'], pointers['evidence'] = labels[0], labels[1], labels[2]
    return pd.DataFrame({
        'Model ID': pointers['model_id'], 'dependency_type': pointers['dependency_type'],
        'dependency': pointers['dependency_id'], 'check_result': pointers['status'],
        'rule': pointers['rule'], 'drifted': pointers['drifted'], 'evidence': pointers['evidence'],
    })


# ---------- libraries ----------

def normalize(package):
    name = re.sub(r'[-_.]+', '-', str(package)).lower()
    return PYPI_ALIASES.get(name, name)


def pypi_latest(package):
    status, data = get_json(f'https://pypi.org/pypi/{package}/json')
    if status == 404:
        return ''  # not on PyPI
    return data['info']['version'] if status == 200 else None


def parse(version):
    try:
        return Version(str(version).strip().split('@')[0])  # drop '@<commit>' from git installs
    except InvalidVersion:
        return None


def label_libraries(versions, refresh):
    versions = versions.copy()
    versions['pkg'] = versions['package'].map(normalize)
    pinned = versions['kind'] == 'pinned'
    latest = cached_lookup('pypi_latest', sorted(set(versions.loc[pinned, 'pkg'])), pypi_latest, refresh, workers=16)

    def rule(row):
        if row['kind'] != 'pinned':
            return 'not_pinned', 'N', f"{row['kind']} requirement", None, None
        newest = latest.get(row['pkg'])
        if newest is None:
            return 'unresolved', '', 'PyPI lookup failed', None, None
        if newest == '':
            return 'not_on_pypi', 'N', 'package not on PyPI', None, None
        have, current = parse(row['version_spec']), parse(newest)
        if have is None or current is None:
            return 'invalid', 'N', 'version cannot be parsed', newest, None
        if have.major >= 1000 or current.major >= 1000 or row['pkg'] in TWO_DIGIT_CALVER:
            return 'calendar_version', 'N', 'calendar versioning, rule does not apply', newest, None
        gap = current.major - have.major
        if gap >= MAJOR_GAP:
            return 'stale', 'Y', f'{gap} majors behind {newest}', newest, gap
        return 'current_enough', 'N', ('dev build; ' if have.is_devrelease else '') + f'{gap} majors behind {newest}', newest, gap

    labels = versions.apply(rule, axis=1, result_type='expand')
    return pd.DataFrame({
        'Model ID': versions['Model ID'], 'dependency_type': 'library',
        'dependency': versions['package'] + ' ' + versions['version_spec'].astype(str),
        'check_result': versions['file'], 'rule': labels[0], 'drifted': labels[1], 'evidence': labels[2],
        'latest_version': labels[3], 'major_gap': labels[4],
    })


# ---------- per model ----------

def summarize(models, deps):
    def any_yes(kind):
        part = deps[deps['dependency_type'] == kind]
        return part.groupby('Model ID')['drifted'].apply(lambda s: 'Y' if (s == 'Y').any() else 'N')

    out = pd.DataFrame({'Model ID': models['Model ID']})
    for kind, column in [('base_model', 'base_drift'), ('dataset', 'dataset_drift'), ('library', 'library_stale')]:
        out[column] = out['Model ID'].map(any_yes(kind)).fillna('N')
    flags = out[['base_drift', 'dataset_drift', 'library_stale']]
    out['drift'] = (flags == 'Y').any(axis=1).map({True: 'Y', False: 'N'})
    out['drift_types'] = flags.apply(lambda r: ';'.join(c for c in flags.columns if r[c] == 'Y'), axis=1)

    counts = deps.groupby(['Model ID', 'dependency_type']).size().unstack(fill_value=0)
    out['n_base_models'] = out['Model ID'].map(counts.get('base_model', {})).fillna(0).astype(int)
    out['n_datasets'] = out['Model ID'].map(counts.get('dataset', {})).fillna(0).astype(int)
    pinned = deps[(deps['dependency_type'] == 'library') & (deps['rule'] != 'not_pinned')]
    out['n_pinned_libs'] = out['Model ID'].map(pinned.groupby('Model ID').size()).fillna(0).astype(int)
    out['n_invalid_tags'] = out['Model ID'].map(deps[deps['rule'] == 'invalid'].groupby('Model ID').size()).fillna(0).astype(int)
    out['n_unresolved'] = out['Model ID'].map(deps[deps['rule'] == 'unresolved'].groupby('Model ID').size()).fillna(0).astype(int)
    drifted = deps[deps['drifted'] == 'Y']
    text = drifted['dependency_type'] + ': ' + drifted['dependency'] + ' (' + drifted['evidence'] + ')'
    out['drift_evidence'] = out['Model ID'].map(text.groupby(drifted['Model ID']).agg(' | '.join).str[:500]).fillna('')
    return out


# ---------- agreement with the human Pass 1 labels ----------

def kappa(a, b):
    labels = sorted(set(a) | set(b))
    table = pd.crosstab(a, b).reindex(index=labels, columns=labels, fill_value=0)
    n = table.values.sum()
    observed = sum(table.loc[k, k] for k in labels) / n
    expected = sum(table.loc[k].sum() * table[k].sum() for k in labels) / n ** 2
    return table, observed, (observed - expected) / (1 - expected) if expected < 1 else float('nan')


def validate(pass1, labels):
    pass1 = pass1.loc[:, ~pass1.columns.astype(str).str.startswith('Unnamed')]
    col = lambda prefix: next(c for c in pass1.columns if str(c).lower().startswith(prefix))
    merged = pass1.merge(labels, on='Model ID', how='left')
    yn = lambda s: s.fillna('').astype(str).str.strip().str.upper()
    checks = [
        ('base model gone', yn(merged[col('base_model_exists')]).map({'N': 'Y', 'Y': 'N'}), merged['base_drift']),
        ('dataset gone', yn(merged[col('dataset_exists')]).map({'N': 'Y', 'Y': 'N', 'INVALID': 'N'}), merged['dataset_drift']),
        ('library stale', yn(merged[col('library_version_stale')]), merged['library_stale']),
        ('drift', yn(merged[col('drift')]), merged['drift']),
    ]
    lines = [f'Agreement with human Pass 1 labels ({len(pass1)} models; rows with blank or "?" human labels skipped)\n']
    for name, human, auto in checks:
        keep = human.isin(['Y', 'N'])
        table, observed, k = kappa(human[keep].rename('human'), auto[keep].rename('automatic'))
        lines.append(f'== {name}: n={keep.sum()}  agreement={observed:.1%}  kappa={k:.2f}\n{table.to_string()}\n')
    human = yn(merged[col('drift')])
    off = merged[human.isin(['Y', 'N']) & (human != merged['drift'])]
    lines.append(f'Disagreements on drift ({len(off)}):')
    for _, r in off.iterrows():
        lines.append(f"  {r['checked_by']:9} human={yn(pd.Series([r[col('drift')]]))[0]} auto={r['drift']}  {r['Model ID']}  "
                     f"auto evidence: {r['drift_evidence'][:160] or '-'}  | human note: {str(r['notes'])[:120] if pd.notna(r['notes']) else '-'}")
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automatically label dependency drift for all final models')
    parser.add_argument('--data', required=True, help='Final Dataset.xlsx downloaded from the shared sheet')
    parser.add_argument('--refresh', action='store_true', help='Ignore cached PyPI / Hugging Face lookups')
    args = parser.parse_args()

    sheets = pd.read_excel(args.data, sheet_name=['final_models_10656', 'pointers', 'final_library_versions_10656', 'pass1_sample_260'])
    models = sheets['final_models_10656']
    deps = pd.concat([label_pointers(sheets['pointers'], args.refresh),
                      label_libraries(sheets['final_library_versions_10656'], args.refresh)], ignore_index=True)
    deps = deps[deps['Model ID'].isin(models['Model ID'])]
    labels = summarize(models, deps)
    labels.insert(1, 'labeled_on', date.today().isoformat())

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    deps.to_csv(os.path.join(OUTPUT_DIR, 'pass3_dependency_labels.csv'), index=False)
    labels.to_csv(os.path.join(OUTPUT_DIR, 'pass3_model_labels.csv'), index=False)
    report = validate(sheets['pass1_sample_260'], labels)
    open(os.path.join(OUTPUT_DIR, 'pass3_validation.txt'), 'w').write(report + '\n')

    n = len(labels)
    print(f'\n{n:,} models labeled')
    for column in ['base_drift', 'dataset_drift', 'library_stale', 'drift']:
        k = (labels[column] == 'Y').sum()
        print(f'  {column:14} Y = {k:5,} ({k / n:.1%})')
    print('\nDependency rules applied:\n' + deps.groupby(['dependency_type', 'rule']).size().to_string())
    print('\n' + report.split('Disagreements')[0])
    print(f'Saved to {OUTPUT_DIR}')
