import os
import argparse
import pandas as pd

# after Pass 1 is done, draw the 50 drifted models to categorize

N = 50
DOUBLE_CODE = 20
SEED = 42
PASS1_SHEET = 'pass1_sample_260'

# Empty cols for coders
CODING_COLUMNS = ['coder_1', 'coder_1_category', 'coder_2', 'coder_2_category', 'final_category', 'evidence']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')


def load_pass1(path, sheet):
    if path.lower().endswith(('.xlsx', '.xls')):
        pass1 = pd.read_excel(path, sheet_name=sheet)
    else:
        pass1 = pd.read_csv(path)
    # drop the empty trailing columns the Google Sheet export adds
    return pass1.loc[:, ~pass1.columns.astype(str).str.startswith('Unnamed')]


def drift_column(pass1):
    # the sheet header is 'drift (Y/N)', older exports used 'drift'
    matches = [c for c in pass1.columns if str(c).strip().lower().startswith('drift')]
    if len(matches) != 1:
        raise SystemExit(f'Expected one drift column, found {matches}')
    return matches[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw Pass 2 subsample from the labeled Pass 1 sheet')
    parser.add_argument('--pass1', required=True, help='Pass 1 sheet as .csv, or the shared .xlsx')
    parser.add_argument('--sheet', default=PASS1_SHEET, help='Tab to read when --pass1 is an .xlsx')
    args = parser.parse_args()

    pass1 = load_pass1(args.pass1, args.sheet)
    column = drift_column(pass1)
    labels = pass1[column].fillna('').astype(str).str.strip().str.lower()

    # flag rows that are still unlabeled or undecided (e.g. blank, 'Y?') so they aren't silently dropped
    unresolved = pass1[~labels.isin(['y', 'yes', 'n', 'no'])]
    if len(unresolved):
        print(f'Warning: {len(unresolved)} Pass 1 rows have no final drift label and are excluded:')
        print(unresolved.assign(label=unresolved[column].fillna('<blank>'))
              .groupby(['checked_by', 'label']).size().to_string())

    drifted = pass1[labels.isin(['y', 'yes'])]
    print(f'{len(drifted)} of {len(pass1)} Pass 1 models are labeled drift = yes')

    sample = drifted.sample(n=min(N, len(drifted)), random_state=SEED)
    sample['double_code'] = ['yes'] * min(DOUBLE_CODE, len(sample)) + ['no'] * max(len(sample) - DOUBLE_CODE, 0)
    sample = sample.sort_values(['task_bucket', 'Model ID'])
    for column in CODING_COLUMNS:
        sample[column] = ''

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_file = os.path.join(OUTPUT_DIR, f'pass2_coding_{len(sample)}.csv')
    sample.to_csv(out_file, index=False)
    print(f'Saved {len(sample)} models to {out_file}')
