import os
import argparse
import pandas as pd

# after Pass 1 is done, draw the 50 drifted models to categorize 

N = 50
DOUBLE_CODE = 20
SEED = 42

# Empty cols for coders
CODING_COLUMNS = ['coder_1', 'coder_1_category', 'coder_2', 'coder_2_category', 'final_category', 'evidence']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw Pass 2 subsample from the labeled Pass 1 sheet')
    parser.add_argument('--pass1', required=True)
    args = parser.parse_args()

    pass1 = pd.read_csv(args.pass1)
    drifted = pass1[pass1['drift'].astype(str).str.strip().str.lower() == 'yes']
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
