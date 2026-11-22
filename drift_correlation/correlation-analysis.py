import os
import argparse
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.proportion import proportion_confint
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Pass 3, Step 2b (RQ3): which model characteristics go together with drift?
#   Part B  one characteristic at a time: drift rate per group, chi-square / Mann-Whitney, Holm correction
#   Part C  all characteristics together: logistic regression, odds ratios with 95% CIs
#   Part D  robustness: random forest vs logistic regression, cross-validated AUC, permutation importance
# Each part runs for every scenario below (two outcomes, with and without the Common Voice models).
# This script only produces the statistics (CSVs in results/); the charts are made by graphs_and_tables/.

SEED = 42
FOLDS = 5
PERMUTATION_REPEATS = 5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_FILE = os.path.join(SCRIPT_DIR, 'pass3_features.csv')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

SCENARIOS = {
    'drift_all':                      ('drift', False),
    'drift_no_common_voice':          ('drift', True),
    'dataset_drift_all':              ('dataset_drift', False),
    'dataset_drift_no_common_voice':  ('dataset_drift', True),
}

# category order and reference group (the first one) for each categorical characteristic
CATEGORIES = {
    'task_bucket': ['NLP', 'Audio', 'Computer Vision', 'Multimodal', 'Reinforcement Learning'],
    'author_type': ['individual', 'organization'],
    'downloads_bucket': ['0-9dl', '10-99dl', '100-999dl', '1000-9999dl', '10k+dl'],
    'upload_age': ['6-12mo', '13-24mo', '25-36mo', '37+mo'],
}
BINARY = ['is_permissive', 'has_requirements_file', 'author_owns_dependency', 'base_from_big_org']
# numeric characteristics; skewed counts are log-transformed before modelling
NUMERIC = {'age_months': False, 'months_since_last_commit': False, 'likes': True, 'n_commits': True,
           'n_base_models': True, 'n_datasets': True, 'n_pinned_libs': True}

# characteristics used in the models (upload_age is left out there: age_months carries the same information)
MODEL_CATEGORICAL = ['task_bucket', 'author_type', 'downloads_bucket']

LABELS = {
    'task_bucket': 'Task', 'author_type': 'Author type', 'downloads_bucket': 'Downloads', 'upload_age': 'Upload age',
    'is_permissive': 'Permissive license', 'has_requirements_file': 'Has requirements.txt',
    'author_owns_dependency': 'Author owns a dependency', 'base_from_big_org': 'Base model from big org',
    'age_months': 'Age (months)', 'months_since_last_commit': 'Months since last commit', 'likes': 'Likes',
    'n_commits': '# commits', 'n_base_models': '# base models', 'n_datasets': '# datasets',
    'n_pinned_libs': '# pinned libraries',
}

# ---------- data ----------

def load_features(path):
    features = pd.read_csv(path)
    before = len(features)
    # groups too small to estimate: 'Other' task (10 models) and 'unknown' author (3 models)
    features = features[features['task_bucket'].isin(CATEGORIES['task_bucket'])
                        & features['author_type'].isin(CATEGORIES['author_type'])].copy()
    print(f'Using {len(features)} of {before} models (dropped tiny groups: Other task, unknown author)')
    for column in ['likes', 'n_commits', 'n_base_models', 'n_datasets', 'n_pinned_libs']:
        features[f'log_{column}'] = np.log1p(features[column])
    return features


def model_column(feature):
    return f'log_{feature}' if NUMERIC.get(feature) else feature


def design_matrix(data):
    # one-hot categoricals (reference group dropped) + binaries + standardized numerics
    parts = []
    for feature in MODEL_CATEGORICAL:
        levels = CATEGORIES[feature]
        dummies = pd.get_dummies(pd.Categorical(data[feature], categories=levels), prefix=feature, dtype=float)
        parts.append(dummies.drop(columns=f'{feature}_{levels[0]}').set_index(data.index))
    parts.append(data[BINARY].astype(float))
    parts.append(data[[model_column(f) for f in NUMERIC]].astype(float))
    matrix = pd.concat(parts, axis=1)
    groups = {f: [c for c in matrix.columns if c.startswith(f'{f}_')] for f in MODEL_CATEGORICAL}
    groups.update({f: [f] for f in BINARY})
    groups.update({f: [model_column(f)] for f in NUMERIC})
    return matrix, groups


# ---------- Part B: one characteristic at a time ----------

def numeric_groups(values, whole_numbers):
    # split a numeric characteristic into (up to) quartile groups with readable names, e.g. '1', '2-3', '9+'
    edges = np.unique(np.quantile(values, [0, 0.25, 0.5, 0.75, 1]))
    if whole_numbers and len(edges) < 4:
        # counts where most models share one value (e.g. # base models is almost always 1): use min, min+1, min+2+
        low = values.min()
        edges = np.array([low - 1, low, low + 1, values.max()])
    bins = pd.cut(values, edges, include_lowest=True, duplicates='drop')
    names = {}
    for i, interval in enumerate(bins.cat.categories):
        if whole_numbers:
            # name each group by the smallest and largest value actually in it
            members = values[bins == interval]
            start, end = int(members.min()), int(members.max())
            last = i == len(bins.cat.categories) - 1
            names[interval] = f'{start}+' if last and end > start else (str(start) if start == end else f'{start}-{end}')
        else:
            names[interval] = f'{max(interval.left, 0):.0f}-{interval.right:.0f}'
    return bins.map(names).astype(str), list(names.values())


def rate_table(data, outcome):
    rows = []
    overall = data[outcome].mean()
    for feature in list(CATEGORIES) + BINARY + list(NUMERIC):
        if feature in NUMERIC:
            groups, order = numeric_groups(data[feature], whole_numbers=feature not in ('age_months', 'months_since_last_commit'))
        elif feature in BINARY:
            groups, order = data[feature].map({0: 'no', 1: 'yes'}), ['no', 'yes']
        else:
            groups, order = data[feature].astype(str), CATEGORIES[feature]
        for group in order:
            in_group = data.loc[groups == group, outcome]
            if in_group.empty:
                continue
            low, high = proportion_confint(in_group.sum(), len(in_group), method='wilson')
            rows.append({'feature': feature, 'label': LABELS[feature], 'group': group, 'models': len(in_group),
                         'drifted': int(in_group.sum()), 'rate': in_group.mean(), 'ci_low': low, 'ci_high': high,
                         'overall_rate': overall})
    return pd.DataFrame(rows)


def cramers_v(table):
    chi2 = chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    return np.sqrt(chi2 / (n * (min(table.shape) - 1)))


def univariate_tests(data, outcome):
    rows = []
    for feature in list(CATEGORIES) + BINARY:
        table = pd.crosstab(data[feature], data[outcome])
        chi2, p, dof, _ = chi2_contingency(table)
        rows.append({'feature': feature, 'test': 'chi-square', 'statistic': chi2, 'p_value': p,
                     'effect_size': cramers_v(table), 'effect_measure': "Cramer's V"})
    for feature in NUMERIC:
        drifted, clean = data.loc[data[outcome] == 1, feature], data.loc[data[outcome] == 0, feature]
        u, p = mannwhitneyu(drifted, clean)
        rows.append({'feature': feature, 'test': 'Mann-Whitney U', 'statistic': u, 'p_value': p,
                     # rank-biserial: +1 = drifted models always higher, -1 = always lower
                     'effect_size': 2 * u / (len(drifted) * len(clean)) - 1, 'effect_measure': 'rank-biserial r',
                     'median_drift_Y': drifted.median(), 'median_drift_N': clean.median()})
    tests = pd.DataFrame(rows)
    tests.insert(1, 'label', tests['feature'].map(LABELS))
    tests['p_holm'] = multipletests(tests['p_value'], method='holm')[1]
    tests['significant'] = tests['p_holm'] < 0.05
    return tests.sort_values('p_holm')


# ---------- Part C: all characteristics together ----------

def logistic_regression(data, outcome):
    data = data.copy()
    numeric = [model_column(f) for f in NUMERIC]
    # standardize numerics so each odds ratio means "per 1 standard deviation"
    data[numeric] = (data[numeric] - data[numeric].mean()) / data[numeric].std()
    terms = [f'C({f}, Treatment("{CATEGORIES[f][0]}"))' for f in MODEL_CATEGORICAL] + BINARY + numeric
    fit = smf.logit(f'{outcome} ~ ' + ' + '.join(terms), data=data).fit(disp=False, maxiter=200)

    ci = fit.conf_int()
    table = pd.DataFrame({'term': fit.params.index, 'odds_ratio': np.exp(fit.params),
                          'ci_low': np.exp(ci[0]), 'ci_high': np.exp(ci[1]), 'p_value': fit.pvalues})
    table = table[table['term'] != 'Intercept']
    table['p_holm'] = multipletests(table['p_value'], method='holm')[1]
    table['label'] = table['term'].map(term_label)
    fit_stats = {'n': int(fit.nobs), 'events': int(data[outcome].sum()), 'pseudo_r2': fit.prsquared,
                 'converged': fit.mle_retvals['converged']}
    return table, fit_stats


def term_label(term):
    # 'C(task_bucket, Treatment("NLP"))[T.Audio]' -> 'Task: Audio (vs NLP)'
    if term.startswith('C('):
        feature = term[2:term.index(',')]
        level = term.split('[T.')[1].rstrip(']')
        return f'{LABELS[feature]}: {level} (vs {CATEGORIES[feature][0]})'
    feature = term[4:] if term.startswith('log_') else term
    suffix = ' (log, per SD)' if term.startswith('log_') else (' (per SD)' if feature in NUMERIC else ' (yes vs no)')
    return LABELS[feature] + suffix


def vif_table(data):
    matrix, _ = design_matrix(data)
    matrix.insert(0, 'const', 1.0)
    values = [variance_inflation_factor(matrix.to_numpy(), i) for i in range(1, matrix.shape[1])]
    return pd.DataFrame({'column': matrix.columns[1:], 'vif': values}).sort_values('vif', ascending=False)


# ---------- Part D: random forest vs logistic regression ----------

def classifiers():
    return {
        'logistic_regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced')),
        'random_forest': RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight='balanced',
                                                n_jobs=-1, random_state=SEED),
    }


def grouped_permutation_importance(model, X, y, groups, rng):
    # shuffle all columns of one characteristic together (e.g. every task dummy) and measure the AUC drop
    baseline = roc_auc_score(y, model.predict_proba(X)[:, 1])
    drops = {}
    for feature, columns in groups.items():
        scores = []
        for _ in range(PERMUTATION_REPEATS):
            shuffled = X.copy()
            shuffled[columns] = X[columns].to_numpy()[rng.permutation(len(X))]
            scores.append(baseline - roc_auc_score(y, model.predict_proba(shuffled)[:, 1]))
        drops[feature] = np.mean(scores)
    return baseline, drops


def cross_validate(data, outcome):
    X, groups = design_matrix(data)
    y = data[outcome].to_numpy()
    rng = np.random.default_rng(SEED)
    aucs, importances = [], []
    for fold, (train, test) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(X, y)):
        for name, model in classifiers().items():
            model.fit(X.iloc[train], y[train])
            auc, drops = grouped_permutation_importance(model, X.iloc[test], y[test], groups, rng)
            aucs.append({'model': name, 'fold': fold, 'auc': auc})
            importances += [{'model': name, 'fold': fold, 'feature': f, 'auc_drop': d} for f, d in drops.items()]

    auc_summary = pd.DataFrame(aucs).groupby('model')['auc'].agg(['mean', 'std']).reset_index()
    importance = (pd.DataFrame(importances).groupby(['model', 'feature'])['auc_drop'].agg(['mean', 'std'])
                  .reset_index())
    importance['rank'] = importance.groupby('model')['mean'].rank(ascending=False).astype(int)
    importance['label'] = importance['feature'].map(LABELS)
    return auc_summary, importance.sort_values(['model', 'rank'])


# ---------- run ----------

def save_design_matrix_sample(features, path):
    # demonstration only: per task, one drifted and one non-drifted model, showing how characteristics become model inputs
    sample = features.groupby(['task_bucket', 'drift'], group_keys=False).apply(lambda g: g.sample(1, random_state=SEED))
    matrix, _ = design_matrix(sample)
    readable = sample[['Model ID', 'drift', 'task_bucket', 'author_type', 'downloads_bucket']]
    readable.join(matrix).to_csv(path, index=False)


def run_scenario(name, features, outcome, exclude_common_voice):
    data = features[features['common_voice_drift'] == 0] if exclude_common_voice else features
    out = os.path.join(RESULTS_DIR, name)
    os.makedirs(out, exist_ok=True)
    print(f'\n=== {name}: {len(data)} models, {int(data[outcome].sum())} with {outcome} = Y ===')

    rates = rate_table(data, outcome)
    tests = univariate_tests(data, outcome)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        odds, fit_stats = logistic_regression(data, outcome)
    vif = vif_table(data)
    auc_summary, importance = cross_validate(data, outcome)

    rates.to_csv(os.path.join(out, 'drift_rates_by_group.csv'), index=False)
    tests.to_csv(os.path.join(out, 'univariate_tests.csv'), index=False)
    odds.to_csv(os.path.join(out, 'logistic_odds_ratios.csv'), index=False)
    vif.to_csv(os.path.join(out, 'vif.csv'), index=False)
    auc_summary.to_csv(os.path.join(out, 'cv_auc.csv'), index=False)
    importance.to_csv(os.path.join(out, 'feature_importance.csv'), index=False)

    print(tests[['feature', 'test', 'effect_size', 'p_holm']].round(4).to_string(index=False))
    print(f"Logistic regression: n={fit_stats['n']}, pseudo R2={fit_stats['pseudo_r2']:.3f}, "
          f"converged={fit_stats['converged']}, max VIF={vif['vif'].max():.1f}")
    print(odds[['label', 'odds_ratio', 'ci_low', 'ci_high', 'p_holm']].round(3).to_string(index=False))
    print(auc_summary.round(3).to_string(index=False))
    return odds.assign(scenario=name), auc_summary.assign(scenario=name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RQ3: correlate model characteristics with drift')
    parser.add_argument('--scenario', choices=list(SCENARIOS), action='append',
                        help='run only these scenarios (default: all four)')
    args = parser.parse_args()

    features = load_features(FEATURES_FILE)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_design_matrix_sample(features, os.path.join(RESULTS_DIR, 'design_matrix_sample_10.csv'))
    all_odds, all_auc = [], []
    for name in args.scenario or SCENARIOS:
        outcome, exclude = SCENARIOS[name]
        odds, auc = run_scenario(name, features, outcome, exclude)
        all_odds.append(odds)
        all_auc.append(auc)

    # side-by-side odds ratios, e.g. to see what changes once the Common Voice models are excluded
    comparison = (pd.concat(all_odds).pivot_table(index='label', columns='scenario', values='odds_ratio')
                  .round(2))
    comparison.to_csv(os.path.join(RESULTS_DIR, 'scenario_odds_ratio_comparison.csv'))
    pd.concat(all_auc).to_csv(os.path.join(RESULTS_DIR, 'scenario_auc_comparison.csv'), index=False)
    print(f'\nSaved results to {RESULTS_DIR}')
