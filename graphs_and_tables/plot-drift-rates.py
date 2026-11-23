import argparse
import numpy as np
import matplotlib.pyplot as plt
from chart_style import BLUE, INK, MUTED, GRID, SURFACE, add_variant_arguments, selected_variants, read_stats, title, save

# One bar chart per characteristic: % of models that drifted in each group, with 95% confidence intervals.
# Source: drift_correlation/results/<variant>/drift_rates_by_group.csv

FEATURES = ['task_bucket', 'author_type', 'downloads_bucket', 'upload_age',
            'is_permissive', 'has_requirements_file', 'author_owns_dependency', 'base_from_big_org',
            'age_months', 'months_since_last_commit', 'likes', 'n_commits',
            'n_base_models', 'n_datasets', 'n_pinned_libs']
SHORT_NAMES = {'Reinforcement Learning': 'RL', 'Computer Vision': 'CV'}


def plot_feature(rates, feature, variant):
    part = rates[rates['feature'] == feature]
    x = np.arange(len(part))
    pct, low, high = part['rate'] * 100, part['ci_low'] * 100, part['ci_high'] * 100
    overall = part['overall_rate'].iloc[0] * 100

    fig, ax = plt.subplots(figsize=(max(5.5, 1.2 * len(part) + 2), 4.6))
    ax.bar(x, pct, width=0.6, color=BLUE)
    ax.errorbar(x, pct, yerr=[pct - low, high - pct], fmt='none', ecolor=MUTED, elinewidth=1, capsize=3)
    ax.axhline(overall, color=MUTED, linestyle='--', linewidth=1)
    for xi, value, top in zip(x, pct, high):
        ax.text(xi, top + 0.5, f'{value:.1f}%', ha='center', va='bottom', fontsize=9, color=INK,
                bbox={'facecolor': SURFACE, 'edgecolor': 'none', 'pad': 1})

    ax.set_xticks(x, [f'{SHORT_NAMES.get(g, g)}\nn={n:,}' for g, n in zip(part['group'], part['models'])])
    ax.set_ylabel('Drift rate (%)')
    ax.set_ylim(0, max(high.max() * 1.25, 5))
    ax.grid(axis='y', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    title(ax, f'Drift rate by {part["label"].iloc[0].lower()}', variant,
          f'dashed = overall {overall:.1f}%, whiskers = 95% CI')
    save(fig, variant, f'drift_rate_{feature}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Drift rate per group, one chart per characteristic')
    parser.add_argument('--feature', nargs='+', choices=FEATURES + ['all'], default=['all'], metavar='NAME',
                        help='characteristic(s) to plot, or "all" (default). One of: ' + ', '.join(FEATURES))
    add_variant_arguments(parser)
    args = parser.parse_args()

    features = FEATURES if 'all' in args.feature else args.feature
    for variant in selected_variants(args.variant):
        rates = read_stats(variant, 'drift_rates_by_group.csv')
        for feature in features:
            plot_feature(rates, feature, variant)
