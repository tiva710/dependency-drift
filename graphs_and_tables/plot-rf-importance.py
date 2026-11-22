import argparse
import matplotlib.pyplot as plt
from chart_style import BLUE, MUTED, GRID, add_variant_arguments, selected_variants, read_stats, title, save

# Random forest feature importance: how much the prediction score (AUC) drops when one characteristic is shuffled.
# Sources: drift_correlation/results/<variant>/feature_importance.csv and cv_auc.csv


def plot(importance, auc, variant):
    forest = importance[importance['model'] == 'random_forest'].sort_values('mean')
    auc = auc.set_index('model')['mean']

    fig, ax = plt.subplots(figsize=(8, 0.38 * len(forest) + 1.6))
    ax.barh(forest['label'], forest['mean'], xerr=forest['std'], color=BLUE, height=0.6,
            error_kw={'ecolor': MUTED, 'elinewidth': 1, 'capsize': 2})
    ax.set_xlabel('Drop in AUC when the characteristic is shuffled')
    ax.grid(axis='x', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    title(ax, 'Random forest feature importance', variant,
          f'AUC: random forest {auc["random_forest"]:.2f}, logistic regression {auc["logistic_regression"]:.2f}')
    save(fig, variant, 'rf_importance.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random forest permutation importance chart')
    add_variant_arguments(parser)
    args = parser.parse_args()
    for variant in selected_variants(args.variant):
        plot(read_stats(variant, 'feature_importance.csv'), read_stats(variant, 'cv_auc.csv'), variant)
