import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from chart_style import BLUE, GRAY, INK, MUTED, GRID, add_variant_arguments, selected_variants, read_stats, title, save

# Effect size of each characteristic tested on its own (chi-square -> Cramer's V, Mann-Whitney -> rank-biserial r).
# Bar length = strength of the relationship; blue = significant after Holm correction, gray = not significant.
# (-) after a name = drifted models have LOWER values of that characteristic.
# Source: drift_correlation/results/<variant>/univariate_tests.csv

SMALL, MEDIUM = 0.1, 0.3   # common rule-of-thumb thresholds


def plot(tests, variant):
    tests = tests.assign(strength=tests['effect_size'].abs()).sort_values('strength')
    names = [f'{label} (-)' if effect < 0 else label for label, effect in zip(tests['label'], tests['effect_size'])]
    colors = [BLUE if significant else GRAY for significant in tests['significant']]
    right = max(MEDIUM + 0.05, tests['strength'].max() * 1.2)

    fig, ax = plt.subplots(figsize=(8, 0.38 * len(tests) + 1.8))
    ax.barh(names, tests['strength'], color=colors, height=0.6)
    for y, value in enumerate(tests['strength']):
        ax.text(value + right * 0.01, y, f'{value:.2f}', va='center', fontsize=8, color=INK)
    for threshold, word in [(SMALL, 'small'), (MEDIUM, 'medium')]:
        ax.axvline(threshold, color=MUTED, linestyle=':', linewidth=1)
        ax.text(threshold, len(tests) - 0.4, f' {word}', fontsize=8, color=MUTED, va='bottom')
    ax.set_xlim(0, right)
    ax.set_xlabel("Effect size (Cramer's V for categories, |rank-biserial r| for numbers)")
    ax.grid(axis='x', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color=BLUE, label='significant (Holm)'), Patch(color=GRAY, label='not significant')],
              loc='lower right', frameon=False, fontsize=9)
    title(ax, 'Effect size, one characteristic at a time', variant, '(-) = lower in drifted models')
    save(fig, variant, 'effect_sizes.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Effect-size chart from the univariate tests')
    add_variant_arguments(parser)
    args = parser.parse_args()
    for variant in selected_variants(args.variant):
        plot(read_stats(variant, 'univariate_tests.csv'), variant)
