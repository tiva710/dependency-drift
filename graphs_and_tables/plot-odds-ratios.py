import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from chart_style import BLUE, GRAY, MUTED, GRID, SURFACE, add_variant_arguments, selected_variants, read_stats, title, save

# Logistic regression odds ratios with 95% confidence intervals (all characteristics together).
# Blue = significant after Holm correction, gray = not significant.
# Source: drift_correlation/results/<variant>/logistic_odds_ratios.csv


def plot(odds, variant):
    odds = odds.sort_values('odds_ratio')
    y = np.arange(len(odds))
    colors = [BLUE if p < 0.05 else GRAY for p in odds['p_holm']]

    fig, ax = plt.subplots(figsize=(9, 0.38 * len(odds) + 1.6))
    ax.hlines(y, odds['ci_low'], odds['ci_high'], colors=colors, linewidth=2)
    ax.scatter(odds['odds_ratio'], y, color=colors, s=40, zorder=3, edgecolors=SURFACE, linewidths=1.5)
    ax.axvline(1, color=MUTED, linestyle='--', linewidth=1)
    ax.set_xscale('log')
    ticks = [t for t in [0.1, 0.25, 0.5, 1, 2, 4, 8] if odds['ci_low'].min() / 1.5 <= t <= odds['ci_high'].max() * 1.5]
    ax.set_xticks(ticks, [f'{t:g}' for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks(y, odds['label'])
    ax.set_xlabel('Odds ratio (log scale)   < 1 less drift  |  > 1 more drift')
    ax.grid(axis='x', color=GRID, linewidth=0.6)
    ax.legend(handles=[plt.Line2D([], [], color=BLUE, marker='o', label='significant (Holm)'),
                       plt.Line2D([], [], color=GRAY, marker='o', label='not significant')],
              loc='lower right', frameon=False, fontsize=9)
    title(ax, 'Odds of drift, all characteristics together', variant, 'logistic regression, lines = 95% CI')
    save(fig, variant, 'odds_ratios.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Odds-ratio chart from the logistic regression')
    add_variant_arguments(parser)
    args = parser.parse_args()
    for variant in selected_variants(args.variant):
        plot(read_stats(variant, 'logistic_odds_ratios.csv'), variant)
