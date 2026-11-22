import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Shared paths, variants and styling for the RQ3 charts.
# The charts only read the CSVs written by drift_correlation/correlation-analysis.py; run that first.

GRAPHS_DIR = os.path.dirname(os.path.abspath(__file__))
STATS_DIR = os.path.join(GRAPHS_DIR, '..', 'drift_correlation', 'results')
OUTPUT_DIR = os.path.join(GRAPHS_DIR, 'output')

VARIANTS = {
    'drift_all': 'All models',
    'drift_no_common_voice': 'Common Voice models excluded',
}

# reference palette: one blue series, gray for "not significant"
BLUE, GRAY, INK, MUTED, GRID, SURFACE = '#2a78d6', '#a3a29d', '#0b0b0b', '#52514e', '#e5e4e0', '#fcfcfb'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'text.color': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'figure.facecolor': SURFACE,
                     'axes.facecolor': SURFACE, 'axes.spines.top': False, 'axes.spines.right': False})


def add_variant_arguments(parser):
    parser.add_argument('--variant', nargs='+', choices=list(VARIANTS) + ['all'], default=['all'], metavar='NAME',
                        help='variant(s) to plot, or "all" (default). One of: ' + ', '.join(VARIANTS))


def selected_variants(choice):
    return list(VARIANTS) if 'all' in choice else choice


def read_stats(variant, file_name):
    path = os.path.join(STATS_DIR, variant, file_name)
    if not os.path.exists(path):
        raise SystemExit(f'Missing {path}. Run drift_correlation/correlation-analysis.py first.')
    return pd.read_csv(path)


def title(ax, heading, variant, note=''):
    # line 1: what the chart shows; line 2: which variant, plus a short reading note
    subtitle = VARIANTS[variant] + (f'\n{note}' if note else '')
    ax.set_title(heading, loc='left', fontsize=11, color=INK, pad=34 if note else 22)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=MUTED, va='bottom')


def save(fig, variant, file_name):
    out_dir = os.path.join(OUTPUT_DIR, variant)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, file_name)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f'Saved {os.path.relpath(path)}')
