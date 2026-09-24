"""Shared print style for every SmartVision AI figure.

A light surface, hairline chrome, thin marks and a validated palette. Colour
carries identity or marks the exception; it never restates a length the chart
already shows.

Categorical slots 1-3 validated on this surface: worst adjacent CVD Delta E 9.2,
worst adjacent normal-vision Delta E 27.6. Slot 3 sits below 3:1 contrast, so
charts using it also ship a table in the report, which is the required relief.
"""

import matplotlib.pyplot as plt

SURFACE = '#fcfcfb'
SERIES_1 = '#2a78d6'    # blue - the default series
SERIES_2 = '#eb6834'    # orange - the exception worth looking at
SERIES_3 = '#1baf7a'    # aqua - third metric

INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRIDLINE = '#e1e0d9'
BASELINE = '#c3c2b7'
FILL_NEUTRAL = '#dfe7f2'    # recessive blue wash for "unremarkable" marks

# Single-hue sequential ramp for magnitude, light to dark.
SEQUENTIAL_BLUE = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']

RC_PARAMS = {
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'axes.edgecolor': BASELINE,
    'axes.linewidth': 0.8,
    'axes.labelcolor': INK_SECONDARY,
    'axes.titlecolor': INK_PRIMARY,
    'axes.titlesize': 10.5,
    'axes.titleweight': 'bold',
    'axes.titlelocation': 'left',
    'axes.titlepad': 8,
    'axes.labelsize': 9.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.axisbelow': True,
    'grid.color': GRIDLINE,
    'grid.linestyle': '-',
    'grid.linewidth': 0.7,
    'xtick.color': INK_MUTED,
    'ytick.color': INK_MUTED,
    'xtick.labelcolor': INK_SECONDARY,
    'ytick.labelcolor': INK_SECONDARY,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'xtick.major.size': 0,
    'ytick.major.size': 0,
    'text.color': INK_PRIMARY,
    'legend.frameon': False,
    'legend.fontsize': 8.5,
    'legend.labelcolor': INK_SECONDARY,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.25,
}


def apply() -> None:
    plt.rcParams.update(RC_PARAMS)


def headline(fig, title: str, subtitle: str = '') -> None:
    """Title and an explanatory deck, both left-aligned above the plots.

    The deck states the finding. A reader who looks at nothing else should
    still leave with the right conclusion.
    """
    fig.text(0.0, 1.0, title, ha='left', va='bottom',
             fontsize=12.5, fontweight='bold', color=INK_PRIMARY)
    if subtitle:
        fig.text(0.0, 0.995, subtitle, ha='left', va='top',
                 fontsize=9.5, color=INK_SECONDARY)


def note(fig, text: str) -> None:
    """A muted source note below the figure."""
    fig.text(0.0, -0.01, text, ha='left', va='top', fontsize=8, color=INK_MUTED)


def value_axis(ax, axis: str = 'y') -> None:
    """Grid on the value axis only; the category axis carries no rules.

    `visible` is passed explicitly because Axes.grid() toggles when it is
    omitted, which would switch off the grid the rcParams just switched on.
    """
    ax.grid(visible=True, axis=axis, which='major')
    ax.grid(visible=False, axis='x' if axis == 'y' else 'y')


def strip_frame(ax) -> None:
    """Remove every spine and tick - for schematics and flow diagrams."""
    ax.grid(visible=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
