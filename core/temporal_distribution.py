import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import config


def set_sci_plot_style():
    """
    Apply the configured font and figure settings.
    """
    plt.rcParams.update({
        'font.family': config.FONT_FAMILY,
        'axes.titlesize': config.FONT_SIZE_TITLE,
        'axes.labelsize': config.FONT_SIZE_LABEL,
        'xtick.labelsize': config.FONT_SIZE_TICK,
        'ytick.labelsize': config.FONT_SIZE_TICK,
        'legend.fontsize': config.FONT_SIZE_TICK,
        'figure.dpi': config.PLOT_DPI,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in'
    })


def plot_temporal_distributions():
    """
    Plot annual and monthly intensity and grid-day counts.
    """
    if not os.path.exists(config.EXTREMES_CSV_OUT):
        raise FileNotFoundError(f"Event CSV not found: {config.EXTREMES_CSV_OUT}. Run the extraction first.")

    logging.info("Reading extreme-rainfall events.")
    df = pd.read_csv(config.EXTREMES_CSV_OUT)

    if df.empty:
        logging.warning("No events available for temporal plots.")
        return


    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month


    set_sci_plot_style()

    # ==========================================

    # ==========================================
    logging.info("Plotting the annual distribution.")
    annual_counts = df.groupby('Year').size().reset_index(name='Frequency')

    fig, ax1 = plt.subplots(figsize=(10, 6))


    sns.stripplot(x='Year', y='Precipitation_mm', data=df, color='#1f77b4',
                  alpha=0.6, size=6, jitter=0.2, ax=ax1)

    ax1.set_xlabel("Year", fontweight='bold')
    ax1.set_ylabel(f"Daily Precipitation (mm) [≥ {config.PERCENTILE}th]", color='#1f77b4', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')


    years = sorted(df['Year'].unique())
    ax1.set_xticks(range(len(years)))
    ax1.set_xticklabels(years, rotation=45)


    ax2 = ax1.twinx()
    sns.lineplot(x=range(len(years)), y='Frequency', data=annual_counts,
                 color='#d62728', marker='o', linewidth=2, markersize=8, ax=ax2)

    ax2.set_ylabel("Frequency (Days)", color='#d62728', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    ax2.set_ylim(0, annual_counts['Frequency'].max() * 1.2)

    # ==========================================

    # ==========================================
    from matplotlib.lines import Line2D


    scatter_proxy = Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
                           markersize=8, alpha=0.6, label='Extreme Events (Intensity)')


    line_proxy = Line2D([0], [0], color='#d62728', marker='o', linewidth=2,
                        markersize=8, label='Frequency (Days)')


    ax1.legend(handles=[scatter_proxy, line_proxy], loc='upper left',
               frameon=True, fontsize=config.FONT_SIZE_TICK)

    plt.title(f"Annual Distribution of Extreme Rainfall Events", fontweight='bold', pad=15)
    plt.tight_layout()

    fig.savefig(config.FIG_ANNUAL_SCATTER, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved the annual figure to {config.FIG_ANNUAL_SCATTER}")

    # ==========================================

    # ==========================================
    logging.info("Plotting the monthly distribution.")


    monthly_counts = df.groupby('Month').size().reindex(range(1, 13), fill_value=0).reset_index()
    monthly_counts.columns = ['Month', 'Frequency']


    month_order = list(range(1, 13))

    fig, ax1 = plt.subplots(figsize=(8, 6))


    sns.stripplot(x='Month', y='Precipitation_mm', data=df, color='#2ca02c',
                  alpha=0.6, size=6, jitter=0.15, ax=ax1, order=month_order)

    ax1.set_xlabel("Month", fontweight='bold')
    ax1.set_ylabel(f"Daily Precipitation (mm) [≥ {config.PERCENTILE}th]", color='#2ca02c', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#2ca02c')

    ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])


    ax2 = ax1.twinx()
    sns.barplot(x='Month', y='Frequency', data=monthly_counts,
                color='#ff7f0e', alpha=0.3, ax=ax2, order=month_order)

    ax2.set_ylabel("Total Frequency (Days)", color='#ff7f0e', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    ax2.yaxis.get_major_locator().set_params(integer=True)

    # ==========================================

    # ==========================================
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch


    scatter_proxy = Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
                           markersize=8, alpha=0.6, label='Extreme Events (Intensity)')


    bar_proxy = Patch(facecolor='#ff7f0e', alpha=0.3, edgecolor='none',
                      label='Total Frequency (Days)')


    ax1.legend(handles=[scatter_proxy, bar_proxy], loc='upper left',
               frameon=True, fontsize=config.FONT_SIZE_TICK)

    plt.title(f"Monthly Distribution of Extreme Rainfall Events", fontweight='bold', pad=15)
    plt.tight_layout()

    fig.savefig(config.FIG_MONTHLY_SCATTER, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved the monthly figure to {config.FIG_MONTHLY_SCATTER}")
