"""
Visualization configuration for causal experiments.
Based on the old plots configuration with fastplot integration.
"""

import matplotlib as mpl
import seaborn as sns
import platform
import warnings
warnings.filterwarnings('ignore')

# Font and display settings
FONT_SIZES = {
    'title': 11,
    'label': 10,
    'legend': 9,
    'tick': 10,
    'significance': 10,
    'suptitle': 12
}

DPI = 300
FIG_SIZE_TRIPLE = (15, 4)

# Centralized metrics configuration with scale management
METRIC_CONFIG = {
    'mean_corr_difference': {
        'title': 'Mean Correlation Difference',
        'slug': 'mean_corr_diff',
        'ylim': (0, 1),
        'scale': 'linear'
    },
    'max_corr_difference': {
        'title': 'Max Correlation Difference',
        'slug': 'max_corr_diff',
        'ylim': (0, 1),
        'scale': 'linear'
    },
    'propensity_metrics_avg_pMSE': {
        'title': 'Propensity MSE',
        'slug': 'propensity_mse',
        'ylim': 'auto',
        'scale': 'linear'
    },
    'propensity_avg_pMSE': {
        'title': 'Propensity Avg pMSE',
        'slug': 'propensity_avg_pmse',
        'ylim': 'auto_tight',
        'scale': 'linear'
    },
    'propensity_pMSE_err': {
        'title': 'Propensity pMSE Error',
        'slug': 'propensity_pmse_err',
        'ylim': 'auto_tight',
        'scale': 'linear'
    },
    'propensity_metrics_avg_acc': {
        'title': 'Propensity Accuracy',
        'slug': 'propensity_acc',
        'ylim': (0, 1),
        'scale': 'linear'
    },
    'propensity_metrics_pMSE_err': {
        'title': 'Propensity MSE Error',
        'slug': 'propensity_mse_err',
        'ylim': 'auto',
        'scale': 'linear'
    },
    'propensity_metrics_acc_err': {
        'title': 'Propensity Accuracy Error',
        'slug': 'propensity_acc_err',
        'ylim': 'auto',
        'scale': 'linear'
    },
    'k_marginal_tvd': {
        'title': 'k-Marginal TVD',
        'slug': '2marginal',
        'ylim': 'auto_tight',
        'scale': 'linear',
        'long_title': 'k-Marginal Total Variation Distance',
        'acronym': 'TVD',
        'short_title': 'kMTVD'
    },
    'k_marginal_tvd_quantile12_laplace': {
        'title': '2-Marginal TVD (B=12, Laplace 0.5)',
        'slug': '2marginal_quantile12_laplace',
        'ylim': 'auto_tight',
        'scale': 'linear',
        'long_title': '2-Marginal Total Variation Distance (B=12, Laplace 0.5)',
        'acronym': 'TVD',
        'short_title': 'kMTVD (B=12 Laplace 0.5)'
    },
    'k_marginal_tvd_quantile20_smoothed': {
        'title': '2-Marginal TVD (B=20, Smoothed)',
        'slug': '2marginal_quantile20_smoothed',
        'ylim': 'auto_tight',
        'scale': 'linear',
        'long_title': '2-Marginal Total Variation Distance (B=20, Smoothed)',
        'acronym': 'TVD',
        'short_title': 'kMTVD (B=20 Smoothed)'
    },
    'k_marginal_tvd_multiresolution': {
        'title': '2-Marginal TVD (Multi-resolution)',
        'slug': '2marginal_multires',
        'ylim': 'auto_tight',
        'scale': 'linear',
        'long_title': '2-Marginal Total Variation Distance (Multi-resolution)',
        'acronym': 'TVD',
        'short_title': 'kMTVD (k=2, multi-res)'
    },
    'mi_matrix_difference': {
        'title': 'MI Matrix Difference',
        'slug': 'mi_diff',
        'ylim': 'auto_tight',
        'scale': 'linear'
    },
    'mi_matrix_native': {
        'title': 'MI Matrix Native',
        'slug': 'mi_diff_native',
        'ylim': 'auto_tight',
        'scale': 'linear'
    },
    'mean_ks_distance': {
        'title': 'Mean KS Distance',
        'slug': 'ks_distance',
        'ylim': (0, 1),
        'scale': 'linear'
    },
    'ks_significant_count': {
        'title': 'KS Significant Count',
        'slug': 'ks_sig_count',
        'ylim': 'auto_int',
        'scale': 'linear'
    },
    'ks_significant_percentage': {
        'title': 'KS Significant Percentage (%)',
        'slug': 'ks_sig_perc',
        'ylim': (0, 100),
        'scale': 'linear'
    },
    'nnaa': {
        'title': 'NNAA',
        'slug': 'nnaa',
        'ylim': (0.4, 0.7),
        'scale': 'linear',
        'optimal_value': 0.5,
        'long_title': 'Nearest-Neighbor Adversarial Accuracy',
        'acronym': 'NNAA',
        'short_title': 'NNAA'
    },
    'correlation_matrix_difference': {
        'title': 'Correlation Matrix Difference',
        'slug': 'correlation_matrix_difference',
        'ylim': 'auto_tight',
        'scale': 'linear',
        'long_title': 'Correlation Matrix Difference',
        'acronym': 'CMD',
        'short_title': 'CMD'
    },
    # Legacy alias for backward compatibility
    'frobenius_corr_norm': {
        'title': 'Correlation Matrix Difference',
        'slug': 'correlation_matrix_difference',
        'ylim': 'auto_tight',
        'scale': 'linear',
        'long_title': 'Correlation Matrix Difference',
        'acronym': 'CMD',
        'short_title': 'CMD'
    },
    'frobenius_spearman_norm': {
        'title': 'Correlation Matrix Difference (Spearman)',
        'slug': 'frobenius_spearman_norm',
        'ylim': 'auto_tight',
        'scale': 'linear'
    },
    'ate_synthetic': {
        'title': 'Estimated ATE',
        'slug': 'ate_estimate',
        'ylim': 'auto_tight',
        'scale': 'linear'
    },
    'ate_difference': {
        'title': 'ATE Difference (Estimate - Ground Truth)',
        'slug': 'ate_difference',
        'ylim': 'auto_tight',
        'scale': 'linear'
    },
    'ate_relative_error': {
        'title': 'ATE Relative Error (%)',
        'slug': 'ate_relative_error',
        'ylim': 'auto_tight',
        'scale': 'linear'
    }
}

# Strategy ordering for consistency
STRATEGY_ORDER = ['original', 'topological', 'worst', 'random']

# Standard seaborn palette for consistent colors
SEABORN_PALETTE = sns.color_palette("Set2", 8)

# Color mapping for column order strategies (Experiment 2)
STRATEGY_COLORS = {
    'original': SEABORN_PALETTE[0],    # Blue
    'topological': SEABORN_PALETTE[1], # Orange  
    'worst': SEABORN_PALETTE[2],       # Green
    'random': SEABORN_PALETTE[3]       # Red
}

# Color mapping for DAG types (Experiment 3)
DAG_TYPE_COLORS = {
    'vanilla': SEABORN_PALETTE[0],       # Blue - baseline
    'dag': SEABORN_PALETTE[1],           # Orange - correct DAG
    'cpdag': SEABORN_PALETTE[2],         # Green - CPDAG
    'wrong_parents': SEABORN_PALETTE[3], # Red - problematic
    'missing_edges': SEABORN_PALETTE[4], # Purple - missing edges
    'extra_edges': SEABORN_PALETTE[5],   # Brown - extra edges
    'disconnected': SEABORN_PALETTE[6]   # Pink - disconnected
}

# Color mapping for causal vs non-causal (Experiment 1)
CAUSAL_COLORS = {
    'with_dag': SEABORN_PALETTE[1],     # Orange - with causal structure
    'without_dag': SEABORN_PALETTE[0]   # Blue - without causal structure
}

# Heatmap configuration
HEATMAP_CONFIG = {
    'annot': True,
    'fmt': '.3f',
    'linewidths': 0.5,
    'linecolor': 'black',
    'square': False,
    'cmap': 'RdYlBu_r',  # Red=high values (worse), Blue=low values (better)
    'cbar': True,
    'annot_kws': {'size': 9}
}

def setup_plotting():
    """Setup matplotlib for publication-quality plots."""
    # Reset to defaults first
    mpl.rcParams.update(mpl.rcParamsDefault)
    
    # Detect system and configure fonts
    system = platform.system()
    
    font_config = {
        'font.family': 'serif',
        'font.serif': [
            'Times New Roman',
            'Times',
            'Nimbus Roman',
            'serif'
        ],
        'font.sans-serif': [
            'Helvetica',
            'Arial',
            'DejaVu Sans',
            'sans-serif'
        ],
        'font.size': FONT_SIZES['tick'],
        # When LaTeX is active, mathtext settings are ignored; keep for fallback.
        'mathtext.fontset': 'stix',
        'mathtext.default': 'regular',
        # Use LaTeX text rendering so the plot matches \usepackage{times}
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{times}',
    }

    mpl.rcParams.update(font_config)
    
    # Academic style
    academic_style = {
        'figure.figsize': (6, 4),
        'figure.dpi': 100,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'savefig.format': 'pdf',
        'axes.linewidth': 1.0,
        'axes.edgecolor': 'black',
        'axes.axisbelow': True,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.labelsize': FONT_SIZES['label'],
        'axes.titlesize': FONT_SIZES['title'],
        'grid.color': 'gray',
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
        'xtick.labelsize': FONT_SIZES['tick'],
        'ytick.labelsize': FONT_SIZES['tick'],
        'xtick.color': 'black',
        'ytick.color': 'black',
        'legend.fontsize': FONT_SIZES['legend'],
        'legend.frameon': False,
        'legend.fancybox': False,
        'legend.shadow': False,
        'legend.framealpha': 1.0,
        'legend.edgecolor': 'black',
        'legend.facecolor': 'white',
        'legend.borderpad': 0.4,
        'legend.columnspacing': 2.0,
        'legend.handlelength': 2.0,
        'legend.handletextpad': 0.8,
        'legend.labelspacing': 0.5,
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        'lines.markeredgewidth': 0.5,
        'patch.linewidth': 1.0,
        'patch.edgecolor': 'black',
        'errorbar.capsize': 3,
        'text.color': 'black',
    }
    mpl.rcParams.update(academic_style)
    
    # Seaborn style
    sns.set_style("whitegrid", {
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0,
        'grid.color': 'gray',
        'grid.alpha': 0.3,
        'axes.grid.axis': 'y',
        'axes.grid': True
    })
    
    # Set the standard seaborn palette
    sns.set_palette(SEABORN_PALETTE)
    
    print(f"[INFO] Academic plotting style configured for {system}")

def get_significance_marker(p_value: float):
    """Return significance marker based on p-value."""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'

def apply_metric_scale(ax, metric: str, data_values=None):
    """Apply appropriate Y-axis scaling for a metric."""
    if metric not in METRIC_CONFIG:
        return
    
    import numpy as np
    import matplotlib.ticker as ticker
    import matplotlib.pyplot as plt
    
    config = METRIC_CONFIG[metric]
    scale = config.get('scale', 'linear')
    ylim = config.get('ylim', 'auto')
    optimal_value = config.get('optimal_value', None)

    data_array = None
    valid_data = None
    if data_values is not None:
        data_array = np.array(data_values)
        error_values = [-1.0, -1]
        valid_mask = ~np.isin(data_array, error_values)
        if valid_mask.any():
            valid_data = data_array[valid_mask]
        else:
            valid_data = None

    # Set scale
    ax.set_yscale(scale)

    custom_ylim_handled = False
    if metric == 'nnaa' and valid_data is not None:
        base_min, base_max = config.get('ylim', (0.4, 0.7))
        data_min = float(np.min(valid_data))
        data_max = float(np.max(valid_data))
        data_range = max(data_max - data_min, 1e-6)
        padding = max(0.02, data_range * 0.15)
        ymin = min(base_min, data_min - padding)
        ymax = max(base_max, data_max + padding)
        ymin = max(0.0, ymin)
        if ymax - ymin < 0.05:
            center = (ymax + ymin) / 2
            ymin = max(0.0, center - 0.025)
            ymax = center + 0.025
        ax.set_ylim(ymin, ymax)
        custom_ylim_handled = True

    # Set Y limits
    if not custom_ylim_handled:
        if ylim == 'auto':
            # Let matplotlib handle it automatically
            pass
        elif ylim == 'auto_tight':
            # Aggressive tight scaling optimized for boxplot visibility
            if data_values is not None:
                data_values = np.array(data_values)

                # Filter out error values (-1.0) that skew the scale
                error_values = [-1.0, -1]  # Common error indicators
                valid_data = data_values[~np.isin(data_values, error_values)]

                if len(valid_data) == 0:
                    # All data is error values, use default scaling
                    return

                # Use boxplot quartiles for better scaling (only valid data)
                q1 = np.percentile(valid_data, 25)
                q3 = np.percentile(valid_data, 75)
                median = np.percentile(valid_data, 50)
                data_min = np.min(valid_data)
                data_max = np.max(valid_data)

                # Calculate IQR for dynamic padding
                iqr = q3 - q1

                # Use more aggressive padding based on data distribution
                if iqr > 0:
                    # Use IQR-based padding (more responsive to data spread)
                    padding = max(iqr * 0.5, (data_max - data_min) * 0.15)
                else:
                    # For tight data, use fixed percentage
                    data_range = data_max - data_min
                    padding = max(data_range * 0.3, 0.01)

                # Set limits with generous padding for boxplot visibility
                ymin = max(0, data_min - padding)  # Don't go below 0 for positive metrics
                ymax = data_max + padding

                # Ensure minimum visual range for very tight data
                if (ymax - ymin) < 0.02:
                    center = (ymax + ymin) / 2
                    ymin = max(0, center - 0.01)
                    ymax = center + 0.01

                ax.set_ylim(ymin, ymax)
        elif ylim == 'auto_int':
            # Auto scale but ensure integer ticks for count data
            if data_values is not None:
                ymin = 0
                ymax = int(np.ceil(np.max(data_values))) + 1
                ax.set_ylim(ymin, ymax)
                # Force integer ticks
                ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        elif isinstance(ylim, tuple):
            # Fixed Y limits
            ax.set_ylim(ylim)
    
    # Improve tick formatting and positioning for small values  
    if scale == 'linear':
        # Get current y limits to decide formatting
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        
        # Set intelligent tick locations based on data (excluding error values)
        if data_values is not None:
            data_values = np.array(data_values)
            
            # Filter out error values (-1.0) for tick calculation too
            error_values = [-1.0, -1]
            valid_data = data_values[~np.isin(data_values, error_values)]
            
            if len(valid_data) == 0:
                return  # No valid data for ticks
                
            data_min = np.min(valid_data)
            data_max = np.max(valid_data)
            data_median = np.median(valid_data)
            
            # Create smart tick locations around the actual data
            if y_range < 0.01:
                # Very tight range - 5 ticks around data
                ticks = np.linspace(y_min, y_max, 5)
                ax.set_yticks(ticks)
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
            elif y_range < 0.1:
                # Small range - 6 ticks with data-centered positioning
                ticks = np.linspace(y_min, y_max, 6) 
                ax.set_yticks(ticks)
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
            elif y_range < 1:
                # Medium range - 5-7 ticks
                ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            else:
                # Large range - standard ticking
                ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    
    # Special handling for NNAA: add horizontal line at optimal value and adaptive ticks
    if metric == 'nnaa':
        y_min, y_max = ax.get_ylim()
        if optimal_value is not None and y_min <= optimal_value <= y_max:
            ax.axhline(y=optimal_value, color='green', linestyle='--', alpha=0.7, linewidth=1.5)

        tick_span = y_max - y_min
        if tick_span <= 0:
            tick_span = 0.1

        tick_step = 0.05 if tick_span <= 0.6 else 0.1
        start = np.floor((y_min - 1e-9) / tick_step) * tick_step
        start = max(0.0, start)
        end = np.ceil((y_max + 1e-9) / tick_step) * tick_step
        ticks = np.arange(start, end + tick_step * 0.5, tick_step)

        if optimal_value is not None:
            ticks = np.union1d(ticks, [optimal_value])

        ticks = np.round(ticks, 3)
        ax.set_yticks(ticks.tolist())
