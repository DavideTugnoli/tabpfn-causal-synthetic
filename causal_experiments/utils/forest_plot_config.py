"""
Configuration for forest plots.
Centralizes font sizes, layout constants, and spacing to ensure consistency
between interventional and new metric experiments.
"""

# Font sizes tuned for publication-ready figures with LaTeX Times font
# Increased sizes for better readability in paper
TITLE_FONT_SIZE = 20  # Increased from 18
LABEL_FONT_SIZE = 18  # Increased from 16
TICK_FONT_SIZE = 17  # Increased from 15
LEGEND_FONT_SIZE = 16  # Increased from 14
LEGEND_TITLE_FONT_SIZE = 16  # Increased from 14
CAPTION_FONT_SIZE = 15  # Increased from 13

# Title font sizes for different plot types
TITLE_FONT_SIZE_SINGLE = TITLE_FONT_SIZE  # 20 - for single plot titles
TITLE_FONT_SIZE_SUBPLOT_COMBINED = TITLE_FONT_SIZE_SINGLE + 10  # 30 - for subplot titles in combined plots
TITLE_FONT_SIZE_SUPTITLE_COMBINED = TITLE_FONT_SIZE_SINGLE + 12  # 32 - for main title (suptitle) in combined plots

# Tick font sizes for different plot types
# Combined font size ratio: 28/18 = 1.56 (used for proportional spacing)
TICK_FONT_SIZE_SINGLE = TITLE_FONT_SIZE - 2  # 18 - for single plots
TICK_FONT_SIZE_SINGLE_ENHANCED = TITLE_FONT_SIZE  # 20 - for single plots with enhanced visibility
TICK_FONT_SIZE_COMBINED = TICK_FONT_SIZE_SINGLE + 10  # 28 - for combined plots

# Label font sizes for different plot types
LABEL_FONT_SIZE_SINGLE = TITLE_FONT_SIZE - 2  # 18 - for single plot labels
LABEL_FONT_SIZE_COMBINED = LABEL_FONT_SIZE_SINGLE + 10  # 28 - for labels in combined plots

# Legend font sizes for different plot types
LEGEND_FONT_SIZE_SINGLE = LEGEND_FONT_SIZE  # 16 - for single plot legends
LEGEND_FONT_SIZE_COMBINED = LEGEND_FONT_SIZE_SINGLE + 10  # 26 - for legends in combined plots

# Legend title font sizes for different plot types
LEGEND_TITLE_FONT_SIZE_SINGLE = LEGEND_TITLE_FONT_SIZE  # 16 - for single plot legend titles
LEGEND_TITLE_FONT_SIZE_COMBINED = LEGEND_TITLE_FONT_SIZE_SINGLE + 10  # 26 - for legend titles in combined plots

# Caption font sizes
CAPTION_FONT_SIZE_SINGLE = CAPTION_FONT_SIZE  # 15 - for single plot captions
CAPTION_FONT_SIZE_ENHANCED = CAPTION_FONT_SIZE + 2  # 17 - for enhanced captions

# Tick label padding
TICK_PAD_Y = 10  # Padding for y-axis tick labels
TICK_PAD_X = 20  # Padding for x-axis tick labels
TICK_PAD_X_SINGLE = 8  # Reduced padding for x-axis tick labels in single plots (to match Y label distance)
# Proportional to single: combined = single * (28/18) = single * 1.56
TICK_PAD_X_COMBINED = 12  # Reduced padding for x-axis tick labels in combined plots (proportional to TICK_PAD_X_SINGLE)

# Marker sizes
MARKER_SIZE_PLOT = 5  # Size of markers in plots
MARKER_SIZE_LEGEND_SINGLE = 10  # Size of markers in legend for single plots
MARKER_SIZE_LEGEND_COMBINED = 10  # Size of markers in legend for combined plots

# Error bar styling
ERROR_BAR_LINEWIDTH = 1.0  # Line width for error bars
ERROR_BAR_CAPSIZE = 2.5  # Cap size for error bars
MARKER_EDGEWIDTH = 1.0  # Edge width for plot markers
MARKER_EDGEWIDTH_LEGEND = 0.8  # Edge width for legend markers

# Legend spacing
LEGEND_COLUMNSPACING_SINGLE = 1.4  # Column spacing in legend for single plots
LEGEND_COLUMNSPACING_COMBINED = 1.8  # Column spacing in legend for combined plots
LEGEND_HANDLETEXTPAD = 0.15  # Spacing between legend handle and text

# Layout constants
AXES_TITLE_PAD = 22  # Padding between title and axes
SUPTITLE_Y = 0.96  # Positioned above the plot area
X_LABEL_PAD = 8  # Fixed padding for x-axis label

# Single-figure spacing tweaks - FIXED values for uniform plot dimensions
SINGLE_BOTTOM_MARGIN = 0.20  # Fixed bottom margin for all single plots
SINGLE_TOP_MARGIN = 0.90  # Fixed top margin for all single plots
SINGLE_LEFT_MARGIN = 0.12  # Fixed left margin
SINGLE_RIGHT_MARGIN = 0.98  # Fixed right margin

# Combined-figure spacing tweaks
# Proportional to single plots: font ratio = 28/18 = 1.56
COMBINED_TITLE_PAD = 12  # Padding between subplot title and axes
COMBINED_SHARED_XLABEL_Y = 0.015  # Shared x-axis label position (well below x-tick labels) - lowered slightly
COMBINED_LEGEND_Y = 0.00  # Legend position (in figure coords, near bottom)
COMBINED_BOTTOM_MARGIN = 0.32  # Bottom margin with legend - space for tick labels + xlabel (reduced for compactness)
COMBINED_BOTTOM_MARGIN_NO_LEGEND = 0.16  # Bottom margin without legend (reduced for compactness)
COMBINED_WSPACE = 0.50  # Horizontal spacing between subplots
COMBINED_WSPACE_DUO = 0.30  # Horizontal spacing for double plots
COMBINED_CAPTION_Y = 0.10  # Caption position
COMBINED_SUPTITLE_Y = 1.02  # Main title position (reduced for compactness)
COMBINED_TOP_MARGIN = 0.82  # Top of subplot area - leave room for titles (reduced for compactness)

# Tick label offsets (in points - absolute units)
# Negative values move labels to the left, away from the y-axis spine
# Proportional: combined = single * (28/18) = single * 1.56
Y_TICK_LABEL_OFFSET_SINGLE = -50.0  # points (font size 18)
Y_TICK_LABEL_OFFSET_COMBINED = -78.0  # points (font size 28, = 50 * 1.56)

# Dataset vertical spacing multiplier (1.0 = no extra space, >1.0 = more space between datasets)
DATASET_SPACING = 1.22

# Extra padding (points) to keep the y-axis label clear of the left-aligned acronyms
# Proportional: combined = single * (28/18) = single * 1.56
Y_LABEL_EXTRA_PAD = 38  # points for single plots (font size 18) - reduced to bring label closer to dataset names
Y_LABEL_EXTRA_PAD_COMBINED = 70  # points for combined plots (= 45 * 1.56)
