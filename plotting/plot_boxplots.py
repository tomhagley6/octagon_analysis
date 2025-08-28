# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Modified version of the function with an option to suppress returns
def plot_boxplot(data_list, label_list, ylabel, xlabel="",
                ylim=(0.0,1), set_aspect=3, fontsize=24, custom_colors=None, color_offset=0, 
                return_data=False):
    ''' 
    Helper function to create boxplots for multiple datasets with optional return values.
    
    Parameters:
    -----------
    data_list : list of arrays
        List containing arrays of data to plot
    label_list : list of strings
        List containing labels for each dataset
    ylabel : str
        Label for y-axis
    xlabel : str
        Label for x-axis (optional)
    ylim : tuple
        Y-axis limits (default: (0.0, 1))
    set_aspect : float
        Aspect ratio (default: 3)
    fontsize : int
        Font size for labels (default: 24)
    custom_colors : list or None
        Optional list of colors to use. If None, uses Paired palette.
    color_offset : int
        Offset to start color selection from the palette (default: 0)
    return_data : bool
        Whether to return the axes, dataframe and data_list (default: False)
    '''
    
    if len(data_list) != len(label_list):
        raise ValueError("The number of datasets must match the number of labels")
    
    # Prepare data
    all_data = []
    all_labels = []
    
    # create labels for each dataset based on the ravelled size of the data
    # ravel to flatten arrays to adapt to arbitrary original shapes
    for data, label in zip(data_list, label_list):
        all_data.append(data.ravel())
        all_labels.append(np.full(data.ravel().size, label))
    
    # Concatenate all data and labels
    data = np.concatenate(all_data)
    labels = np.concatenate(all_labels)

    # Create DataFrame for Seaborn
    df = pd.DataFrame({
        "Probability": data,
        "Condition": labels
    })
    
    # Create color palette
    if custom_colors is None:
        paired = sns.color_palette("Paired")
        # Use as many colors as needed from the palette
        # using modulo here so that if there are more datasets than colours (12), 
        # it will wrap around and return to the beginning
        # Added color_offset to start from a specific position in the palette
        custom_palette = [paired[(i + color_offset) % len(paired)] for i in range(len(data_list))]
    else:
        # Use provided colors
        custom_palette = custom_colors

    # Create figure with larger size for poster visibility
    plt.figure(figsize=(10, 8))
    
    # Create boxplot with custom flier (outlier) properties - 'o' and 'x' overlapped with low alpha
    ax = sns.boxplot(x="Condition", y="Probability", data=df, palette=custom_palette, width=.8, 
                     showmeans=False, showfliers=True, 
                     flierprops=dict(markerfacecolor='none', marker='o', markersize=8, 
                                     markeredgecolor='black', alpha=1))
    
    # Add 'x' markers on top of the 'o' markers for outliers
    for flier in ax.findobj(plt.Line2D):
        if flier.get_marker() == 'o' and flier.get_alpha() == 1.0:
            # Get the positions of the outliers
            xdata, ydata = flier.get_xdata(), flier.get_ydata()
            # Plot 'x' markers at the same positions
            plt.plot(xdata, ydata, 'x', color='black', markersize=6, alpha=1)
    
    # Increase linewidth for better visibility
    for box in ax.artists:
        box.set_edgecolor('black')
        box.set_linewidth(2)
    
    # Make whiskers and caps thicker
    for whisker in ax.lines:
        whisker.set_linewidth(2)

    # Format plot with increased font sizes for poster visibility
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.xticks(fontsize=fontsize - 2)  # Larger font for x-tick labels
    plt.yticks(fontsize=fontsize - 2)  # Larger font for y-tick labels
    plt.ylim(ylim)  # Set y-axis limits for probabilities
    plt.gca().set_aspect(set_aspect)
    plt.tight_layout()

    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    if return_data:
        return ax, df, data_list
    else:
        return ax  # Only return the axis object by default

# %%
# # Example usage with multiple datasets and color offset
# np.random.seed(42)  # For reproducibility

# # Create some sample data
# solo_data = np.random.beta(8, 2, size=30)
# social_data = np.random.beta(4, 2, size=30)
# control_data = np.random.beta(6, 6, size=30)

# # Create data list and labels list
# data_list = [solo_data, social_data, control_data]
# label_list = ['Solo', 'Competition', 'Control']

# # Example 1: Using color_offset to start from a different position in the palette
# plt.figure(figsize=(15, 5))

# # Create three subplots to demonstrate different color offsets
# plt.subplot(1, 3, 1)
# ax1, _, _ = plot_boxplot(
#     data_list=data_list, 
#     label_list=label_list, 
#     ylabel="Probability", 
#     xlabel="Condition",
#     ylim=(0.0, 1.0),
#     fontsize=14,
#     color_offset=0  # Start from the beginning of the palette
# )
# plt.title("Default Colors (offset=0)", fontsize=16)

# plt.subplot(1, 3, 2)
# ax2, _, _ = plot_boxplot(
#     data_list=data_list, 
#     label_list=label_list, 
#     ylabel="Probability", 
#     xlabel="Condition",
#     ylim=(0.0, 1.0),
#     fontsize=14,
#     color_offset=4  # Start from the 5th color in the palette
# )
# plt.title("Color Offset = 4", fontsize=16)

# plt.subplot(1, 3, 3)
# ax3, _, _ = plot_boxplot(
#     data_list=data_list, 
#     label_list=label_list, 
#     ylabel="Probability", 
#     xlabel="Condition",
#     ylim=(0.0, 1.0),
#     fontsize=14,
#     color_offset=8  # Start from the 9th color in the palette
# )
# plt.title("Color Offset = 8", fontsize=16)

# plt.tight_layout()
# plt.show()

# # Example 2: Using custom colors (overrides color_offset)
# paired = sns.color_palette("Paired")
# custom_colors = [paired[1], paired[5], paired[9]]  # Light blue, orange, light green

# plt.figure(figsize=(10, 6))
# ax4, _, _ = plot_boxplot(
#     data_list=data_list, 
#     label_list=label_list, 
#     ylabel="Probability", 
#     xlabel="Condition",
#     ylim=(0.0, 1.0),
#     fontsize=16,
#     custom_colors=custom_colors,  # This takes precedence over color_offset
#     color_offset=2  # This will be ignored when custom_colors is provided
# )
# plt.title("Custom Colors (ignores offset)", fontsize=18)
# plt.show()

# %%
# # Example usage of plot_boxplot_silent
# np.random.seed(42)
# data1 = np.random.beta(8, 2, size=10)
# data2 = np.random.beta(4, 2, size=10)
# data3 = np.random.beta(6, 6, size=10)

# # By default, only returns the axes object (minimal output)
# ax = plot_boxplot(
#     data_list=[data1, data2, data3],
#     label_list=['Group A', 'Group B', 'Group C'],
#     ylabel='Score',
#     color_offset=2
# )
# plt.title("Silent Version - Default")

# # If you need the data, you can still get it
# plt.figure()
# ax, df, data = plot_boxplot(
#     data_list=[data1, data2, data3],
#     label_list=['Group A', 'Group B', 'Group C'],
#     ylabel='Score',
#     color_offset=4,
#     return_data=True  # Set to True to get all return values
# )
# plt.title("Silent Version - With Data")
# print("Example of accessing the dataframe:")
# print(df.head())


