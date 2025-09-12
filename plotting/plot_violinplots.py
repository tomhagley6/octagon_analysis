# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
def plot_violinplot(data_list, label_list, ylabel, xlabel="",
                ylim=(0.0,1), set_aspect=3, fontsize=34, custom_colors=None, color_offset=0, 
                return_data=False, inner='box', split=False, bw_adjust=1.0):
    ''' 
    Helper function to create violin plots for multiple datasets with optional return values.
    
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
    inner : {'box', 'quartile', 'point', 'stick', None}
        Representation of the datapoints in the violin interior (default: 'box')
    split : bool
        Whether to split the violins when there are only two categories (default: False)
    bw_adjust : float
        Adjusts the bandwidth of the kernel density estimation (default: 1.0)
        Higher values create smoother plots, lower values show more detail
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
    
    # Create violinplot with custom properties
    ax = sns.violinplot(x="Condition", y="Probability", data=df, palette=custom_palette, 
                     inner=inner, split=split, bw_adjust=bw_adjust, width=0.8)
    
    # Increase linewidth for better visibility
    for violin in ax.collections:
        violin.set_edgecolor('black')
        violin.set_linewidth(2)
    
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
# # Example usage with multiple datasets
# np.random.seed(42)  # For reproducibility

# # Create different distribution types to showcase violin plot advantages
# # Normal distribution
# normal_dist = np.random.normal(0.6, 0.15, size=200)
# normal_dist = np.clip(normal_dist, 0, 1)  # Clip to 0-1 range

# # Bimodal distribution (mixture of two normals)
# bimodal_dist = np.concatenate([
#     np.random.normal(0.3, 0.1, size=100),
#     np.random.normal(0.7, 0.1, size=100)
# ])
# bimodal_dist = np.clip(bimodal_dist, 0, 1)  # Clip to 0-1 range

# # Skewed distribution (using beta)
# skewed_dist = np.random.beta(2, 5, size=200)

# # Create data list and labels list
# data_list = [normal_dist, bimodal_dist, skewed_dist]
# label_list = ['Normal', 'Bimodal', 'Skewed']

# # Plot with default settings
# ax1 = plot_violinplot(
#     data_list=data_list, 
#     label_list=label_list, 
#     ylabel="Probability", 
#     xlabel="Distribution Type",
#     inner='box'  # Show box plot inside violin
# )
# plt.title("Violin Plot with Inner Boxplot", fontsize=20)
# plt.show()

# %%
# # Comparing different inner representations
# fig, axes = plt.subplots(1, 4, figsize=(20, 6))

# inner_types = ['box', 'quartile', 'point', None]
# titles = ['With Boxplot', 'With Quartiles', 'With Points', 'Without Inner']

# for i, (inner_type, title) in enumerate(zip(inner_types, titles)):
#     plt.sca(axes[i])
#     ax = plot_violinplot(
#         data_list=data_list, 
#         label_list=label_list, 
#         ylabel="" if i > 0 else "Probability", 
#         xlabel="Distribution Type",
#         fontsize=14,
#         inner=inner_type,
#         color_offset=i*3  # Use different colors for each plot
#     )
#     plt.title(title, fontsize=16)

# plt.tight_layout()
# plt.show()

# %%
# # Demonstrating bandwidth adjustment for kernel density estimation
# fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# bw_adjusts = [0.5, 1.0, 2.0]
# titles = ['Low Bandwidth (0.5)', 'Default Bandwidth (1.0)', 'High Bandwidth (2.0)']

# for i, (bw_adjust, title) in enumerate(zip(bw_adjusts, titles)):
#     plt.sca(axes[i])
#     ax = plot_violinplot(
#         data_list=data_list, 
#         label_list=label_list, 
#         ylabel="" if i > 0 else "Probability", 
#         xlabel="Distribution Type",
#         fontsize=14,
#         inner='quartile',
#         bw_adjust=bw_adjust,  # Adjust the bandwidth
#         color_offset=i*3  # Use different colors for each plot
#     )
#     plt.title(title, fontsize=16)

# plt.tight_layout()
# plt.show()

# %%
# # Direct comparison between boxplots and violin plots
# # First create a function to mimic the boxplot function from plot_boxplots.ipynb
# def plot_boxplot_for_comparison(data_list, label_list, ylabel, xlabel="",
#                 ylim=(0.0,1), set_aspect=3, fontsize=24, custom_colors=None, color_offset=0):
#     # Similar implementation as the original plot_boxplot function
#     all_data = []
#     all_labels = []
    
#     for data, label in zip(data_list, label_list):
#         all_data.append(data.ravel())
#         all_labels.append(np.full(data.ravel().size, label))
    
#     data = np.concatenate(all_data)
#     labels = np.concatenate(all_labels)

#     df = pd.DataFrame({
#         "Probability": data,
#         "Condition": labels
#     })
    
#     if custom_colors is None:
#         paired = sns.color_palette("Paired")
#         custom_palette = [paired[(i + color_offset) % len(paired)] for i in range(len(data_list))]
#     else:
#         custom_palette = custom_colors

#     ax = sns.boxplot(x="Condition", y="Probability", data=df, palette=custom_palette, width=.8, 
#                      showmeans=False, showfliers=True, 
#                      flierprops=dict(markerfacecolor='none', marker='o', markersize=8, 
#                                      markeredgecolor='black', alpha=1))
    
#     for flier in ax.findobj(plt.Line2D):
#         if flier.get_marker() == 'o' and flier.get_alpha() == 1.0:
#             xdata, ydata = flier.get_xdata(), flier.get_ydata()
#             plt.plot(xdata, ydata, 'x', color='black', markersize=6, alpha=1)
    
#     for box in ax.artists:
#         box.set_edgecolor('black')
#         box.set_linewidth(2)
    
#     for whisker in ax.lines:
#         whisker.set_linewidth(2)

#     plt.ylabel(ylabel, fontsize=fontsize)
#     plt.xlabel(xlabel, fontsize=fontsize)
#     plt.xticks(fontsize=fontsize - 2)
#     plt.yticks(fontsize=fontsize - 2)
#     plt.ylim(ylim)
#     plt.gca().set_aspect(set_aspect)
#     plt.tight_layout()

#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)
    
#     return ax

# # Now compare boxplots and violin plots side by side
# fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# plt.sca(axes[0])
# plot_boxplot_for_comparison(
#     data_list=data_list, 
#     label_list=label_list, 
#     ylabel="Probability", 
#     xlabel="Distribution Type",
#     fontsize=16
# )
# plt.title("Boxplot Representation", fontsize=18)

# plt.sca(axes[1])
# plot_violinplot(
#     data_list=data_list, 
#     label_list=label_list, 
#     ylabel="Probability", 
#     xlabel="Distribution Type",
#     fontsize=16,
#     inner='box'
# )
# plt.title("Violin Plot Representation", fontsize=18)

# plt.tight_layout()
# plt.show()


