# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
def plot_likelihoods_with_significance(average_likelihoods, average_nlls, plotting_names,
                                        results, pairs, bonferroni_alpha, title=True,
                                          title1=None, title2=None,
                                          exclude_specific_pairs=None):
    """
    Plots a barplot of average_likelihoods with significance markers for smallest significant difference per model.
    """
    import matplotlib.pyplot as plt

    # Create figure with a single plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(average_likelihoods)), average_likelihoods, tick_label=plotting_names)
    ax.set_ylabel('Average likelihood', fontsize=16)
    if title:
        # Use title2 if provided, otherwise fall back to title1, or use a default
        plot_title = title2 if title2 else (title1 if title1 else 'Average Likelihoods with Significance')
        ax.set_title(plot_title)
    ax.set_ylim(bottom=0.7, top=0.75)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i, value in enumerate(average_likelihoods):
        ax.text(i, value + 0.005, f'{value:.3f}', ha='center', va='bottom', fontsize=16)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if any(len(name) > 18 for name in plotting_names):
        plt.xticks(rotation=45)

    # Add significance markers
    significant_pairs = [pair for pair, res in zip(pairs, results) if res[4]]
    y_max = max(average_likelihoods)
    
    # Adjust the spacing for significance lines
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    h = y_range * 0.02  # Height of the bracket
    vspace = y_range * 0.2  # Vertical space between brackets
    y_offset = y_range * 0.1  # Initial offset from the top of the highest bar
    
    # Make sure there's enough room above bars for significance markers
    ax.set_ylim(top=y_max + y_offset + vspace * 8)  # Allow space for at least 8 significance lines
    
    shown_pairs = set()

    # Handle excluded indices
    if exclude_specific_pairs is None:
        exclude_specific_pairs = []

    for i in range(len(average_nlls)):
        lower_models = [j for j in range(len(average_nlls)) if average_nlls[j] < average_nlls[i]]
        min_diff = None
        min_j = None
        min_p = None
        for j in lower_models:
            # handle excluded indices
            if (i,j) in exclude_specific_pairs or (j,i) in exclude_specific_pairs:
                continue

            if (i, j) in significant_pairs:
                res_idx = pairs.index((i, j))
                diff = average_nlls[i] - average_nlls[j]
                if min_diff is None or diff < min_diff:
                    min_diff = diff
                    min_j = j
                    min_p = results[res_idx][3]
            elif (j, i) in significant_pairs:
                res_idx = pairs.index((j, i))
                diff = average_nlls[i] - average_nlls[j]
                if min_diff is None or diff < min_diff:
                    min_diff = diff
                    min_j = j
                    min_p = results[res_idx][3]
        if min_j is not None and (min_j, i) not in shown_pairs and (i, min_j) not in shown_pairs:
            if min_p < bonferroni_alpha / 100:
                stars = '***'
            elif min_p < bonferroni_alpha / 10:
                stars = '**'
            else:
                stars = '*'
            x1, x2 = i, min_j
            y = y_max + y_offset + len(shown_pairs) * vspace
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='k')
            ax.text((x1 + x2) / 2, y + h, stars, ha='center', va='bottom', color='k', fontsize=22)
            shown_pairs.add((i, min_j))
    plt.tight_layout()
    plt.show()


