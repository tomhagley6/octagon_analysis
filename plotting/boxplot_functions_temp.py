def _setup_solo_social_boxplot(solo_probabilities, social_probabilities, ylabel, xlabel="",
                             ylim=(0.0,1), set_aspect=3, fontsize=24):
    ''' Helper function to set up common elements for solo vs social boxplots. '''
    
    # Prepare data
    data = np.concatenate([solo_probabilities.ravel(), social_probabilities.ravel()])
    labels = np.concatenate([
        np.full(solo_probabilities.size, 'Solo'),
        np.full(social_probabilities.size, 'Competition')
    ])

    # Create DataFrame for Seaborn
    df = pd.DataFrame({
        "Probability": data,
        "Condition": labels
    })
    
    # Create color palette
    paired = sns.color_palette("Paired")
    custom_palette = [paired[2], paired[3]]

    # Create figure with larger size for poster visibility
    plt.figure(figsize=(10, 8))
    
    # Create boxplot with custom flier (outlier) properties
    ax = sns.boxplot(x="Condition", y="Probability", data=df, palette=custom_palette, width=.8, 
                     showmeans=False, showfliers=True, 
                     flierprops=dict(marker='x', markersize=10, markeredgecolor='black'))
    
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
    
    return ax, df, solo_probabilities, social_probabilities


def boxplot_solo_social_probability(solo_probabilities, social_probabilities, ylabel, xlabel="",
                                  ylim=(0.0,1), set_aspect=3, fontsize=24):
    ''' Plotting function to create a boxplot comparing Solo and Competition conditions
        showing medians without individual paired lines.
        Takes arrays of probabilities for solo and social conditions. '''

    # Use shared setup function
    ax, df, _, _ = _setup_solo_social_boxplot(
        solo_probabilities, social_probabilities, ylabel, xlabel,
        ylim, set_aspect, fontsize
    )

    return plt.gca()


def boxplot_solo_social_probability_paired(solo_probabilities, social_probabilities, ylabel, xlabel="",
                                         ylim=(0.0,1), set_aspect=3, fontsize=24):
    ''' Plotting function to create a boxplot comparing Solo and Competition conditions
        with individual paired lines connecting data points.
        Takes arrays of probabilities for solo and social conditions. '''

    # Use shared setup function
    ax, df, solo_probabilities, social_probabilities = _setup_solo_social_boxplot(
        solo_probabilities, social_probabilities, ylabel, xlabel,
        ylim, set_aspect, fontsize
    )
    
    # Add connecting lines between paired data points
    # We need to ensure that we only connect non-NaN data points
    valid_mask = ~np.isnan(solo_probabilities.ravel()) & ~np.isnan(social_probabilities.ravel())
    solo_valid = solo_probabilities.ravel()[valid_mask]
    social_valid = social_probabilities.ravel()[valid_mask]
    
    # Draw lines connecting paired data points
    for i in range(len(solo_valid)):
        plt.plot(
            ['Solo', 'Competition'],  # x-coordinates
            [solo_valid[i], social_valid[i]],  # y-coordinates
            color='k',  # black line color
            linestyle='-',  # solid line
            marker='x',  # marker for the endpoints
            linewidth=1,
            alpha=0.4  # slight transparency
        )
    
    return plt.gca()
