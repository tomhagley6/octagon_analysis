import globals
import numpy as np
from parse_data import preprocess
import matplotlib.pyplot as plt
import plotting.plot_trajectory as plot_trajectory_module
import plotting.plot_octagon as plot_octagon

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def plot_sorted_session_trajectories(
    df,
    step_penalties,
    color_trajectory='lightseagreen',
    color_position='deeppink',
    color_by_quartile=False,
    quartile_colors=None
):
    num_trials = len(df)
    num_cols = 8
    num_rows = (num_trials + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))
    axes = np.array(axes).flatten()

    session_ids = list(range(num_trials))

    if quartile_colors is None:
        quartile_colors = {
            "q1": "lightseagreen",
            "q2": "darkseagreen",
            "q3": "goldenrod",
            "q4": "orangered",
        }

    step_penalties = np.asarray(step_penalties)

    # Assign ordered quartiles by rank, avoiding duplicate-edge issues from pd.qcut
    sorted_idx = np.argsort(step_penalties)
    quartile_labels = np.empty(len(step_penalties), dtype=object)

    quartile_bins = np.array_split(sorted_idx, 4)
    for q, idxs in zip(["q1", "q2", "q3", "q4"], quartile_bins):
        quartile_labels[idxs] = q

    # Sort sessions by step penalty
    sorted_trials = sorted(
        zip(step_penalties, df, session_ids, quartile_labels),
        key=lambda x: x[0]
    )

    for i, (penalty, trial_list, session_id, q_label) in enumerate(sorted_trials):
        ax = plot_octagon.plot_octagon(ax=axes[i])

        this_color = quartile_colors[q_label] if color_by_quartile else color_trajectory

        plot_trajectory_module.plot_session_trajectory(
            ax,
            trial_list,
            colour_player_1=this_color,
            alpha=0.4,
            slice_onset_markers=True
        )

        plot_trajectory_module.mark_session_slice_onsets(
            ax,
            trial_list,
            chosen_player=0,
            color=color_position
        )

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_title(
            f'{q_label} | step penalty: {penalty:.6f}\n'
            f'original session number: {session_id}',
            fontsize=12,
            fontweight='bold'
        )

    for j in range(num_trials, len(axes)):
        axes[j].set_visible(False)

    if color_by_quartile:
        legend_handles = [
            Patch(facecolor=quartile_colors[q], label=q)
            for q in ["q1", "q2", "q3", "q4"]
        ]
        fig.legend(
            handles=legend_handles,
            title="step penalty quartile",
            loc="upper right",
            fontsize=12,
            title_fontsize=12
        )

    plt.tight_layout()
    plt.show()