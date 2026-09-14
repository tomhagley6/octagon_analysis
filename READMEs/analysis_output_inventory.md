# Analysis output inventory

Where the final analysis and plotting output lives in this repo. Split into notebooks
that produce output (run these, they emit the figures and stats) and the module
functions that are the terminal output step. Everything else is plumbing that feeds
these.

Figure counts are the images stored in each notebook's saved output.

---

## 1. Human-data output notebooks

### [demos/earlier_behaviour_figures/](../demos/earlier_behaviour_figures/) — the core human behavioural figures

| Notebook | Figures | Produces |
|---|---|---|
| [probability_chose_wall_demo.ipynb](../demos/earlier_behaviour_figures/probability_chose_wall_demo.ipynb) | 16 | The main P(choose High) results. Solo vs social boxplots/violins, Wilcoxon signed-rank, P(choose Low \| Low first visible), P(High) split by High-first-visible vs High-second-visible, the first/second-seen ratio plot, condition-count stacked bars, and performance-vs-P(Low) scatter |
| [opponent_visibility_analysis_demo.ipynb](../demos/earlier_behaviour_figures/opponent_visibility_analysis_demo.ipynb) | 2 | P(choice) conditioned on opponent visibility at slice onset, Wilcoxon test, plus start-position plots. Second half is explicitly marked `### Sandbox` |
| [trajectory_efficiency_demo.ipynb](../demos/earlier_behaviour_figures/trajectory_efficiency_demo.ipynb) | 5 | Direct:actual path ratios per player/session, split by win/loss, correlated against proportion score |
| [p_choose_high_across_walls_solo_demo.ipynb](../demos/earlier_behaviour_figures/p_choose_high_across_walls_solo_demo.ipynb) | 3 | P(choose High) per individual wall, solo. Two variants: pooled mean (SE of a proportion) and mean-of-player-means (SEM) |
| [plot_trial_trajectories_demo.ipynb](../demos/earlier_behaviour_figures/plot_trial_trajectories_demo.ipynb) | 9 | Trial and session trajectory panels, winning trajectories by wall separation, flip/rotate versions, slice-onset markers |
| [head_angle_direction_demo.ipynb](../demos/earlier_behaviour_figures/head_angle_direction_demo.ipynb) | 6 | Head-angle vectors on the octagon and head-angle-to-wall traces |

### [demos/heatmaps/](../demos/heatmaps/) — spatial heatmaps

| Notebook | Figures | Produces |
|---|---|---|
| [heatmap_demo.ipynb](../demos/heatmaps/heatmap_demo.ipynb) | 19 | Slice-onset / trigger-activation occupancy heatmaps, winner vs loser, and winner−loser difference maps |
| [heatmap_demo2.ipynb](../demos/heatmaps/heatmap_demo2.ipynb) | 4 | P(H \| H first vis) and P(H \| L first vis) spatial maps, with the full filter → flip/rotate → bin → ratio pipeline written out step by step. Saves the pkls in [heatmap_demo2_variables/](../demos/heatmaps/heatmap_demo2_variables/) |
| [heatmap_phigh_demo.ipynb](../demos/heatmaps/heatmap_phigh_demo.ipynb) | 3 | Same analysis refactored into [analysis/heatmap_phigh.py](../analysis/heatmap_phigh.py) functions, plus the see-high − see-low difference heatmap. Saves to [heatmap_phigh_variables/](../demos/heatmaps/heatmap_phigh_variables/) |
| [continuous_concentric_heatmap.ipynb](../demos/heatmaps/continuous_concentric_heatmap.ipynb) | 0 | Data prep only — writes the pkls the concentric heatmap methods consume |

### [demos/](../demos/) top level

- [reaction_time_demo.ipynb](../demos/reaction_time_demo.ipynb) — response-time summary stats + per-trial trace (1 fig)
- [trajectory_direction_demo.ipynb](../demos/trajectory_direction_demo.ipynb) — movement-direction vectors and cosine similarity to walls (6 figs)
- [was_first_wall_chosen_demo.ipynb](../demos/was_first_wall_chosen_demo.ipynb) — first-visible-wall-chosen logic, prints only, no figures

### [external/octagon_plots_sep10.ipynb](../external/octagon_plots_sep10.ipynb)

Social P(High) heatmaps for Low-first-visible and High-first-visible, plus a per-wall-polygon
coloured octagon. Uses [external/occupancy_and_strategy.py](../external/occupancy_and_strategy.py).

---

## 2. RL-agent output notebooks (not human data)

- [demos/strategy_characterisation/competitive/model_characterisation_demo260723.ipynb](../demos/strategy_characterisation/competitive/model_characterisation_demo260723.ipynb) — current batch, 7 figs
- [model_characterisation_demo260715.ipynb](../demos/strategy_characterisation/competitive/model_characterisation_demo260715.ipynb), [model_reward_characterisation_demo.ipynb](../demos/strategy_characterisation/competitive/model_reward_characterisation_demo.ipynb), [scanning_behaviour_demo.ipynb](../demos/strategy_characterisation/competitive/scanning_behaviour_demo.ipynb)
- [demos/strategy_characterisation/solo/optimal_solo_behaviour_demo.ipynb](../demos/strategy_characterisation/solo/optimal_solo_behaviour_demo.ipynb)
- [demos/tournament/](../demos/tournament/) — `tournament_analysis_260723.ipynb` (0 figs, not yet run through), `tournament_analysis_demo.ipynb`, `tournament_analysis_demo_2.ipynb`
- Written notes: [demos/strategy_characterisation/fingerprint_metrics.md](../demos/strategy_characterisation/fingerprint_metrics.md)

---

## 3. Module functions that are the final output step

### Creates a complete figure

| Function | File |
|---|---|
| `boxplot_probability_choose_wall` | [plot_probability_chose_wall.py:25](../plotting/plot_probability_chose_wall.py#L25) |
| `boxplot_solo_social_probability` | [plot_probability_chose_wall.py:158](../plotting/plot_probability_chose_wall.py#L158) |
| `boxplot_solo_social_probability_paired` | [plot_probability_chose_wall.py:173](../plotting/plot_probability_chose_wall.py#L173) |
| `plot_probability_choose_high_solo_social` | [plot_probability_chose_wall.py:420](../plotting/plot_probability_chose_wall.py#L420) |
| `plot_performance_against_probability_low_when_first_visible_df` | [plot_probability_chose_wall.py:299](../plotting/plot_probability_chose_wall.py#L299) |
| `plot_boxplot` | [plot_boxplots.py:9](../plotting/plot_boxplots.py#L9) |
| `plot_violinplot` | [plot_violinplots.py:8](../plotting/plot_violinplots.py#L8) |
| `plot_counts_condition_vs_chosen` | [plot_counts_seen_vs_chosen.py:27](../plotting/plot_counts_seen_vs_chosen.py#L27) |
| `plot_likelihoods_with_significance` | [likelihoods_with_LRT_barchart.py:6](../plotting/likelihoods_with_LRT_barchart.py#L6) |
| `plot_heatmap_phigh` | [heatmap_phigh.py:259](../analysis/heatmap_phigh.py#L259) |
| `plot_response_times` | [response_times.py:134](../analysis/response_times.py#L134) |
| `ConcentricHeatmapMethods.plot_comparison` | [continuous_concentric_heatmap.py:242](../plotting/continuous_concentric_heatmap.py#L242) |
| `plot_single_trial_flip_rotate_trajectories`, `plot_multiple_trials_flip_rotate_trajectories` | [flipped_rotated_trajectory_testing_functions.py:84,110](../plotting/flipped_rotated_trajectory_testing_functions.py#L84) |
| `plot_single_trial_first_wall_visibility`, `plot_multiple_trials_first_wall_visibility` | [wall_visibility_order_testing_functions.py:163,239](../plotting/wall_visibility_order_testing_functions.py#L163) |
| `plot_head_to_head`, `plot_reward_head_to_head`, `plot_overall_ranking`, `plot_interaction`, `plot_score_vs_reward` | [tournament_analysis.py](../analysis/tournament_analysis.py#L279) (agent data) |

### Draws onto an axis you supply (terminal drawing step, but you make the figure)

All of [plotting/plot_trajectory.py](../plotting/plot_trajectory.py) — `plot_trial_trajectory`
([L70](../plotting/plot_trajectory.py#L70)), `plot_trial_trajectory_colour_map`
([L25](../plotting/plot_trajectory.py#L25)), `plot_trial_winning_trajectory`
([L125](../plotting/plot_trajectory.py#L125)), `plot_session_trajectory`
([L187](../plotting/plot_trajectory.py#L187)), `mark_session_slice_onsets`
([L226](../plotting/plot_trajectory.py#L226)), `plot_trial_slice_onset_positions`
([L244](../plotting/plot_trajectory.py#L244)), `plot_session_trajectories_high_low`
([L306](../plotting/plot_trajectory.py#L306)); plus `plot_octagon`
([plot_octagon.py:397](../plotting/plot_octagon.py#L397)) and `plot_trajectory_with_circle`
([occupancy_and_strategy.py:47](../external/occupancy_and_strategy.py#L47)).

### Returns final statistics (end of the analysis chain, no figure)

- **Choice ~ visibility:** `probability_first_wall_chosen_and_low_multiple_sessions_social` ([wall_visibility_and_choice.py:613](../analysis/wall_visibility_and_choice.py#L613)), `..._combined_solo` ([L650](../analysis/wall_visibility_and_choice.py#L650)), `given_wall_chosen_conditioned_on_visibility_multiple_sessions_social` ([L698](../analysis/wall_visibility_and_choice.py#L698)), `..._solo` ([L752](../analysis/wall_visibility_and_choice.py#L752))
- **Choice ~ conditions:** `probability_trial_outcome_given_conditions_all_sessions` ([conditioned_player_choice.py:123](../analysis/conditioned_player_choice.py#L123)) — the generalised version
- **Opponent visibility:** [opponent_visibility.py:208](../analysis/opponent_visibility.py#L208), [:343](../analysis/opponent_visibility.py#L343), [:404](../analysis/opponent_visibility.py#L404), [:454](../analysis/opponent_visibility.py#L454)
- **Scores:** `proportion_score_sessions_df` ([player_scores.py:42](../analysis/player_scores.py#L42)), `proportion_score_subset_sessions_df` ([L74](../analysis/player_scores.py#L74)), `player_scores_sessions_df` ([L146](../analysis/player_scores.py#L146)), `proportion_wins_sessions` ([L169](../analysis/player_scores.py#L169)), `proportion_score_solo_sessions_df` ([L188](../analysis/player_scores.py#L188))
- **Efficiency:** `ratio_direct_to_absolute_distances_multiple_sessions` ([trajectory_efficiency.py:151](../analysis/trajectory_efficiency.py#L151)) and the winner/loser splits at [L76](../analysis/trajectory_efficiency.py#L76)/[L104](../analysis/trajectory_efficiency.py#L104)
- **P(High) aggregation:** `get_probability_chose_high_solo_social_all_sessions_df` ([plot_probability_chose_wall.py:607](../plotting/plot_probability_chose_wall.py#L607)), `get_probability_chose_high_social_all_sessions` ([L793](../plotting/plot_probability_chose_wall.py#L793)), `get_probability_chose_high_solo_all_sessions_combined` ([L855](../plotting/plot_probability_chose_wall.py#L855))
- **Heatmap pipelines:** `filter_pipeline_p_high_first_seen_wall` ([heatmap_phigh.py:191](../analysis/heatmap_phigh.py#L191)), `heatmap_counts_division` ([L228](../analysis/heatmap_phigh.py#L228)), `heatmap_lowbincounts_filter` ([L241](../analysis/heatmap_phigh.py#L241))
- **Loser's inferred choice:** `infer_loser_choice_session` ([loser_inferred_choice.py:183](../analysis/loser_inferred_choice.py#L183)), `player_wall_choice_win_or_loss` ([L230](../analysis/loser_inferred_choice.py#L230))
- **Response times:** `calculate_response_times` ([response_times.py:73](../analysis/response_times.py#L73)) and `ResponseTimes.print_summary_stats` ([L48](../analysis/response_times.py#L48))

---

## 4. Not output code

- **46 `.py`/`.ipynb` pairs** across [analysis/](../analysis/), [plotting/](../plotting/), [parse_data/](../parse_data/), [data_extraction/](../data_extraction/), [trajectory_analysis/](../trajectory_analysis/) — the `.ipynb` in each pair is just the nbconvert source of the `.py`, not a run.
- [sandbox/](../sandbox/) — 40 exploratory notebooks. A few still hold results not reproduced in demos: [wall_visibility_analysis_sandbox.ipynb](../sandbox/wall_visibility_analysis_sandbox.ipynb) (121 cells, 12 figs), [losers_inferred_choice_sandbox.ipynb](../sandbox/losers_inferred_choice_sandbox.ipynb) (12 figs), [trajectory_spread.ipynb](../sandbox/trajectory_spread.ipynb) (6 figs), [sandbox/glm/](../sandbox/glm/) (the k-fold/LOOCV GLM analyses, up to 18 figs).
- [sandbox/individuals_analysis_table/individuals_analysis_table.ipynb](../sandbox/individuals_analysis_table/individuals_analysis_table.ipynb) is a stub — it lists nine intended per-individual analyses and sketches three data structures, but computes nothing.
- [tests/](../tests/) — visual checks of flip/rotate, wall-visibility order, loser's inferred choice, trial proportions.
- [archive/](../archive/) — superseded, including [archive/benny_files/](../archive/benny_files/) and [archive/glm_analysis/](../archive/glm_analysis/).
- [sandbox/glm/test_kfoldCV_glm_analysis.ipynb](../sandbox/glm/test_kfoldCV_glm_analysis.ipynb) fails to parse as JSON — it's corrupt.
