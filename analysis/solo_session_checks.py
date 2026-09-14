# %%
import numpy as np
import pandas as pd

# %% [markdown]
# ### Checks on solo sessions and the trials cut from them
#
# Solo trial lists arrive one entry per session, ordered by identify_filepaths.get_filenames as
# consecutive first/second pairs for each player (404 sessions = 202 players x 2 solos).
#
# Analysis drops the opening trials of a player's *first* solo, because that is their first
# exposure to the task and the choices are confounded by learning the controls. The second solo
# is never cut. Two forms of the cut are used in the repo, and they behave differently when the
# first solo is unusually short:
#
#   'combined' - concatenate, then cut the front:  (first + second)[cut:]
#                used by plot_probability_chose_wall and p_choose_high_across_walls_solo_demo.
#                If the first solo has fewer trials than the cut, the remainder comes out of the
#                second solo, which is not intended.
#
#   'first'    - cut, then concatenate:  first[cut:] + second
#                used by wall_visibility_and_choice. Never reaches the second solo, but if the
#                first solo has fewer trials than the cut it contributes nothing at all.

# %%
# defaults for the flags
MIN_TRIALS = 20
MAX_CUT_FRACTION = 0.25


# %%
def solo_cut_table(trial_lists_solo, cut_trials=5, cut_style='combined',
                   min_trials=MIN_TRIALS, max_cut_fraction=MAX_CUT_FRACTION):
    ''' Return a table of what a given cut removes from each player's solos.
        Takes the per-session solo trial lists, the cut size, and which cut form is being used.

        One row per player. cut_style is 'combined' or 'first', described at the top of this
        module. Flags are advisory: they mark sessions worth looking at, not sessions to exclude. '''

    if len(trial_lists_solo) % 2 != 0:
        raise ValueError(f"expected first/second solo pairs, got {len(trial_lists_solo)} sessions")

    if cut_style not in ('combined', 'first'):
        raise ValueError(f"cut_style must be 'combined' or 'first', got {cut_style!r}")

    rows = []
    for pair, idx in enumerate(range(0, len(trial_lists_solo), 2)):

        n_first = len(trial_lists_solo[idx])
        n_second = len(trial_lists_solo[idx + 1])

        # work out where the cut actually lands
        cut_from_first = min(cut_trials, n_first)
        if cut_style == 'combined':
            # anything the first solo cannot absorb is taken off the front of the second
            cut_from_second = max(0, cut_trials - n_first)
        else:
            # the cut is applied to the first solo alone, so it stops there
            cut_from_second = 0

        remaining_first = n_first - cut_from_first
        fraction_first_cut = cut_from_first / n_first if n_first else np.nan

        rows.append({
            'pair': pair,
            'session_index_first': idx,
            'session_index_second': idx + 1,
            'n_first': n_first,
            'n_second': n_second,
            'cut_trials': cut_trials,
            'cut_from_first': cut_from_first,
            'cut_from_second': cut_from_second,
            'remaining_first': remaining_first,
            'fraction_first_cut': fraction_first_cut,
        })

    table = pd.DataFrame(rows)

    # flag the cases worth knowing about
    # the cut has run past the end of the first solo and is eating second solo trials
    table['spills_into_second'] = table['cut_from_second'] > 0
    # the first solo is short enough that what survives the cut is thin
    table['short_first'] = table['n_first'] < min_trials
    table['short_second'] = table['n_second'] < min_trials
    # the cut takes a large share of the first solo, even though the session is not short overall
    table['heavy_cut'] = table['fraction_first_cut'] > max_cut_fraction
    # the first solo contributes no trials at all after the cut
    table['first_solo_empty'] = table['remaining_first'] == 0

    flag_cols = ['spills_into_second', 'short_first', 'short_second', 'heavy_cut', 'first_solo_empty']
    table['flagged'] = table[flag_cols].any(axis=1)

    return table


# %%
def report_solo_cuts(trial_lists_solo, cut_trials=5, cut_style='combined',
                     min_trials=MIN_TRIALS, max_cut_fraction=MAX_CUT_FRACTION, verbose=True):
    ''' Print which players are affected by a given cut, and return the table.
        Takes the per-session solo trial lists, the cut size, and which cut form is being used. '''

    table = solo_cut_table(trial_lists_solo, cut_trials=cut_trials, cut_style=cut_style,
                           min_trials=min_trials, max_cut_fraction=max_cut_fraction)

    if not verbose:
        return table

    n_players = len(table)
    print(f"{n_players} players, cut_trials={cut_trials}, cut_style='{cut_style}'")
    print(f"first solo trials : min {table['n_first'].min()}, "
          f"median {int(table['n_first'].median())}, max {table['n_first'].max()}")
    print(f"second solo trials: min {table['n_second'].min()}, "
          f"median {int(table['n_second'].median())}, max {table['n_second'].max()}")
    print()

    # report each flag separately, worst first, so a player can appear under more than one
    checks = [
        ('spills_into_second', 'cut runs past the first solo and removes second solo trials'),
        ('first_solo_empty', 'first solo contributes no trials after the cut'),
        ('short_first', f'first solo has fewer than {min_trials} trials'),
        ('heavy_cut', f'cut removes more than {max_cut_fraction:.0%} of the first solo'),
        ('short_second', f'second solo has fewer than {min_trials} trials'),
    ]

    for flag, description in checks:
        hits = table[table[flag]]
        print(f"{flag}: {len(hits)} - {description}")
        for _, row in hits.sort_values('n_first').iterrows():
            print(f"    pair {int(row['pair']):3d} (sessions {int(row['session_index_first'])}, "
                  f"{int(row['session_index_second'])}): "
                  f"first {int(row['n_first'])} trials, second {int(row['n_second'])} trials, "
                  f"cut {int(row['cut_from_first'])} from first"
                  + (f" + {int(row['cut_from_second'])} from second" if row['cut_from_second'] else "")
                  + f", {int(row['remaining_first'])} left in first")
    print()
    print(f"{int(table['flagged'].sum())} of {n_players} players flagged by at least one check")

    return table


# %%
def compare_cut_sizes(trial_lists_solo, cut_sizes=(1, 5, 10), cut_style='combined',
                      min_trials=MIN_TRIALS, max_cut_fraction=MAX_CUT_FRACTION):
    ''' Return one row per cut size summarising how many players each flag catches.
        Takes the per-session solo trial lists and the cut sizes used across the analysis. '''

    rows = []
    for cut in cut_sizes:
        table = solo_cut_table(trial_lists_solo, cut_trials=cut, cut_style=cut_style,
                               min_trials=min_trials, max_cut_fraction=max_cut_fraction)
        rows.append({
            'cut_trials': cut,
            'spills_into_second': int(table['spills_into_second'].sum()),
            'first_solo_empty': int(table['first_solo_empty'].sum()),
            'heavy_cut': int(table['heavy_cut'].sum()),
            'short_first': int(table['short_first'].sum()),
            'flagged': int(table['flagged'].sum()),
            'median_fraction_cut': table['fraction_first_cut'].median(),
        })

    return pd.DataFrame(rows)
