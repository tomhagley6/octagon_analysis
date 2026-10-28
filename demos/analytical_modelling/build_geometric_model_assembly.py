''' Build geometric_model_assembly.ipynb beside this script. Edit this script, not the notebook.
    Run with the conda env python, then execute the notebook with the octagon_analysis kernel. '''

import os

import nbformat as nbf

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'geometric_model_assembly.ipynb')
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r'''
# Assembling the geometric race model

**Question.** Central models beat peripheral models in every tournament so far, yet human peripheral players score about as fast as human central players. Can a peripheral player, playing as well as the game allows, keep up with a central player?

This notebook builds a model of one race between a peripheral and a central agent from the arena's geometry and the game's rules, one component at a time. Each step adds one limit of the real game, and shows what it does to the score rate. It is written to be read for understanding. The analysis that built the model, with its dead ends, is in `nearer_wall_choice.ipynb`.

**Plan**
1. The arena, the task and the score.
2. Step 1: geometry only, both walls known at onset.
3. Step 2: the central agent senses walls with rays.
4. Step 3: the peripheral agent senses walls with rays.
5. Step 4: choosing a wall from partial knowledge.
6. Step 5: inferring where unseen walls are.
7. Step 6: the Δ rule with inference about both walls, and how Δ was identified.
8. Score rate across the steps and across trial configurations.
9. Known problems and limits.

**Choices and assumptions** (full list, with sources, in `READMEs/analytical_modelling_notes.md`)
- The model looks for how well each strategy *can* be played within the game's limits, not for how the trained agents play (user's choice). Turning and movement rates are kept close to the agents', as human players are likely to be less efficient still (user's choice).
- An agent's behaviour may depend only on what its rays have sensed, its own position and heading, and the arena layout. Never on where an unseen wall is (user's rule).
- Key assumption: the central agent always goes High, as trained central agents do (High on 0.998 of 260929 central won trials). This may not be its best play against a peripheral opponent.
- Peripheral agents start from the logged slice-onset positions and headings of the six 260929 peripheral models (2,376 agent-trials). The central agent starts at the exact centre with a random heading.
- Under the Δ rule, an agent in its own alcove but off its trigger backs into the trigger without turning when its chosen wall is its own with probability at least 0.8 (an arbitrary value picked for now; user's choice). Walls are known only from experience (ray hits, or standing on the own trigger at onset without the trial ending, which shows the own wall is unlit) or inference.
- No start-up delay, no timing noise, no use of the opponent's position, and the agents do not block each other.
- Trials follow the repo convention (first logged trial dropped, incomplete final trial dropped). Agent simulations have no spawn trial at the arena centre.

**Modules.** All numbers come from `analysis/analytical_race.py` (tests in `tests/test_analytical_race.py`), using the agents' ray sensor from `analysis/agent_raycast.py`.
''')

code(r'''
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

import data_strings
import data_extraction.data_saving as data_saving
import analysis.analytical_race as ar
import analysis.agent_raycast as agent_raycast
import analysis.agent_loser_choice as lc
import trajectory_analysis.visibility_occlusion as vo

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 220)

RECOMPUTE = False                   # rebuild the cached tables
BATCH = '260929'
SIM_DATA_FOLDER = os.path.join(data_strings.SIM_DATA_FOLDER, '2609', BATCH)
CUT = 0.5                           # peripheral below this mean onset centrality
D_GRID = np.round(np.arange(0, 3.81, 0.05), 2)     # thresholds swept (s); 3.8 s is the longest gap possible

# colours: walls, agents and rules
C_HIGHW, C_LOWW, C_WALL = '#d1605e', '#5b8db8', '#c4c4c4'
C_PERIPH, C_CENTRAL, C_ROUTE_C = '#5b8db8', '#9a9a9a', '#e39b5b'
C_PERIPH_DARK, C_PERIPH_LIGHT = '#3d6d9c', '#a9c7e3'           # peripheral wins at High / at Low
C_INK = '#3a3a3a'                                              # text and route lines (soft black)
C_REF = '#a8a8a8'                                              # reference lines
C_STEPS = sns.light_palette('#3d6d9c', 8).as_hex()[1:]

# set a soft plotting style for every figure
sns.set_theme(context='notebook', style='white', font='Liberation Sans', rc={
    'axes.edgecolor': '#a0a0a0', 'axes.linewidth': 1.3, 'axes.labelcolor': '#444444', 'axes.titlesize': 13,
    'axes.titleweight': 'normal', 'axes.spines.top': False, 'axes.spines.right': False,
    'text.color': '#3a3a3a', 'xtick.color': '#666666', 'ytick.color': '#666666',
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'axes.labelsize': 12, 'legend.frameon': False, 'legend.fontsize': 11,
    'xtick.major.width': 1.3, 'ytick.major.width': 1.3, 'xtick.major.size': 5, 'ytick.major.size': 5,
    'xtick.bottom': True, 'ytick.left': True, 'lines.linewidth': 2.6, 'patch.linewidth': 0, 'figure.dpi': 110})
''')

md(r'''
### Load the inputs

The peripheral positions and headings, the agents' measured turning and closing rates, the fixed time per trial outside the race, and the cached model races (one row per peripheral agent-trial, layout and search direction; about 40 minutes to build on 12 cores if not cached).
''')

code(r'''
# split the batch by mean onset centrality, and get the peripheral positions and headings
tables = data_saving.load_data(f'model_characterisation_{BATCH}', data_saving.cache_dir)['result']
centrality = tables['slice'].groupby('model')['centrality'].mean()
PERIPHERAL = sorted(centrality[centrality < CUT].index)
CENTRAL = sorted(centrality[centrality >= CUT].index)
race_df = data_saving.load_data(f'race_table_{BATCH}', data_saving.cache_dir)['result']
periph = race_df[race_df['model'].isin(PERIPHERAL)].reset_index(drop=True)
traj = data_saving.cached(f'agent_trajectories_{BATCH}_peripheral',
                          lambda: lc.collect_trajectories(SIM_DATA_FOLDER, PERIPHERAL, verbose=False),
                          recompute=RECOMPUTE, stamp=tuple(PERIPHERAL))
onset = periph.merge(traj[['model', 'trial_num', 'player', 'yaw']], on=['model', 'trial_num', 'player'])
POS = onset[['x', 'z']].to_numpy(float)
HEAD = np.degrees([y[0] for y in onset['yaw']])

# get the turning rate and closing speed measured from central agents
timing_df = data_saving.cached(f'trial_timing_{BATCH}_central5',
                               lambda: ar.collect_trial_timing(SIM_DATA_FOLDER, CENTRAL[:5], verbose=False),
                               recompute=RECOMPUTE, stamp=(tuple(CENTRAL[:5]), 'v2'))
alignment = data_saving.cached(f'central_alignment_{BATCH}_central5',
                               lambda: ar.collect_central_alignment(SIM_DATA_FOLDER, CENTRAL[:5], verbose=False),
                               recompute=RECOMPUTE, stamp=(tuple(CENTRAL[:5]), ar.ALIGN_TOL, 'rays', 'v4'))
AGENT_OVERHEAD = timing_df['overhead_s'].mean()
# take the human social mean time outside the race (agent logs round each delay up to a 1/3 s frame)
OVERHEAD = ar.human_overhead('social')
ROT, CLOSE = ar.rotation_speed(alignment), ar.closing_speed(alignment)

# get the model races and each position's paths and known-wall times
races = data_saving.cached(f'periph_ray_races_{BATCH}',
                           lambda: ar.build_races_rays(POS, HEAD, ROT, CLOSE, groups=onset['model'].to_numpy()),
                           recompute=RECOMPUTE,
                           stamp=(tuple(PERIPHERAL), len(POS), round(ROT, 3), round(CLOSE, 3), ar.HEAD_STEP, ar.TRACE_DT,
                                  True, True, 'v10'))
D_KNOWN, T_KNOWN = data_saving.cached(f'periph_known_wall_times_{BATCH}',
                                      lambda: ar.known_wall_times(POS, HEAD, ROT, CLOSE),
                                      recompute=RECOMPUTE, stamp=(len(POS), round(ROT, 3), round(CLOSE, 3), 'v1'))
print(f"peripheral models {PERIPHERAL}: {len(POS)} agent-trials, {len(races)} races")
''')

code(r'''
# set up the shared pieces: pooled races, central time distributions, inferred search directions
pooled = races.assign(group=0, prob=races['prob'] / races['prob'].sum())
inferred = ar.infer_direction(races, ar.onset_sensed(POS, HEAD))
pooled_inf = inferred.assign(group=0, prob=inferred['prob'] / inferred['prob'].sum())
CT_STRAIGHT = None                                          # central agent knows High and goes straight
CT_RAYS = ar.central_times(ROT, close_speed=CLOSE)          # rays, random search direction
CT_INF = ar.central_times_inferred(ROT, CLOSE)              # rays, inferred search direction
pi_, hi_, lo_ = pooled['pos'].to_numpy(), pooled['high'].to_numpy(), pooled['low'].to_numpy()
GAP_KNOWN = (D_KNOWN[pi_, hi_] - D_KNOWN[pi_, lo_]) / ar.SPEED
W_PROB = pooled['prob'].to_numpy()


def ends_at_low(ev, rows):
    """ Return whether each race ends at Low, counting instant Low wins for every rule.
        Takes an evaluate result and its race rows. """

    inst_high = (rows['t_high'] == 0).to_numpy()
    inst_low = ((rows['t_low'] == 0) & (rows['t_high'] > 0)).to_numpy()
    return np.where(inst_high, False, np.where(inst_low, True, ev['go_low'].to_numpy()))


def summary(rows, policy, delta=None, central_t=None):
    """ Return one rule's scores as a Series, with "ends at Low" counted the same way for every rule.
        Takes pooled race rows, the policy, its delta and the central times. """

    s = ar.summarise(rows, OVERHEAD, policy, delta, central_t).iloc[0]
    ev = ar.evaluate(rows, policy, delta, central_t)
    w = rows['prob'].to_numpy()
    return pd.Series({'peripheral score/s': s['periph_score_s'], 'central score/s': s['central_score_s'],
                      'margin': s['periph_score_s'] - s['central_score_s'],
                      'peripheral score/trial': s['periph_score_trial'], 'central score/trial': s['central_score_trial'],
                      'peripheral win share': s['periph_win'], 'ends at Low': float((w * ends_at_low(ev, rows)).sum()),
                      'race (s)': s['race_s']})


def with_times(t_high, t_low, go_low):
    """ Return pooled race rows with given peripheral times and Low choices (used through the
        'low_first' policy, which reads the choices from the low_first column). """

    return pooled_inf.assign(t_high=t_high, t_low=t_low, low_first=go_low)


def ref_score(central_t):
    """ Return central-vs-central score per second for a central time distribution. """

    return ar.central_reference(OVERHEAD, central_t=central_t)['score_s']


# drawing helpers
WALL_SEGS = vo.get_wall_segments()
WALL_IDS = np.repeat(np.arange(ar.NUM_WALLS), 5)
BACK, H_OUT = ar.alcove_back_positions()[0], 180.0      # example agent: back of alcove 0, facing out


def draw_arena(ax, high=None, low=None, shade=None, cmap='Purples'):
    """ Draw the arena walls, with High red and Low blue, or every wall shaded by a value (0-1).
        Takes the axis, the High and Low row indices and an optional value per wall. """

    for (a, b), w in zip(WALL_SEGS, WALL_IDS):
        if shade is not None:
            col = plt.get_cmap(cmap)(0.15 + 0.85 * shade[w]) if shade[w] > 0 else C_WALL
            lw = 3.2 if shade[w] > 0 else 1.6
        else:
            col = C_HIGHW if w == high else C_LOWW if w == low else C_WALL
            lw = 3.4 if col != C_WALL else 1.6
        ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=lw * 1.5)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.axis('off')


def draw_rays(ax, q, h, color=C_INK):
    """ Draw the 7 rays from a position to their first hit. Takes the axis, position, heading (deg) and colour. """

    _, _, dist = agent_raycast.cast_rays(np.asarray(q, float)[None], np.radians([h]))
    for dvec, dd in zip(agent_raycast.ray_directions(np.radians([h]))[0], dist[0]):
        ax.plot([q[0], q[0] + dd * dvec[0]], [q[1], q[1] + dd * dvec[1]], color=color, lw=0.6, alpha=0.7)


def route_xy(state, t0, t1, n=150):
    """ Return positions along a motion between two times on its own clock. """

    return np.array([state(t)[0] for t in np.linspace(t0, t1, n)])


def draw_central(ax, high):
    """ Draw the central agent and its straight route to High. Takes the axis and the High row index. """

    pts = ar.path_points(np.zeros(2), high)
    ax.plot(pts[:, 0], pts[:, 1], color=C_ROUTE_C, lw=2)
    ax.plot(0, 0, 'o', color=C_CENTRAL, ms=7)


FS_BASE = 24                                    # presentation text size (pt) for tick labels, legends and notes


def presentation_text(fig, base=FS_BASE, inner=None):
    """ Set large text and heavier axes on a figure for slides: titles base+4, axis labels base+2,
        everything else base. White text (inside bars or circles) is set to inner (default base-4).
        Takes the figure, the base size and the size for white text. """

    inner = base - 4 if inner is None else inner
    for ax in fig.axes:
        ax.title.set_fontsize(base + 4)
        ax.xaxis.label.set_size(base + 2); ax.yaxis.label.set_size(base + 2)
        ax.tick_params(labelsize=base, width=2, length=7)
        for sp_ in ax.spines.values():
            sp_.set_linewidth(2)
        for t in ax.texts:
            white = t.get_color() in ('white', '#ffffff', (1.0, 1.0, 1.0, 1.0))
            t.set_fontsize(inner if white else base)
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                t.set_fontsize(base)


def win_share(t, high, low, central_t=None):
    """ Return the share of races a peripheral time wins against the central times of a layout. """

    if central_t is None:
        return float(t < ar.TRIGGER_REACH / ar.SPEED)
    ct = central_t[(low - high) % ar.NUM_WALLS] if isinstance(central_t, dict) else central_t
    return float((np.asarray(ct) > t).mean())
''')

md(r'''
## 1. The arena, the task and the score

**Motivation.** Everything in the model follows from the arena's shape, how the walls are lit, and how a trial is scored. This section sets those out.

**Method and terms**
- *Reachable space*: where an agent's centre can go (grey outline): the octagon (wall face 16.75 units from the centre) plus eight alcoves (3 units deep, 3.8 wide).
- *Trigger line*: an agent wins a wall when its body first overlaps that wall's trigger box, which happens when its centre crosses a line 18.25 units from the arena centre inside the alcove (thin black lines).
- *Layouts*: each trial lights two walls, High (red, worth 1.0) and Low (blue, worth 0.4). High is equally likely at any wall. Low is 1, 2 or 4 walls away (45°, 90°, 180° trials) with probability 0.5, 0.25, 0.25, on either side. That gives 40 layouts.
- *Race*: the first agent to reach a lit wall's trigger wins it; the other scores nothing.
- *Score per second* (primary measure): total expected score over total expected trial time. Trial time is the race plus the mean time per trial outside it, 6.55 s in the human social sessions (trial start to slice onset, trigger to trial end, and the inter-trial interval). Two of these delays are random; the subsection after the table explains the value. Score per trial is the secondary measure.
- *Random elements* (the central heading, and search directions that tie) are averaged exactly over all their values, not sampled. A race whose winner depends on the central heading counts as a fractional win (section 8).
- *Margin*: peripheral minus central score per second. A central player against the same central opponent ties, so its margin is 0.

**How to read the figure.** (a) The reachable space, trigger lines, the arena centre (grey dot) and an alcove back (black square), where most peripheral agents wait. (b)-(d) One example layout of each trial type, with its probability.

**Why it is plotted.** The trigger lines show where each race ends. The layouts show how often a lit wall is near any given alcove.

**How to read the table.** The model's inputs, with their sources.
''')

code(r'''
fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
ax = axes[0]
poly = np.vstack([ar.reach_polygon(), ar.reach_polygon()[:1]])
ax.plot(poly[:, 0], poly[:, 1], color=C_WALL, lw=1)
for w in range(ar.NUM_WALLS):
    a_, b_ = ar._pt(ar.TRIGGER_REACH, -ar.H, w), ar._pt(ar.TRIGGER_REACH, ar.H, w)
    ax.plot([a_[0], b_[0]], [a_[1], b_[1]], color=C_INK, lw=1)
ax.plot(0, 0, 'o', color=C_CENTRAL, ms=7); ax.plot(*BACK, 's', color=C_INK, ms=7)
ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.axis('off')
ax.set_title('(a) Reachable space and trigger lines', fontsize=10)
lay_tab = ar.layout_table()
for ax, (lab, h, l) in zip(axes[1:], (('(b) 45° trial', 0, 1), ('(c) 90° trial', 0, 2), ('(d) 180° trial', 0, 4))):
    draw_arena(ax, h, l)
    sep = ((l - h) % 8)
    p_type = lay_tab.loc[lay_tab['sep'] == min(sep, 8 - sep), 'prob'].sum()
    ax.set_title(f'{lab}: probability {p_type:.2f}', fontsize=10)
plt.tight_layout(); plt.show()

inputs = pd.DataFrame([
    ('speed (units/s)', ar.SPEED, 'moveSpeed in OctagonAgent.cs; logged median 10.0'),
    ('turning rate (deg/s)', ROT, 'median turning rate of central agents before facing High (logs)'),
    ('closing speed while turning (units/s)', CLOSE, 'median of central agents (logs); best possible 9.46'),
    ('trigger line (units from centre)', ar.TRIGGER_REACH, 'trigger box front 18.86 less body radius 0.61'),
    ('mean time outside the race per trial (s)', OVERHEAD, 'human social sessions: start delay + end delay + ITI (logs)'),
    ('ray sensor', '7 rays, ±55 deg', 'SelectivePassThroughRaycast (agent_raycast.py)'),
], columns=['input', 'value', 'source']).set_index('input')
display(inputs)
''')

md(r'''
### Time outside the race

**Motivation.** Score per second needs the time each trial spends outside the race. The game sets these delays, and two of them are random. This subsection checks the delays in the agent and human logs, and sets the value the model uses.

**What the game does.** The agent build runs `OctagonArenaSettings.ITI()`. The human build runs `TrialHandler.cs`, with the same values from `GeneralGlobals.cs`.
- Trigger to trial end is fixed at 2 s (`trialEndDuration`).
- The inter-trial interval (ITI) is random, uniform over 2-5 s (`ITIMin`, `ITIMax`), mean 3.5 s.
- Trial start to slice onset is random, uniform over 0.5-1.5 s (`trialStartDurationMin`, `trialStartDurationMax`), mean 1.0 s.

These settings give a mean of 6.5 s per trial outside the race, with sd 0.91 s.

**What the model uses.** The mean time outside the race in the human social sessions, 6.55 s (`ar.human_overhead`; user's choice, as human play is the endpoint). The agent logs give 7.09 s, inflated by rounding (below). The delays are not treated as fixed. Over many trials, score per second is total score over total time. Only the mean time outside the race enters that ratio, as long as it does not depend on the race. The random delays do not change who wins: the model starts each race at the logged slice-onset positions, which already include whatever the agents did during the start delay.

**Terms.** Each part is measured between logged events (`ar.trial_delays`), with the first trial of each session dropped. The log of a trial runs from its trial start to the next trial's start, so the ITI is at its end.
- `trial start to slice onset`: the start delay.
- `trigger to trial end`: the fixed end delay.
- `trial end to next trial`: the ITI.

**How to read the first table.** The agent logs (5 central models, one row per part). It shows the logged mean, sd, minimum and maximum (s), beside the game setting. The last row is their sum.

**How to read the second table.** The number of agent trials at each logged start delay (rounded to 0.01 s).

**How to read the third table.** The same parts in the human logs, one row per part and session type (social, solo). Columns are the mean, sd and the 10th, 50th and 90th percentiles (s). For a uniform 0.5-1.5 s delay these percentiles are 0.6, 1.0 and 1.4. For a uniform 2-5 s ITI they are 2.3, 3.5 and 4.7. `outside setting` is the share of trials with a part outside its setting's range.

**How to read the fourth table.** The final rule (Δ rule, both walls inferred) scored with each candidate time outside the race. Δ is chosen separately for each, as in section 7. The human social mean is the one used.
''')

code(r'''
import contextlib, io
import parse_data.prepare_sim_data as prepare_sim_data


def overhead_parts(models):
    """ Return one row per agent-trial with the start delay, end delay and ITI (s) from the logs.
        Takes the model list. Drops the first logged trial. """

    rows = []
    for m_ in models:
        with contextlib.redirect_stdout(io.StringIO()):
            _, trial_list = prepare_sim_data.prepare_single_model_data(m_, sim_data_folder=SIM_DATA_FOLDER)
        for trial in trial_list[1:]:
            d = ar.trial_delays(trial)
            if d is not None:
                rows.append({'model': m_, **d})
    return pd.DataFrame(rows)


# get the agent delays
parts = data_saving.cached(f'overhead_parts_{BATCH}_central5', lambda: overhead_parts(CENTRAL[:5]),
                           recompute=RECOMPUTE, stamp=(tuple(CENTRAL[:5]), 'v1'))
cols = list(ar.DELAY_PARTS)
parts['time outside the race'] = parts[cols].sum(axis=1)
tab = parts.drop(columns='model').agg(['mean', 'std', 'min', 'max']).T
tab['game setting'] = ['uniform 0.5-1.5 (mean 1.0)', 'fixed 2.0', 'uniform 2-5 (mean 3.5)', 'mean 6.5, sd 0.91']
display(tab.round(3))
display(parts['trial start to slice onset'].round(2).value_counts().sort_index().rename('trials').to_frame())

# get the human delays, and flag parts outside the game's ranges
hparts = ar.human_trial_delays(recompute=RECOMPUTE)
hparts['time outside the race'] = hparts[cols].sum(axis=1)
out_range = {'trial start to slice onset': (0.49, 1.51), 'trigger to trial end': (1.99, 2.2), 'trial end to next trial': (1.99, 5.01)}
htab = []
for kind, g in hparts.groupby('session type'):
    for c in cols + ['time outside the race']:
        r_lo, r_hi = out_range.get(c, (-np.inf, np.inf))
        htab.append({'session type': kind, 'part': c, 'trials': len(g), 'mean': g[c].mean(), 'sd': g[c].std(),
                     'p10': g[c].quantile(0.1), 'median': g[c].median(), 'p90': g[c].quantile(0.9),
                     'outside setting': ((g[c] < r_lo) | (g[c] > r_hi)).mean()})
display(pd.DataFrame(htab).set_index(['session type', 'part']).round(3))

# score the final rule with each candidate time outside the race, choosing delta for each
comp = {}
for lab, ov in (('human social mean (used)', OVERHEAD), ('agent logged mean', AGENT_OVERHEAD), ('game settings mean', 6.5)):
    sw = pd.DataFrame([ar.summarise(pooled, ov, 'delta2', d, CT_INF).iloc[0].to_dict() | {'delta': d} for d in D_GRID])
    b = sw.loc[sw['periph_score_s'].idxmax()]
    comp[lab] = {'time outside the race (s)': ov, 'best Δ (s)': b['delta'], 'peripheral score/s': b['periph_score_s'],
                 'central score/s': b['central_score_s'], 'margin': b['periph_score_s'] - b['central_score_s'],
                 'central vs central score/s': ar.central_reference(ov, central_t=CT_INF)['score_s']}
display(pd.DataFrame(comp).T.astype(float).round(4))
''')

md(r'''
**Interpretation.**
- The agent logged mean (7.09 s) is 0.59 s above the game settings' 6.5 s. Each part is longer than its setting. The start delay averages 1.2 s (setting 1.0 s) and takes only the values 0.67, 1.00, 1.33 and 1.67 s. The end delay runs from 2.00 to 2.33 s (setting 2.0 s).
- This fits each wait ending on the next frame of the agent simulation, with frames 1/3 s of game time apart. The project's `Maximum Allowed Timestep` is 0.333 s (`ProjectSettings/TimeManager.asset`), and Unity resumes timed waits only on frames. A sped-up simulation (ML-Agents runs at a raised time scale) would hit that cap on every frame. Movement runs on the 0.02 s physics step, so races are not affected. That the time scale reaches the cap is likely but not checked.
- The human logs show no rounding. The start delay and ITI have the percentiles of uniform draws (third table), and their means are 1.0 s and 3.5 s. The mean time outside the race is 6.545 s in social and 6.484 s in solo sessions, close to the settings' 6.5 s. The social end delay is about 45 ms longer than in solo (2.045 vs 2.002 s); the cause is not checked.
- The choice changes the level of every score per second, not the comparison. With the agent mean (7.09 s) scores are about 7% lower than with the human mean (fourth table), the best Δ is the same 1.05 s, and the margin changes by at most 0.0001.
''')

md(r'''
## 2. Step 1: geometry only

**Motivation.** The simplest version asks what the peripheral position is worth if both agents knew both lit walls at onset and set off along their shortest paths at once.

**Method.** The central agent goes straight to High from the centre (18.25 units, 1.82 s). The peripheral agent follows a rule. "Closest" goes to the wall with the shorter path. The threshold rule goes to Low only if High is more than Δ seconds further than Low (Δ = 0 is closest, Δ ≥ 3.8 s is always High). Paths bend only at the alcove mouth corners.

**Terms.** Walls from own alcove: how many walls from the peripheral agent's alcove to a wall (0 = own, 4 = opposite).

**How to read the figure.** (a) Shortest paths from an alcove back to walls 1-4 away (coloured, with their lengths) and the central agent's path (orange). (b) Path length by walls from own alcove; the dashed line is the central agent's path. (c) Score per second against Δ for the peripheral agent (blue) and the central agent (grey), with the best Δ marked.

**Why it is plotted.** A peripheral agent can only win a wall it reaches before the central agent, so (b) shows which walls it can win. (c) shows how the choice threshold trades Low wins against High wins.
''')

code(r'''
d_back = ar.path_lengths(BACK[None])[0]
d_centre = ar.TRIGGER_REACH
sweep0 = pd.DataFrame([dict(delta=d, **summary(with_times(D_KNOWN[pi_, hi_] / ar.SPEED, D_KNOWN[pi_, lo_] / ar.SPEED, GAP_KNOWN > d),
                                               'low_first', central_t=CT_STRAIGHT)) for d in D_GRID])
D_BEST0 = float(sweep0.loc[sweep0['peripheral score/s'].idxmax(), 'delta'])

fig, axes = plt.subplots(1, 3, figsize=(27, 9), gridspec_kw=dict(width_ratios=[1, 1, 1.3]))
ax = axes[0]
draw_arena(ax)
for off, col in zip((1, 2, 3, 4), sns.color_palette('muted', 4).as_hex()):
    p = ar.path_points(BACK, off)
    ax.plot(p[:, 0], p[:, 1], color=col, lw=4, label=f'{off} away: {d_back[off]:.1f}')
pts_c = ar.path_points(np.zeros(2), 4)
ax.plot(pts_c[:, 0], pts_c[:, 1], color=C_ROUTE_C, lw=4)
ax.plot(0, 0, 'o', color=C_CENTRAL, ms=13)
ax.plot(*BACK, 's', color=C_INK, ms=12)
ax.legend(loc='lower left', fontsize=7, frameon=False, title='path (units)', title_fontsize=FS_BASE, bbox_to_anchor=(-0.05, -0.12))
ax.set_title('(a) Shortest paths from an alcove back', fontsize=10)
ax = axes[1]
offs = np.arange(5)
ax.bar(offs, d_back[[0, 1, 2, 3, 4]], color=C_PERIPH, width=0.6)
for x, v in zip(offs, d_back[[0, 1, 2, 3, 4]]):
    ax.text(x, v + 0.5, f'{v:.1f}', ha='center', fontsize=9)
ax.axhline(d_centre, color=C_ROUTE_C, ls='--', lw=3)
ax.text(-0.4, d_centre + 0.6, f'central agent {d_centre:.2f}', color=C_ROUTE_C, fontsize=8)
ax.set_xlabel('walls from own alcove'); ax.set_ylabel('shortest path to the trigger (units)')
ax.set_title('(b) Path length by wall', fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
ax = axes[2]
ax.plot(sweep0['delta'], sweep0['peripheral score/s'], color=C_PERIPH, lw=4, label='peripheral')
ax.plot(sweep0['delta'], sweep0['central score/s'], color=C_CENTRAL, lw=4, label='central')
ax.axvline(D_BEST0, color=C_REF, ls=':', lw=3)
ax.text(D_BEST0, ax.get_ylim()[1], f' best Δ {D_BEST0:.2f} s', va='top', fontsize=8)
ax.set_xlabel('threshold Δ (s): extra travel time to High\nbefore going to Low')
ax.set_ylabel('score per second (expected)')
ax.legend(frameon=False, fontsize=8)
ax.set_title('(c) Choosing Δ with both walls known', fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
presentation_text(fig)
plt.tight_layout(); plt.show()

step1 = pd.DataFrame({
    'always High': summary(with_times(D_KNOWN[pi_, hi_] / 10, D_KNOWN[pi_, lo_] / 10, np.zeros(len(pooled), bool)), 'low_first'),
    'closest (Δ = 0)': summary(with_times(D_KNOWN[pi_, hi_] / 10, D_KNOWN[pi_, lo_] / 10, GAP_KNOWN > ar.TIE), 'low_first'),
    f'best Δ = {D_BEST0:.2f} s': summary(with_times(D_KNOWN[pi_, hi_] / 10, D_KNOWN[pi_, lo_] / 10, GAP_KNOWN > D_BEST0), 'low_first')}).T
step1.loc['central vs central'] = [ref_score(CT_STRAIGHT), ref_score(CT_STRAIGHT), 0, 0.5, 0.5, 0.5, 0, ar.TRIGGER_REACH / 10]
display(step1.round(4))
''')

md(r'''
**How to read the table** (used for every step). One row per peripheral rule, against the same central agent, then central vs central. Ends at Low is the share of trials the peripheral agent's race ends at Low, including instant wins (an agent standing on a lit trigger at onset wins it at once).
''')

md(r'''
## 3. Step 2: the central agent senses with rays

**Motivation.** A real agent does not know where High is at onset: it senses walls with 7 rays and must turn to find and face High. This step gives the central agent that limit.

**Method.**
1. The central agent's heading at onset is uniform at random. Its 7 rays (0, ±18.3, ±36.7, ±55° from the heading) sense any part of a lit wall they hit first. From the centre, High is sensed when it is within 77.5° of the heading.
2. If High is sensed, it turns the short way to face it. If not, it turns in place in the direction that should find High sooner (inference from the walls it sees, step 5), or at random on a tie.
3. Once High is sensed it keeps turning to face it (331°/s), closing on it at 7.57 units/s, then travels straight at 10 units/s.

**Terms.** Central time: slice onset to reaching High's trigger (s).

**How to read the figure.** (a) A central agent facing away from High (red) with Low (blue) in view: its rays (black), the arc it turns (orange, labelled with the direction), then its route. (b) The distribution of the central agent's time to High over headings and layouts, with straight travel (1.82 s) and the mean marked.
- The tall bars at 1.82-1.88 s are headings with High in view (within 77.5°). The agent closes on High while it turns, so many headings give nearly the same time.
- Beyond 1.88 s, time rises in step with the angle turned (331°/s), up to 282.5° for a search that goes the long way.
- The bars form blocks, not one peak per wall. A given turn angle comes from two headings, one each side of High. The block height is the share of those two headings whose search direction gives that turn (both, one, or none).
- The search direction changes in steps as the heading changes. It changes where a wall enters or leaves the sensed range, which happens every 20-25° (each of the 8 walls has two edges). It also changes where the two directions' expected search times cross. So the blocks follow these switch points, not the walls.

**Why it is plotted.** The central agent's time is what the peripheral agent races against. Sensing makes it slower and more variable.
''')

code(r'''
h_ex, low_rel_ex = 150.0, 2                  # heading, and Low two walls clockwise of High (High at wall 0)
t_ex, d_ex = ar.central_time_one(h_ex, low_rel_ex, ROT, CLOSE)
ct_all = np.concatenate([v for v in CT_INF.values()])
ct_w = np.concatenate([np.full(len(v), ar.low_offset_weights()[k] / len(v)) for k, v in CT_INF.items()])

fig, axes = plt.subplots(1, 2, figsize=(22, 9.5))
ax = axes[0]
draw_arena(ax, 0, low_rel_ex)
draw_rays(ax, np.zeros(2), h_ex)
turn = (d_ex * (0 - h_ex)) % 360
arc = np.radians(h_ex + d_ex * np.linspace(0, turn, 60))
ax.plot(3 * np.sin(arc), 3 * np.cos(arc), color=C_ROUTE_C, lw=4)
pts_c = ar.path_points(np.zeros(2), 0)
ax.plot(pts_c[:, 0], pts_c[:, 1], color=C_ROUTE_C, lw=4)
ax.plot(0, 0, 'o', color=C_CENTRAL, ms=13)
ax.set_title(f"(a) Central agent, High out of view\nturns {turn:.0f}° {'clockwise' if d_ex == 1 else 'anticlockwise'}, "
             f"reaches High at {t_ex:.2f} s", fontsize=10)
ax = axes[1]
ax.hist(ct_all, bins=60, weights=ct_w, color=C_CENTRAL)
ct_mean_all = float((ct_all * ct_w).sum())
ax.axvline(ar.TRIGGER_REACH / ar.SPEED, color=C_INK, ls='--', lw=2.5)
ax.axvline(ct_mean_all, color=C_ROUTE_C, lw=3)
ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
ax.text(ar.TRIGGER_REACH / ar.SPEED, ax.get_ylim()[1] * 0.95, ' straight 1.82 s', fontsize=8)
ax.text(ct_mean_all, ax.get_ylim()[1] * 0.85, f' mean {ct_mean_all:.2f} s', fontsize=8, color=C_ROUTE_C)
ax.set_xlabel('central agent: time from onset to High (s)'); ax.set_ylabel('share of headings\nand layouts')
ax.set_title('(b) Central time to High', fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
presentation_text(fig)
plt.tight_layout(); plt.show()

step2 = pd.DataFrame({
    f'best Δ (step 1), straight central': step1.loc[f'best Δ = {D_BEST0:.2f} s'],
    f'best Δ (step 1), central with rays': summary(with_times(D_KNOWN[pi_, hi_] / 10, D_KNOWN[pi_, lo_] / 10, GAP_KNOWN > D_BEST0),
                                                    'low_first', central_t=CT_INF)}).T
step2.loc['central vs central, with rays'] = [ref_score(CT_INF), ref_score(CT_INF), 0, 0.5, 0.5, 0.5, 0,
                                              ar.central_reference(OVERHEAD, central_t=CT_INF)['race_s']]
display(step2.round(4))
''')

md(r'''
## 4. Step 3: the peripheral agent senses with rays

**Motivation.** From inside its alcove the peripheral agent's rays are blocked by the alcove walls, so it often cannot see the walls next to it, which are the ones it can win.

**Method.**
1. The peripheral agent senses with the same 7 rays, from its logged onset position and heading.
2. A wall its rays hit at onset is known: it turns onto the first stretch of its shortest path while closing along it (7.57 units/s), then follows the path at 10 units/s, turning to face each new stretch at bends while still moving.
3. A wall not sensed at onset must be found: the agent moves out of its alcove toward the centre and turns at the same time (turning and moving are separate controls) until a ray hits it.

**Terms.** Sensed at onset: a ray hits the wall at the onset heading. Search time: onset until a ray first hits the wall.

**How to read the figure.** (a) Rays at onset from an alcove back facing out: the far Low is hit, the adjacent High is not. (b) The search for the hidden High: moving out while turning (dashed), the point where High is sensed (red dot, with the rays then), and the path (solid). (c) Share of agent-trials with a wall sensed at onset, by walls from own alcove. (d) Mean time to a wall with rays (bars) and with the wall known at onset (dots); the dashed line is the central agent's mean.

**Why it is plotted.** (c) and (d) show which walls the alcove hides and what finding them costs.
''')

code(r'''
hg, lw_ = 1, 3                                # adjacent High, Low three walls away
fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
ax = axes[0]
draw_arena(ax, hg, lw_); draw_rays(ax, BACK, H_OUT); ax.plot(*BACK, 's', color=C_INK, ms=7)
ax.set_title('(a) Rays at onset: Low hit, adjacent High hidden', fontsize=9)
ax = axes[1]
draw_arena(ax, hg, lw_)
d_good = 1 if ar.step_search_trace(BACK, H_OUT, 1, ROT)['first'][hg] <= ar.step_search_trace(BACK, H_OUT, -1, ROT)['first'][hg] else -1
tr = ar.high_trace(BACK, H_OUT, hg, d_good, ROT, CLOSE)
ax.plot(*route_xy(tr['state'], 0, tr['sensed']).T, color=C_INK, ls='--', lw=1.5)
qs, hs = tr['state'](tr['sensed'])
draw_rays(ax, qs, hs, color=C_HIGHW)
ax.plot(*route_xy(tr['state'], tr['sensed'], tr['arrive']).T, color=C_INK, lw=2)
ax.plot(*BACK, 's', color=C_INK, ms=7); ax.plot(*qs, 'o', color=C_HIGHW, ms=7)
ax.set_title(f"(b) Search: High sensed at {tr['sensed']:.2f} s, reached at {tr['arrive']:.2f} s", fontsize=9)
by_off = pooled.assign(off=pooled['high_off'].abs()).groupby('off').apply(lambda g: pd.Series(dict(
    sensed0=np.average(g['high_sensed0'], weights=g['prob']),
    t_rays=np.average(g['t_high'], weights=g['prob']))), include_groups=False)
by_off['t_known'] = pd.Series(np.average(T_KNOWN[pi_, hi_], weights=W_PROB)).reindex(by_off.index)
tk = pd.DataFrame(dict(off=pooled['high_off'].abs(), t=T_KNOWN[pi_, hi_], w=W_PROB)).groupby('off').apply(
    lambda g: np.average(g['t'], weights=g['w']), include_groups=False)
ax = axes[2]
ax.bar(by_off.index, by_off['sensed0'], color=C_PERIPH, width=0.6)
for x, v in zip(by_off.index, by_off['sensed0']):
    ax.text(x, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
ax.set_ylim(0, 1.12); ax.set_xlabel('walls from own alcove'); ax.set_ylabel('share sensed at onset')
ax.set_title('(c) Wall sensed at onset', fontsize=10); ax.spines[['top', 'right']].set_visible(False)
ax = axes[3]
ax.bar(by_off.index, by_off['t_rays'], color=C_PERIPH, width=0.6, label='with rays')
ax.plot(tk.index, tk.values, 'o', color=C_INK, ms=7, label='known at onset')
for x, v, k in zip(by_off.index, by_off['t_rays'], tk.values):
    ax.text(x, v + 0.08, f'{v:.2f}\n({k:.2f})', ha='center', fontsize=8)
ct_mean = sum(ar.low_offset_weights()[k] * CT_INF[k].mean() for k in CT_INF)
ax.axhline(ct_mean, color=C_CENTRAL, ls='--', lw=1.2)
ax.text(-0.4, ct_mean + 0.06, f'central {ct_mean:.2f} s', fontsize=8, color='#666666')
ax.set_ylim(0, by_off['t_rays'].max() * 1.25)
ax.set_xlabel('walls from own alcove'); ax.set_ylabel('mean time from onset to the wall (s)')
ax.legend(frameon=False, fontsize=8, loc='upper left')
ax.set_title('(d) Time to a wall', fontsize=10); ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.show()

# score the best step-1 choice with each extra limit: turning onto the path, then finding the chosen wall
step3 = pd.DataFrame({
    'best Δ, walls known, sets off at once': step2.iloc[1],
    'best Δ, walls known, turns onto its path': summary(with_times(T_KNOWN[pi_, hi_], T_KNOWN[pi_, lo_], GAP_KNOWN > D_BEST0),
                                                         'low_first', central_t=CT_INF),
    'best Δ, chosen wall found with rays (not playable)': summary(with_times(pooled_inf['t_high'], pooled_inf['t_low'], GAP_KNOWN > D_BEST0),
                                                                  'low_first', central_t=CT_INF)}).T
display(step3.round(4))
''')

md(r'''
**How to read the table.** The best step-1 choice (Δ chosen with both walls known), against the central agent with rays, as each limit is added. The last row still chooses the wall from both walls' true positions at onset, which the agent cannot know; it is shown only to isolate the cost of finding the chosen wall.
''')

md(r'''
## 5. Step 4: choosing a wall from partial knowledge

**Motivation.** With rays, the peripheral agent may know only one lit wall, or none, when it has to act. A rule may only use what has been sensed.

**Method.** Three rules, with the search direction chosen at random:
- *High, always*: search for High if needed, then go to it.
- *Low if seen first*: go to whichever lit wall a ray hits first (both at once goes High).
- *Nearest once both sensed*: head for High until both walls have been sensed, then go to the one with the shorter remaining path.

**How to read the figure.** Two example layouts for the agent at an alcove back. (a) "Low if seen first": the far Low is hit first, so the agent goes there (solid), giving up the hidden adjacent High it would reach on the dashed route. (b) "Nearest": the agent sets off for a High two walls away (dashed), sees the adjacent Low as it leaves the alcove (star), and switches (solid). Titles give times and the central agent's mean.

**Why it is plotted.** It shows why committing to the first wall seen fails from an alcove, and why waiting to compare both walls works.
''')

code(r'''
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
# (a) Low if seen first: far Low seen at onset, adjacent High hidden
ax = axes[0]
hg, lw_ = 1, 3
draw_arena(ax, hg, lw_)
tr_h = ar.high_trace(BACK, H_OUT, hg, d_good, ROT, CLOSE)
tr_l = ar.known_trace(BACK, H_OUT, lw_, ROT, CLOSE)
ax.plot(*route_xy(tr_h['state'], 0, tr_h['arrive']).T, color=C_INK, ls='--', lw=1.3)
ax.plot(*route_xy(tr_l['state'], 0, tr_l['arrive']).T, color=C_INK, lw=2)
draw_central(ax, hg); ax.plot(*BACK, 's', color=C_INK, ms=7)
ax.set_title(f"(a) Low if seen first: to Low {tr_l['arrive']:.2f} s (wins {win_share(tr_l['arrive'], hg, lw_, CT_INF):.2f})\n"
             f"instead of High {tr_h['arrive']:.2f} s (would win {win_share(tr_h['arrive'], hg, lw_, CT_INF):.2f})", fontsize=9)
# (b) nearest: heads for High two away, sees the adjacent Low, switches
ax = axes[1]
hg, lw_ = 2, 1
draw_arena(ax, hg, lw_)
tr = ar.high_trace(BACK, H_OUT, hg, 1, ROT, CLOSE)
first = ar.trace_first_sensed(tr)
t_dec = max(tr['sensed'], first[lw_])
q, hh = tr['state'](t_dec)
lt = ar.known_trace(q, hh, lw_, ROT, CLOSE)
ax.plot(*route_xy(tr['state'], 0, tr['arrive']).T, color=C_INK, ls='--', lw=1.3)
ax.plot(*route_xy(tr['state'], 0, t_dec).T, color=C_INK, lw=2)
ax.plot(*route_xy(lt['state'], 0, lt['arrive']).T, color=C_INK, lw=2)
ax.plot(*q, '*', color=C_INK, ms=13)
draw_central(ax, hg); ax.plot(*BACK, 's', color=C_INK, ms=7)
ax.set_title(f"(b) Nearest: switches at {t_dec:.2f} s, Low at {t_dec + lt['arrive']:.2f} s (wins "
             f"{win_share(t_dec + lt['arrive'], hg, lw_, CT_INF):.2f});\nHigh would take {tr['arrive']:.2f} s "
             f"(central mean {CT_INF[(lw_ - hg) % 8].mean():.2f} s)", fontsize=9)
plt.tight_layout(); plt.show()

step4 = pd.DataFrame({lab: summary(pooled, pol, central_t=CT_INF) for lab, pol in
                      (('High, always', 'high'), ('Low if seen first', 'low_first'), ('Nearest once both sensed', 'nearest'))}).T
step4.loc['central vs central'] = [ref_score(CT_INF), ref_score(CT_INF), 0, 0.5, 0.5, 0.5, 0,
                                   ar.central_reference(OVERHEAD, central_t=CT_INF)['race_s']]
display(step4.round(4))
''')

md(r'''
## 6. Step 5: inferring where unseen walls are

**Motivation.** What the rays have sensed narrows down where an unseen wall can be: an unlit wall cannot be lit, and Low is always 1, 2 or 4 walls from High. An agent can use this to search the right way, and to judge an unseen wall's distance.

**Method.** The agent combines the layout rules with the walls its rays have hit into a probability for each wall being High (or Low). If the wall it wants is unseen, it searches in the direction with the lower expected time until a ray hits it. The rules in this step infer once, at onset (`ar.infer_direction`). The Δ rule in step 6 infers again at every new wall (`ar.joint_posterior`). The subsection after the table explains both.

**Terms.** Probability of High (or Low) at a wall: the share of the remaining layouts, weighted by their prior, with that wall lit as High (or Low).

**How to read the figure.** Walls shaded by probability (darker = more likely; grey = ruled out), for the agent at an alcove back facing out; the dotted wall is the one seen, and the thin lines are the rays at onset. (a) High three walls away is seen; where could Low be? (b) The far Low is seen and the adjacent High is hidden; where could High be? (c) Mean search time for a hidden High, with a random and an inferred search direction.

**Why it is plotted.** (a) and (b) show what the agent can work out from one sensed wall. (c) shows how much that speeds up its search.

**How to read the table.** As in step 4, with inferred search directions.
''')

code(r'''
S_back = ar.onset_sensed(BACK[None], [H_OUT])[0]
fig, axes = plt.subplots(1, 3, figsize=(22, 7.5))
for ax, (hg, lw_, which, lab) in zip(axes[:2], ((3, 1, 'low', '(a) High seen 3 away: where is Low?'),
                                                 (1, 3, 'high', '(b) Low seen 3 away: where is High?'))):
    J = ar.joint_posterior(S_back, hg if S_back[hg] else None, lw_ if S_back[lw_] else None)
    prob = J.sum(axis=0) if which == 'low' else J.sum(axis=1)
    draw_arena(ax, shade=prob / prob.max(), cmap='Blues' if which == 'low' else 'Reds')
    seen_w = hg if which == 'low' else lw_
    for (a_, b_), w in zip(WALL_SEGS, WALL_IDS):
        if w == seen_w:
            ax.plot([a_[0], b_[0]], [a_[1], b_[1]], color=C_HIGHW if which == 'low' else C_LOWW, lw=2.5, ls=':')
    for w in range(8):
        if prob[w] > 0:
            p_ = ar._pt(13.0, 0.0, w)
            ax.text(p_[0], p_[1], f'{prob[w]:.2f}', ha='center', va='center', fontsize=9)
    p_ = ar._pt(13.0, 0.0, seen_w)
    ax.text(p_[0], p_[1], 'seen\nHigh' if which == 'low' else 'seen\nLow', ha='center', va='center', fontsize=8,
            color=C_HIGHW if which == 'low' else C_LOWW)
    draw_rays(ax, BACK, H_OUT); ax.plot(*BACK, 's', color=C_INK, ms=7)
    ax.set_title(lab, fontsize=10)
hidden = ~pooled['high_sensed0']
search = pd.Series({'random direction': np.average(pooled.loc[hidden, 'high_tsense'], weights=pooled.loc[hidden, 'prob']),
                    'inferred direction': np.average(pooled_inf.loc[hidden, 'high_tsense'], weights=pooled_inf.loc[hidden, 'prob'])})
ax = axes[2]
ax.bar(range(2), search.values, color=[C_WALL, C_PERIPH], width=0.6)
for x, v in enumerate(search.values):
    ax.text(x, v + 0.005, f'{v:.3f} s', ha='center', fontsize=9)
ax.set_xticks(range(2), ['random\ndirection', 'inferred\ndirection']); ax.set_ylabel('mean search time\nfor a hidden High (s)')
ax.set_title('(c) Search time', fontsize=10); ax.spines[['top', 'right']].set_visible(False)
ax.set_ylim(0, search.max() * 1.25)
presentation_text(fig)
plt.tight_layout(); plt.show()

step5 = pd.DataFrame({lab: summary(pooled_inf, pol, central_t=CT_INF) for lab, pol in
                      (('High, always', 'high'), ('Low if seen first', 'low_first'), ('Nearest once both sensed', 'nearest'))}).T
display(step5.round(4))
''')

md(r'''
### How the inference works

**Where each function is used.**
- `ar.infer_direction` sets the search direction for the step 5 rules ("High, always", "Low if seen first", "nearest once both sensed"). It runs once per race, at onset.
- `ar.joint_posterior` is used by the Δ rule (step 6). It is recomputed at onset and each time a ray first hits a new wall.

**How `ar.infer_direction` forms the posterior** (through `ar.high_posterior_given`).
1. Start from the layout prior. High is at each wall with probability 1/8. Low is 1 wall from High with probability 0.5 (0.25 each side), 2 walls with 0.25 (0.125 each side) and 4 walls with 0.25. This gives 40 layouts.
2. Remove layouts that conflict with the walls the rays hit at onset. A hit wall that is not lit can be neither High nor Low. If Low was hit, Low is fixed at that wall. If Low was not hit, Low can only be at a wall not hit.
3. Add up the remaining layouts for each High wall and rescale to sum to 1. This is P(High at each wall).

So it uses every wall hit at onset, including unlit walls and Low. It gives the same probabilities as `ar.joint_posterior` summed over Low (checked on 2133 random cases of walls hit).

**How the direction is chosen** (`ar._choose_direction`).
1. For each direction, take the time until a ray first hits each wall during that direction's search (`high_tsense`). The search is the agent moving out of its alcove while turning.
2. Weight these times by P(High at each wall) and add them. This is the expected time to sense High.
3. Take the direction with the lower expected time. If they are equal (within 1e-9 s), each direction keeps probability 0.5.

**What `ar.infer_direction` does not do.** It does not update during the search. Walls swept past without finding High are not used to reconsider the direction. The search keeps turning the way chosen at onset until a ray hits High. Under "Low if seen first" and "nearest", a Low hit during the search can change which wall the agent goes to, but not the direction of a search for High that is still running.

**What the Δ rule does instead.** The set of walls hit so far is kept for the whole trial and grows with each new wall. At each new wall, `ar.joint_posterior` is recomputed from all of them. It gives the expected path length of each unseen wall, which sets the choice between High and Low. A search direction is chosen from the updated posterior, in the same way as above, only when a new search starts. That is at onset, or when the agent switches to a wall it has not seen. A search already running for the same wall keeps its direction.

**Choices.**
- The agent remembers every wall it has sensed in the trial (user's choice).
- A running search never reverses, even when the walls swept show the likely side is empty (user's choice). Humans may reverse, but the effect is likely small and it adds complexity.

**How to read the table.** A worked example of `ar.infer_direction` for the agent of figure (b) above. It is at the back of alcove 0 facing out, High is at wall 1 (hidden) and Low is at wall 3 (hit at onset). One row per wall.
- `hit_at_onset`: a ray hits the wall at onset. Wall 0 counts as hit because the rays hit the sides of the agent's own alcove, which belong to wall 0's object.
- `prior_high`: P(High at the wall) before sensing (1/8 each).
- `posterior_high`: P(High at the wall) given the walls hit (0 = ruled out).
- `t_first_cw_s`, `t_first_ccw_s`: time until a ray first hits the wall when searching clockwise (direction +1, heading increasing) or anticlockwise (-1). Facing out of alcove 0, a clockwise turn brings wall 7 into view first. Each is 0 for a wall hit at onset.
- The last row gives each direction's expected time to sense High (sum of posterior x time). The direction chosen is printed below.
''')

code(r'''
# get the walls hit at onset and the High posterior for figure (b)'s agent (Low at wall 3 hit, High hidden)
S_ex = ar.onset_sensed(BACK[None], [H_OUT])[0]
post_ex = ar.high_posterior_given(S_ex, 3 if S_ex[3] else None)
# get each wall's first-sense time along each search direction (moving out of the alcove while turning)
first_ex = {d: ar.step_search_trace(BACK, H_OUT, d, ROT, move=ar.search_moves(BACK, True))['first'] for d in (1, -1)}
ex = pd.DataFrame({'hit_at_onset': S_ex, 'prior_high': ar.pair_prior().sum(axis=1), 'posterior_high': post_ex,
                   't_first_cw_s': np.where(S_ex, 0.0, first_ex[1]), 't_first_ccw_s': np.where(S_ex, 0.0, first_ex[-1])},
                  index=pd.Index(range(ar.NUM_WALLS), name='wall'))
# compute the expected time to sense High in each direction, and the direction chosen
keep = post_ex > 0
e_cw, e_ccw = (post_ex[keep] * ex['t_first_cw_s'][keep]).sum(), (post_ex[keep] * ex['t_first_ccw_s'][keep]).sum()
d_ex = ar._choose_direction(post_ex, np.c_[ex['t_first_cw_s'], ex['t_first_ccw_s']])
ex.loc['expected time to sense High (s)'] = ['', '', '', e_cw, e_ccw]
display(ex.round(3))
print(f"direction chosen: {dict([(1, 'clockwise'), (-1, 'anticlockwise'), (0, 'tie (50/50)')])[d_ex]}")
''')

md(r'''
## 7. Step 6: the Δ rule with inference about both walls

**Motivation.** The best rule in step 1 was a threshold. With inference, the agent can apply it even to a wall it has not yet seen, using that wall's expected distance, and so pick the walls it is more likely to win.

**Method** (`delta2`).
1. At onset and whenever a ray first hits a new wall, the agent works out where both walls could be (step 5).
2. It goes for Low if (distance to High - distance to Low) / 10 units/s > Δ, otherwise High. A sensed wall's distance is its shortest path; an unseen wall's is the expected path over where it could be.
3. If the chosen wall is sensed it goes to it; if not, it searches for it (moving out of its alcove while turning, in the direction that should find it sooner).
4. It re-decides at every new wall and may switch either way.
5. Backpedal. If the agent is in its own alcove but off its trigger, and the chance that its chosen wall is its own wall is at least 0.8, it backs straight into the trigger at full speed without turning. Inference is enough; it need not see the wall, so turning to look would only cost time. The 0.8 is an arbitrary value picked for now (user's choice).
6. The agent knows a wall only from experience or inference. A ray hit is experience. So is standing on its own trigger at onset without the trial ending, which shows its own wall is unlit. This is treated like a ray hit on that wall, even when no ray reaches it.

**How Δ was identified.** Δ is swept from 0 to 3.8 s and the value with the highest peripheral score per second is taken. Each race's outcome is stored as a function of Δ, so the sweep is exact. The choice is checked by choosing Δ on five of the six sets of start positions (each set comes from one trained peripheral model) and scoring it on the sixth.

**How to read the first figure.** Three routes for the agent at an alcove back, at the best Δ. (a) High three walls away is seen; the agent infers Low is probably near its own alcove, searches that way (dashed), sees Low (star) and goes there. (b) A hidden adjacent High with a far Low seen: High is probably near, so it keeps searching for High. (c) Both walls far: it goes for High. Titles give the outcome and times.

**How to read the second figure.** (a) Score per second against Δ for the peripheral agent (blue) and the central agent (grey), with the High-only version of the rule (inference about High only, dashed) and the first model (dotted) for comparison; the best Δ and the range within 0.5% of it are marked. (b) The peripheral margin against Δ. (c) Score per second against Δ for each set of start positions (labelled by the trained model they come from), with each set's own best Δ marked (dots). The rule set is the same; only where the agent starts differs. (d) For each race that ends at Low at small Δ, the Δ above which it no longer does (weighted share of trials): the steps in (a) sit where many races switch together.

**Why it is plotted.** It shows how sharply the result depends on Δ, why the curve has steps, and that the chosen Δ is not tuned to particular start positions.
''')

code(r'''
sweep6 = pd.DataFrame([dict(delta=d, **summary(pooled, 'delta2', d, CT_INF)) for d in D_GRID])
sweep_hi = pd.DataFrame([dict(delta=d, **summary(pooled_inf, 'delta', d, CT_INF)) for d in D_GRID])
D_BEST6 = float(sweep6.loc[sweep6['peripheral score/s'].idxmax(), 'delta'])
top6 = sweep6.loc[sweep6['peripheral score/s'] >= 0.995 * sweep6['peripheral score/s'].max(), 'delta']

# draw three example routes at the best delta
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (hg, lw_, lab) in zip(axes, ((3, 1, '(a) Far High seen, Low inferred near'), (1, 3, '(b) Hidden adjacent High, far Low seen'),
                                      (4, 5, '(c) Both walls far'))):
    draw_arena(ax, hg, lw_)
    r = ar.delta2_route(BACK, H_OUT, hg, lw_, 1, D_BEST6, ROT, CLOSE)
    for state, s0, e, t0, is_search in r['segments']:
        ax.plot(*route_xy(state, s0, e).T, color=C_INK, lw=1.4 if is_search else 2, ls='--' if is_search else '-')
    for t_d, q, to_low in r['decisions'][1:]:
        ax.plot(*q, '*', color=C_LOWW if to_low else C_HIGHW, ms=13)
    draw_central(ax, hg); ax.plot(*BACK, 's', color=C_INK, ms=7)
    ax.set_title(f"{lab}\nto {'Low' if r['go_low'] else 'High'} at {r['time']:.2f} s (wins {win_share(r['time'], hg, lw_, CT_INF):.2f}); "
                 f"central mean {CT_INF[(lw_ - hg) % 8].mean():.2f} s", fontsize=9)
plt.tight_layout(); plt.show()

# how delta was identified
per_model = pd.concat([ar.summarise(races, OVERHEAD, 'delta2', d, CT_INF).assign(delta=d) for d in D_GRID], ignore_index=True)
HI_ = np.column_stack([pooled[f'd2_hi_{k}'] for k in range(ar.MAX_PIECES)])
LOW_ = np.column_stack([pooled[f'd2_low_{k}'] for k in range(ar.MAX_PIECES)])
# find, per race, the delta above which it stops ending at Low (first piece boundary where Low changes to High)
stop = np.full(len(pooled), np.nan)
for k in range(ar.MAX_PIECES - 1):
    change = (LOW_[:, k] > 0.5) & (LOW_[:, k + 1] < 0.5) & np.isnan(stop)
    stop[change] = HI_[change, k]

fig, axes = plt.subplots(2, 2, figsize=(15, 9))
ax = axes[0, 0]
ax.plot(sweep6['delta'], sweep6['peripheral score/s'], color=C_PERIPH, lw=2, label='peripheral')
ax.plot(sweep6['delta'], sweep6['central score/s'], color=C_CENTRAL, lw=2, label='central')
ax.plot(sweep_hi['delta'], sweep_hi['peripheral score/s'], color=C_PERIPH, lw=1, ls='--', label='peripheral, High-only inference')
ax.plot(sweep0['delta'], sweep0['peripheral score/s'], color=C_PERIPH, lw=1, ls=':', label='peripheral, step 1 (walls known)')
ax.axhline(ref_score(CT_INF), color=C_INK, ls='--', lw=1)
ax.text(0.02, ref_score(CT_INF), f'central vs central {ref_score(CT_INF):.4f}', va='bottom', fontsize=8)
ax.axvspan(top6.min(), top6.max(), color='#ececec', alpha=0.5)
ax.axvline(D_BEST6, color=C_REF, ls=':', lw=1.5)
ax.text(D_BEST6, ax.get_ylim()[0], f' best Δ {D_BEST6:.2f} s\n within 0.5%: {top6.min():.2f}-{top6.max():.2f} s', fontsize=8, va='bottom')
ax.set_ylabel('score per second (expected)'); ax.legend(frameon=False, fontsize=8, loc='center right')
ax.set_title('(a) Score per second against Δ', fontsize=10)
ax = axes[0, 1]
ax.plot(sweep6['delta'], sweep6['margin'], color=C_PERIPH, lw=2)
ax.axhline(0, color=C_INK, ls='--', lw=1); ax.text(3.8, 0, 'central strategy (tied) 0', ha='right', va='bottom', fontsize=8)
ax.axvline(D_BEST6, color=C_REF, ls=':', lw=1.5)
ax.set_ylabel('peripheral minus central score per second'); ax.set_title('(b) Margin', fontsize=10)
ax = axes[1, 0]
for k, (m_, g) in enumerate(per_model.groupby('group')):
    col = sns.color_palette('muted')[k]
    ax.plot(g['delta'], g['periph_score_s'], color=col, lw=1.5, label=f'positions from model {m_}')
    b = g.loc[g['periph_score_s'].idxmax()]
    ax.plot(b['delta'], b['periph_score_s'], 'o', color=col, ms=6)
ax.set_ylabel('peripheral score per second'); ax.legend(frameon=False, fontsize=8, ncol=2)
ax.set_title('(c) Each set of start positions', fontsize=10)
ax = axes[1, 1]
ok = ~np.isnan(stop)
ax.hist(stop[ok], bins=np.arange(0, 3.85, 0.05), weights=W_PROB[ok], color=C_PERIPH)
ax.set_ylabel('share of trials that stop ending at Low'); ax.set_title('(d) Where races switch from Low to High', fontsize=10)
for ax in axes.ravel():
    ax.set_xlabel('threshold Δ (s)'); ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.show()

# check the choice of delta by leaving one model out
loo = []
for m_ in sorted(races['group'].unique()):
    train = per_model[per_model['group'] != m_].groupby('delta')['periph_score_s'].mean()
    d_tr = float(train.idxmax())
    test = per_model[per_model['group'] == m_].set_index('delta')
    loo.append(dict(held_out_positions_from_model=m_, delta_chosen_on_others=d_tr, score_at_that_delta=test.loc[d_tr, 'periph_score_s'],
                    best_score_for_this_model=test['periph_score_s'].max(), its_own_best_delta=float(test['periph_score_s'].idxmax()),
                    margin_at_chosen_delta=test.loc[d_tr, 'periph_score_s'] - test.loc[d_tr, 'central_score_s']))
display(pd.DataFrame(loo).set_index('held_out_positions_from_model').round(4))
''')

md(r'''
**How to read the table.** Each row leaves out the start positions from one peripheral model, chooses Δ on the other five sets, and scores that Δ on the left-out set, beside the best that set could get with its own best Δ.
''')

md(r'''
## 8. Score rate across the steps and across trial configurations

**Motivation.** This section puts the steps side by side, and shows where in the arena the final model's score comes from.

**Method.** The steps in order (central agent with rays from step 2 onward; each rule at its own best Δ where it has one):
- 1 Full-knowledge: both agents know both walls at onset and go straight (section 2, best Δ).
- 2 + central rays: the central agent senses with rays and infers its search direction (section 3). The peripheral agent still knows both walls.
- 3 Peripheral rays, under 'choose High': the peripheral agent also senses with rays (sections 4-6). Once it senses, it needs a rule that uses only what it has sensed, so this step uses the simplest one, always going High. The drop from step 2 therefore combines two changes: sensing with rays, and losing the Δ rule.
- 4 Nearest once both sensed (section 5).
- 5 Δ rule with inference about both walls (section 7, best Δ).

Two components of sections 4-5 are left out of this figure. Turning onto the path barely changes the score, and finding a wall chosen with full knowledge is not playable (the choice uses where the walls are before they are sensed).

**Terms.** Trial type: 45°, 90° or 180° between High and Low. Layout group: where High is relative to the peripheral agent's alcove; instant means the agent starts on a lit trigger.

**How to read the figures.** Figure 1: peripheral (blue) and central (grey) score per second for each step, with central vs central (dashed) for the central agent of that step. Figure 2: peripheral expected score per trial in each layout (High's and Low's walls from own alcove) for steps 1, 3, 4 and 5. Figure 3: for the final rule, who wins and at which wall in each layout group (left), and each group's contribution to score per trial (right).

**Why it is plotted.** Figure 1 shows which components cost or gain score. Figures 2 and 3 show which layouts the peripheral agent wins, and why its score stays close to the central agent's.

**How to read the tables.** First: the steps' scores. Second: score per trial by trial type for step 1 and the final rule. Third: wins and score by layout group for the final rule.
''')

code(r'''
steps = pd.DataFrame({
    '1 full-knowledge': step1.loc[f'best Δ = {D_BEST0:.2f} s'],
    '2 + central rays': step2.iloc[1],
    "3 peripheral rays, under 'choose High'": step5.loc['High, always'],
    '4 nearest once both sensed': step5.loc['Nearest once both sensed'],
    f'5 Δ rule, both walls inferred (Δ {D_BEST6:.2f} s)': summary(pooled, 'delta2', D_BEST6, CT_INF)}).T
refs = [ref_score(CT_STRAIGHT)] + [ref_score(CT_INF)] * 4
display(steps.round(4))

fig, ax = plt.subplots(figsize=(18, 9))
x = np.arange(len(steps))
ax.bar(x - 0.18, steps['peripheral score/s'], width=0.36, color=C_PERIPH, label='peripheral')
ax.bar(x + 0.18, steps['central score/s'], width=0.36, color=C_CENTRAL, label='central')
for xi, (a_, b_, r_) in enumerate(zip(steps['peripheral score/s'], steps['central score/s'], refs)):
    ax.text(xi - 0.18, a_ / 2, f'{a_:.4f}', ha='center', fontsize=7, rotation=90, va='center', color='white')
    ax.text(xi + 0.18, b_ / 2, f'{b_:.4f}', ha='center', fontsize=7, rotation=90, va='center', color='white')
    ax.plot([xi - 0.4, xi + 0.4], [r_, r_], color=C_INK, ls='--', lw=1)
ax.set_xticks(x, ['1 full-\nknowledge', '2 + central\nrays', "3 peripheral rays,\n'choose High'", '4 nearest once\nboth sensed', '5 Δ rule, both\nwalls inferred'], fontsize=8)
ax.set_ylabel('score per second (expected)'); ax.set_ylim(0, steps[['peripheral score/s', 'central score/s']].max().max() * 1.25)
ax.legend(frameon=False, fontsize=8, loc='upper right'); ax.spines[['top', 'right']].set_visible(False)
ax.set_title('Score per second at each step (dashed: central vs central)', fontsize=10)
presentation_text(fig)
plt.tight_layout(); plt.show()


def layout_grid(rows, policy, delta=None, central_t=None):
    """ Return the peripheral expected score per trial in each layout (Low row x High column, walls from own alcove). """

    ev = ar.evaluate(rows, policy, delta, central_t)
    g = pd.DataFrame(dict(h=rows['high_off'].abs(), l=rows['low_off'].abs(), w=rows['prob'], s=ev['periph_score']))
    return g.groupby(['l', 'h']).apply(lambda x: np.average(x['s'], weights=x['w']), include_groups=False).unstack().reindex(
        index=range(5), columns=range(5))


grids = [('1 full-knowledge', layout_grid(with_times(D_KNOWN[pi_, hi_] / 10, D_KNOWN[pi_, lo_] / 10, GAP_KNOWN > D_BEST0), 'low_first')),
         ("3 peripheral rays, under 'choose High'", layout_grid(pooled_inf, 'high', central_t=CT_INF)),
         ('4 nearest once both sensed', layout_grid(pooled_inf, 'nearest', central_t=CT_INF)),
         ('5 Δ rule, both walls inferred', layout_grid(pooled, 'delta2', D_BEST6, CT_INF))]
fig, axes = plt.subplots(1, 4, figsize=(19, 4.6), sharey=True)
for ax, (lab, grid) in zip(axes, grids):
    im = ax.imshow(grid.to_numpy(), origin='lower', cmap='Blues', vmin=0, vmax=1)
    for (li, hi), v in np.ndenumerate(grid.to_numpy()):
        if np.isfinite(v):
            ax.text(hi, li, f'{v:.2f}', ha='center', va='center', fontsize=8, color='white' if v > 0.6 else 'black')
    ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xlabel('High: walls from own alcove'); ax.set_title(lab, fontsize=10)
axes[0].set_ylabel('Low: walls from own alcove')
fig.colorbar(im, ax=axes, label='peripheral expected score per trial', shrink=0.85)
plt.show()

# score per trial by trial type, for step 1 and the final rule
def by_type(rows, policy, delta=None, central_t=None):
    """ Return peripheral and central score per trial by trial type (45, 90, 180 degrees). """

    ev = ar.evaluate(rows, policy, delta, central_t)
    t = rows['sep'].map({1: '45°', 2: '90°', 4: '180°'})
    g = pd.DataFrame(dict(t=t, w=rows['prob'], p=ev['periph_score'], c=ev['central_score']))
    return g.groupby('t').apply(lambda x: pd.Series(dict(share=x['w'].sum(), peripheral=np.average(x['p'], weights=x['w']),
                                                         central=np.average(x['c'], weights=x['w']))), include_groups=False)


types = pd.concat({'1 full-knowledge': by_type(with_times(D_KNOWN[pi_, hi_] / 10, D_KNOWN[pi_, lo_] / 10, GAP_KNOWN > D_BEST0), 'low_first'),
                   '5 Δ rule, both walls inferred': by_type(pooled, 'delta2', D_BEST6, CT_INF)}, axis=1).reindex(['45°', '90°', '180°'])
display(types.round(3))

# who wins, at which wall, by where High is (final rule)
ev7 = ar.evaluate(pooled, 'delta2', D_BEST6, CT_INF)
inst = ((pooled['t_high'] == 0) | (pooled['t_low'] == 0)).to_numpy()
low7 = ends_at_low(ev7, pooled)
win7 = ev7['periph_win'].to_numpy()
GROUPS = ['instant (on a lit trigger)', 'High own alcove', 'High adjacent', 'High 2+ walls away']
grp = np.select([inst, pooled['high_off'].abs() == 0, pooled['high_off'].abs() == 1], GROUPS[:3], GROUPS[3])
parts = pd.DataFrame(dict(grp=grp, share=W_PROB, ph=W_PROB * win7 * ~low7, pl=W_PROB * win7 * low7, cw=W_PROB * (1 - win7)))
outcome = parts.groupby('grp')[['share', 'ph', 'pl', 'cw']].sum().reindex(GROUPS)
outcome['ps'], outcome['cs'] = outcome['ph'] + 0.4 * outcome['pl'], outcome['cw']
outcome.loc['total'] = outcome.sum()
outcome.columns = ['share of trials', 'peripheral wins at High', 'peripheral wins at Low', 'central wins (High)',
                   'peripheral score/trial', 'central score/trial']
og = outcome.loc[GROUPS]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
ax = axes[0]
xg = np.arange(len(GROUPS))
ax.bar(xg, og['peripheral wins at High'], color=C_PERIPH_DARK, width=0.6, label='peripheral wins at High (1.0)')
ax.bar(xg, og['peripheral wins at Low'], bottom=og['peripheral wins at High'], color=C_PERIPH_LIGHT, width=0.6, label='peripheral wins at Low (0.4)')
ax.bar(xg, og['central wins (High)'], bottom=og['peripheral wins at High'] + og['peripheral wins at Low'], color=C_CENTRAL, width=0.6,
       label='central wins at High (1.0)')
for xi, (ph, pl, cw) in enumerate(zip(og['peripheral wins at High'], og['peripheral wins at Low'], og['central wins (High)'])):
    for base, v, dark in ((0, ph, True), (ph, pl, False), (ph + pl, cw, True)):
        if v > 0.02:
            ax.text(xi, base + v / 2, f'{v:.3f}', ha='center', va='center', fontsize=8, color='white' if dark else 'black')
ax.set_xticks(xg, GROUPS, fontsize=8); ax.set_ylabel('share of all trials'); ax.legend(frameon=False, fontsize=8, loc='upper left')
ax.set_title('Who wins, by where High is (final rule)', fontsize=10)
ax = axes[1]
ax.bar(xg - 0.18, og['peripheral score/trial'], width=0.36, color=C_PERIPH, label='peripheral')
ax.bar(xg + 0.18, og['central score/trial'], width=0.36, color=C_CENTRAL, label='central')
for xi, (a_, b_) in enumerate(zip(og['peripheral score/trial'], og['central score/trial'])):
    ax.text(xi - 0.18, a_ + 0.005, f'{a_:.3f}', ha='center', fontsize=7)
    ax.text(xi + 0.18, b_ + 0.005, f'{b_:.3f}', ha='center', fontsize=7)
ax.set_xticks(xg, [g.replace(' (on a lit trigger)', '').replace('High ', '') for g in GROUPS], fontsize=8)
ax.set_ylabel('contribution to score per trial'); ax.legend(frameon=False, fontsize=8)
ax.set_title(f"Score per trial: {outcome.loc['total', 'peripheral score/trial']:.3f} vs {outcome.loc['total', 'central score/trial']:.3f}",
             fontsize=10)
for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.show()
display(outcome.round(3))
''')

md(r'''
### How random headings and search directions enter the score

**Motivation.** The central agent's heading is random, and both agents' search directions can be. These change the time to a wall, and so who wins. This subsection shows how the model turns them into a score.

**Method.** Nothing is sampled. Every random element is enumerated over a fine grid, or over both of its cases, and averaged exactly.
1. **Central heading.** The central agent's time to High is computed for 3,600 evenly spaced headings (every 0.1°), each equally likely. This is done once for each Low position relative to High, because the inferred search depends on where Low is (`ar.central_times_inferred`). Where the two search directions tie, both are kept at half weight.
2. **Peripheral start.** The position and heading are the logged slice-onset values, so the peripheral agent has one time per race row.
3. **Peripheral search direction.** Each race row is one start position × one layout × one search direction (two rows per position and layout). A row's weight is the layout probability, shared equally over start positions and the two directions. When inference picks a direction, that row takes the whole weight. When the two directions tie, both keep half.
4. **Win and race time** (`ar._against_central`). Each peripheral row is compared with every central time for its layout. The peripheral win is the share of central times later than the peripheral time (an exact tie counts half). The race time is the mean of the earlier arrival over all central times.
5. **Score.** A row's expected score is win share × wall value for the peripheral agent, and (1 − win share) × 1.0 for the central agent. Score per second is the weighted total score over the weighted total of (race time + time outside the race).

**Terms.**
- `win share`: the share of central headings the peripheral agent beats, from 0 (always loses) to 1 (always wins).
- `decided by central heading`: a race with win share strictly between 0 and 1.

**How to read the first table.** One row per peripheral arrival time, one column per Low position relative to High, in walls (+ = clockwise). Each value is the share of central headings that arrive later, so it is the peripheral agent's win share at that time.

**How to read the second table.** The final rule (Δ rule, both walls inferred, at the best Δ), with trials split by whether the central heading decides them. Columns are the share of trials, each agent's contribution to score per trial, and the mean peripheral win share.
''')

code(r'''
# get the share of central headings arriving after each peripheral time, by Low position
t_ex = [1.8, 1.9, 2.0, 2.2, 2.4, 2.6]
share_later = pd.DataFrame({k: [float((v > t).mean()) for t in t_ex] for k, v in sorted(CT_INF.items())},
                           index=pd.Index(t_ex, name='peripheral arrival (s)'))
share_later.columns = [f'Low {int(ar.signed_offset(k)):+d}' if k != 4 else 'Low 4' for k in share_later.columns]
display(share_later.round(3))
print(f"earliest central arrival {min(v.min() for v in CT_INF.values()):.2f} s")

# split the final rule's trials by whether the central heading decides them
ev_f = ar.evaluate(pooled, 'delta2', D_BEST6, CT_INF)
w_f, pw_f = pooled['prob'].to_numpy(), ev_f['periph_win'].to_numpy()
kind = np.where(pw_f >= 1, 'peripheral always wins', np.where(pw_f <= 0, 'peripheral always loses', 'decided by central heading'))
split = pd.DataFrame({'w': w_f, 'kind': kind, 'pw': pw_f,
                      'ps': w_f * ev_f['periph_score'].to_numpy(), 'cs': w_f * ev_f['central_score'].to_numpy()})
split = split.groupby('kind')[['w', 'pw', 'ps', 'cs']].apply(lambda g: pd.Series({
    'share of trials': g['w'].sum(), 'peripheral score/trial': g['ps'].sum(), 'central score/trial': g['cs'].sum(),
    'mean peripheral win share': np.average(g['pw'], weights=g['w'])}))
split.loc['total'] = [split['share of trials'].sum(), split['peripheral score/trial'].sum(), split['central score/trial'].sum(),
                      np.average(pw_f, weights=w_f)]
display(split.round(3))
''')

md(r'''
**Interpretation.**
- The central agent never arrives before 1.82 s, so a faster peripheral arrival always wins. Later arrivals win a fraction of the time, which falls quickly. At 2.0 s the peripheral agent beats 0.40-0.47 of central headings, and at 2.4 s only 0.04-0.09 (first table).
- In the final rule, the central heading decides only 0.041 of trials, where the peripheral agent wins 0.418 on average (second table). The peripheral agent always wins 0.564 and always loses 0.395. These are decided by geometry alone, mainly by whether the lit wall is next to the peripheral agent's alcove.
- The contested trials add 0.014 of the peripheral agent's 0.416 score per trial and 0.024 of the central agent's 0.419 (second table). So the result does not hinge on how the central heading is modelled.
''')

md(r'''
## 9. Known problems and limits

From an audit of every step of the model (2026-10-22; details in `READMEs/analytical_modelling_notes.md`).

**Fixed before this notebook.**
1. The stored time at which each wall is first sensed was, in 0.24% of races, taken from a hypothetical stepped-out position instead of the agent's start position, so a wall could count as sensed at onset when it was not (an information leak). It affected the inferred search direction of the High-only rules and the "Low if seen first" rule by at most 0.0005 per second (with the earlier 7.09 s time outside the race); the final Δ rule was not affected.
2. At Δ = 0, two equally distant walls were split by rounding error. Ties now go to High.

**Remaining limits, to keep in mind.**
3. The two agents are not optimised equally. The peripheral agent's Δ is tuned and it re-decides at every new wall. The central agent always goes High, chooses its search direction once, and stands still while High is unseen. Near-equal scores compare a tuned peripheral player with a stereotyped central one.
4. The choice compares expected distances, leaving out the time needed to find an unseen wall. This was chosen deliberately: searching takes about 0.15-0.27 s, and adding it would be like shifting Δ by that much for some decisions, over a range (0.30-1.15 s) where the score barely changes.
5. Decisions are revisited only when a new wall is sensed, and a search keeps its turning direction once chosen (no reversing, by choice; see section 6).
6. When the peripheral agent moves out of its alcove to search, it moves straight out along the alcove's axis. Moving out at an angle toward the side where the wanted wall probably is could reveal it slightly sooner; this is a possible refinement, not an error.
7. Peripheral start positions are logged self-play positions from six trained models; humans wait in different places.
8. Turning and closing rates come from trained central agents; humans are likely slower.
''')

md(r'''
## Interpretation

**Step 1: geometry alone.** From an alcove back the adjacent walls are 14.4 units away against the central agent's 18.25; every other wall is 26 units or more. With both walls known at onset, the best threshold (Δ = 1.00 s) gives the peripheral agent 0.0546 per second against the central agent's 0.0523, a lead of 0.0024. Always going High gives 0.0462 against 0.0722.

**Step 2: the central agent senses.** Sensing and turning raise the central agent's mean time to High from 1.82 s to 2.03 s. Against the step-1 peripheral choice it scores 0.0515 (peripheral 0.0543), so sensing costs the central agent slightly more than it costs the race.

**Step 3: the peripheral agent senses.** From an alcove the rays reach an adjacent wall at onset in only 0.13 of agent-trials, and finding one raises the mean time to it from 1.46 s to 1.59 s. Turning onto the path costs almost nothing (0.0542 against 0.0515). Having to find the chosen wall turns the lead into a small deficit (0.0524 against 0.0535), even with the choice still made from knowledge the agent could not have.

**Step 4: choosing from partial knowledge.** With rules that use only what has been sensed, "High, always" scores 0.0444 against 0.0718, and "Low if seen first" only 0.0286, because from an alcove a far Low is often seen before a hidden adjacent High (example: Low at 3.47 s, lost, instead of High at 1.47 s, won). "Nearest once both sensed" does best of the three, 0.0501 against 0.0578.

**Step 5: inference.** Inferring the search direction cuts the search for a hidden High from 0.254 s to 0.148 s, but barely changes the scores ("nearest" 0.0501 against 0.0578 without, 0.0502 against 0.0579 with). The layout rules often cannot tell which side a hidden adjacent High is on: in the example, the wrong side has probability 0.67.

**Step 6: the Δ rule with inference about both walls.**
- At the best Δ of 1.05 s (any Δ from 0.25 to 1.15 s is within 0.5%), the peripheral agent scores 0.0527 per second against the central agent's 0.0531, a margin of -0.0004 (0.7% behind). Central vs central is 0.0591, so both agents score about 10-11% less against each other than central agents do against each other.
- The gain over "nearest" comes from inferring an unseen Low. With High seen three walls away and Low adjacent and hidden, expected score per trial in that layout rises from 0.21 ("nearest") to 0.38 (section 8, figure 2), close to the 0.40 of step 1.
- Δ is not tuned to particular start positions: chosen on five sets and scored on the sixth, it loses at most 0.0002 (positions from model 20). The steps in the Δ curve sit where many races stop taking Low together, near 0.8 s, 1.17 s (Low adjacent with High two walls away) and 1.8-2.1 s.
- The rule set is the same throughout; only the start positions differ. At the chosen Δ, the agent out-scores its central opponent from the positions of models 20, 23, 32 and 34 (margin +0.0012 to +0.0031) and trails from those of models 3 and 15 (-0.0049 and -0.0055). Model 20 waits nearer its alcove mouth, where adjacent walls are more often in view; why positions from models 3 and 15 do worse than the similar positions from model 23 is not yet established.

**Where the score comes from (final rule).** The peripheral agent wins 0.305 of trials at High and 0.276 at Low, the central agent 0.419, all at High; score per trial is 0.416 against 0.419. The peripheral agent wins nearly all adjacent-High layouts (0.179 of 0.192 of trials) and all instant ones (0.233), while the central agent wins most layouts with High two or more walls away (0.409 of 0.567), where the peripheral agent takes Low in 0.156. By trial type, the peripheral agent loses the 45° trials (0.348 against 0.504 per trial) and wins the 90° and 180° trials (0.452 against 0.404; 0.515 against 0.263).

**How far to trust this.** The near-equal score rates hold under the model's assumptions, and the audit (section 9) found no remaining information leaks. But they compare a peripheral player with a tuned threshold and full use of inference against a central player that always goes High, decides its search once, and waits while High is unseen. A central player with its own threshold or a better search could do better. The result also depends on where the peripheral agent starts (logged positions from six trained models) and on agent-like turning and closing rates.
''')

md(r"""
## 10. Summary figures for a short presentation

**Motivation.** Four figures that explain the model and its main result quickly: what one race looks like, where in the arena each strategy wins, the head-to-head score rate, and why the two strategies end up close. All use the final model (Δ rule with inference about both walls, Δ at its best).

**Terms.**
- Peripheral agent: waits at the back of its alcove (logged positions of trained peripheral agents). Central agent: waits at the arena centre with a random heading, and always goes High.
- High is worth 1.0 and Low 0.4. The first agent to reach either lit wall takes its points; the other scores nothing.
- Score per second: expected points per trial over expected trial time (race plus 6.55 s outside it).
""")

md(r"""
### Figure A. One race, three outcomes

**How to read the figure.** Each panel is one layout, seen from above, with the peripheral agent at the back of the top alcove (black square) facing into the arena, and the central agent at the centre (grey dot). High is red and Low blue. The peripheral agent's route is black (dashed while it searches), and the central agent's route to High is orange. The titles give each agent's arrival time and the share of races the peripheral agent wins (over the central agent's random headings).

**Why it is plotted.** It shows what the model computes in a single trial, and the three ways a trial can go. (a) High is next to the peripheral agent, which wins it. (b) High is far but Low is next to the peripheral agent, which takes the smaller reward first. (c) Both walls are far, and the central agent wins.
""")

code(r"""
examples = ((1, 3, '(a) High next to the peripheral agent'), (3, 1, '(b) High far, Low next to it'),
            (4, 5, '(c) Both walls far'))
fig, axes = plt.subplots(1, 3, figsize=(24, 9))
for ax, (hg, lw_, lab) in zip(axes, examples):
    draw_arena(ax, hg, lw_)
    r = ar.delta2_route(BACK, H_OUT, hg, lw_, 1, D_BEST6, ROT, CLOSE)
    for state, s0, e, t0, is_search in r['segments']:
        ax.plot(*route_xy(state, s0, e).T, color=C_INK, lw=4, ls='--' if is_search else '-')
    pts_c = ar.path_points(np.zeros(2), hg)
    ax.plot(pts_c[:, 0], pts_c[:, 1], color=C_ROUTE_C, lw=4)
    ax.plot(0, 0, 'o', color=C_CENTRAL, ms=13)
    ax.plot(*BACK, 's', color=C_INK, ms=12)
    ct = CT_INF[(lw_ - hg) % 8]
    ws = win_share(r['time'], hg, lw_, CT_INF)
    ax.set_title(f"{lab}\nperipheral {r['time']:.2f} s ({'Low' if r['go_low'] else 'High'}), central {ct.mean():.2f} s\n"
                 f"peripheral wins {ws:.2f}", fontsize=13)
presentation_text(fig)
plt.tight_layout(); plt.show()
""")

md(r"""
### Figure B. Who wins, by where High is

**How to read the figure.** The arena from above, with the peripheral agent's own alcove at the top. The outline is red because each wall is read as the case where High is at that wall. In front of each wall, the red circle is the probability the peripheral agent wins that trial at High, and the blue circle the probability it wins at Low (circle area is proportional to probability; the value is inside). The black number, between the circles and the wall, is the total probability the peripheral agent wins; the central agent wins the rest. Values under 0.02 are not drawn. Values are averaged over all logged start positions and every Low position. With High next to the peripheral agent, its wins at Low are mostly trials with Low on its own wall, which it wins at once. The table gives the same values (`high_off` is High's wall counted from the peripheral agent's own wall, positive clockwise).

**Why it is plotted.** It shows the two strategies' territories. The peripheral agent owns its own wall and the two next to it. The central agent wins most trials with High further away, though the peripheral agent takes some of them at Low.
""")

code(r"""
ev_f = ar.evaluate(pooled, 'delta2', D_BEST6, CT_INF)
w_f = pooled['prob'].to_numpy()
win_f = ev_f['periph_win'].to_numpy()
at_low = ev_f['go_low'].to_numpy()
terr = pd.DataFrame({'high_off': pooled['high_off'].to_numpy(), 'w': w_f, 'p_high': w_f * win_f * ~at_low,
                     'p_low': w_f * win_f * at_low, 'c': w_f * (1 - win_f),
                     'ps': w_f * ev_f['periph_score'].to_numpy(), 'cs': w_f * ev_f['central_score'].to_numpy()})
terr = terr.groupby('high_off').sum()
terr = terr.div(terr['w'], axis=0).assign(share=terr['w'])
terr.columns = ['_w', 'peripheral wins at High', 'peripheral wins at Low', 'central wins', 'peripheral score/trial',
                'central score/trial', 'share of trials']
terr = terr.drop(columns='_w')
terr['peripheral wins'] = terr['peripheral wins at High'] + terr['peripheral wins at Low']

fig, ax = plt.subplots(figsize=(12, 12))
# draw every wall red (in each wall's case, that wall is High)
for (a_, b_), w in zip(WALL_SEGS, WALL_IDS):
    ax.plot([a_[0], b_[0]], [a_[1], b_[1]], color=C_HIGHW, lw=3.0)
S_MAX = 5200                                     # marker area (points^2) for a probability of 1
for w in range(8):
    row = terr.loc[int(ar.signed_offset(w))]
    # place a red (High) and a blue (Low) circle in front of the wall, sized by probability
    # leave out values too small to draw (under 0.02), and centre a lone circle
    shown = [(val, col) for val, col in ((row['peripheral wins at High'], C_HIGHW), (row['peripheral wins at Low'], C_LOWW))
             if val >= 0.02]
    offs = [0.0] if len(shown) == 1 else [-1.6, 1.6]
    for q_off, (val, col) in zip(offs, shown):
        xy = ar._pt(11.0, q_off, w)
        ax.scatter(*xy, s=S_MAX * val, color=col, zorder=3)
        ax.text(*xy, f'{val:.2f}', ha='center', va='center', fontsize=10, color='white', zorder=4)
    # write the total between the circles and the wall
    top = ar._pt(14.4, 0.0, w)
    ax.text(*top, f"{row['peripheral wins']:.2f}", ha='center', va='center', fontsize=14, color=C_INK)
ax.plot(*BACK, 's', color=C_INK, ms=12); ax.plot(0, 0, 'o', color=C_CENTRAL, ms=12)
ax.text(BACK[0] + 3.3, BACK[1], 'peripheral', fontsize=12, ha='left', va='center'); ax.text(0.9, 0, 'central', fontsize=12, va='center')
ax.scatter([], [], s=300, color=C_HIGHW, label='peripheral wins at High')
ax.scatter([], [], s=300, color=C_LOWW, label='peripheral wins at Low')
ax.legend(frameon=False, fontsize=12, loc='lower left', bbox_to_anchor=(0.0, -0.06))
ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.axis('off')
for sp_ in ax.spines.values():
    sp_.set_visible(False)
ax.set_title('Peripheral win probability when High is at each wall\n(black: total; circles: at High and at Low)', fontsize=14)
presentation_text(fig, inner=14)
plt.tight_layout(); plt.show()
display(terr.round(3))
""")

md(r"""
### Figure C. The head-to-head result

**How to read the figure.** (a) Score per second for the peripheral and the central agent playing each other, with central vs central play (dashed) for reference. (b) Where each agent's points per trial come from. The peripheral agent's bar is split into points from High wins (1.0 each) and from Low wins (0.4 each). The labels above the bars give the share of races each agent wins.

**Why it is plotted.** (a) is the main result: the two strategies score almost equally per second against each other, and both score less than in central vs central play. (b) shows how they get there by different routes. The peripheral agent wins more races, but many of its wins are worth only 0.4.
""")

code(r"""
fin = steps.iloc[-1]
ref_f = ref_score(CT_INF)
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
ax = axes[0]
ax.bar([0, 1], [fin['peripheral score/s'], fin['central score/s']], color=[C_PERIPH, C_CENTRAL], width=0.55)
for x_, v in enumerate([fin['peripheral score/s'], fin['central score/s']]):
    ax.text(x_, v + 0.0008, f'{v:.4f}', ha='center', fontsize=15)
ax.axhline(ref_f, color=C_INK, ls='--', lw=2)
ax.text(1.85, ref_f + 0.0012, f'central vs central {ref_f:.4f}', ha='right', va='bottom', fontsize=12)
ax.set_xticks([0, 1], ['peripheral', 'central']); ax.set_xlim(-0.5, 1.9)
ax.set_ylabel('score per second (expected)'); ax.set_ylim(0, ref_f * 1.2)
ax.set_title('(a) Peripheral vs central, head to head', fontsize=15)
ax = axes[1]
tot = outcome.loc['total']
ph, pl, cw = tot['peripheral wins at High'], tot['peripheral wins at Low'], tot['central wins (High)']
ax.bar(0, ph, color=C_PERIPH_DARK, width=0.55, label='from High wins (1.0)')
ax.bar(0, pl * ar.LOW_SCORE, bottom=ph, color=C_PERIPH_LIGHT, width=0.55, label='from Low wins (0.4)')
ax.bar(1, cw, color=C_CENTRAL, width=0.55)
ax.text(0, ph + pl * ar.LOW_SCORE + 0.01, f'{ph + pl * ar.LOW_SCORE:.3f} per trial\nwins {ph + pl:.2f} of races', ha='center', fontsize=13)
ax.text(1, cw + 0.01, f'{cw:.3f} per trial\nwins {cw:.2f} of races', ha='center', fontsize=13)
ax.set_xticks([0, 1], ['peripheral', 'central']); ax.set_ylim(0, 0.6)
ax.set_ylabel('expected points per trial'); ax.legend(frameon=False, fontsize=12, loc='upper right')
ax.set_title('(b) Where the points come from', fontsize=15)
for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=13, width=2, length=7)
    ax.yaxis.label.set_size(14)
    for sp_ in ax.spines.values():
        sp_.set_linewidth(2)
presentation_text(fig)
plt.tight_layout(); plt.show()
""")

md(r"""
### Figure D. The result depends on the trial type

**How to read the figure.** Points per trial for each agent by the angle between High and Low (45°, 90°, 180°), for the final model. The labels under the axis give how often each trial type occurs.

**Why it is plotted.** It shows that the balance comes from averaging over trial types. The central agent wins the common 45° trials, where Low sits next to High. The peripheral agent wins the 90° and 180° trials, where Low is more often near its own alcove when High is far.
""")

code(r"""
tt = types['5 Δ rule, both walls inferred']
fig, ax = plt.subplots(figsize=(12, 8))
xt = np.arange(len(tt))
ax.bar(xt - 0.19, tt['peripheral'], width=0.38, color=C_PERIPH, label='peripheral')
ax.bar(xt + 0.19, tt['central'], width=0.38, color=C_CENTRAL, label='central')
for x_, (a_, b_) in enumerate(zip(tt['peripheral'], tt['central'])):
    ax.text(x_ - 0.19, a_ + 0.012, f'{a_:.2f}', ha='center', fontsize=15)
    ax.text(x_ + 0.19, b_ + 0.012, f'{b_:.2f}', ha='center', fontsize=15)
ax.set_xticks(xt, [f'{t}\n({s:.0%} of trials)' for t, s in zip(tt.index, tt['share'])])
ax.tick_params(labelsize=15, width=2, length=7)
for sp_ in ax.spines.values():
    sp_.set_linewidth(2)
ax.set_ylabel('expected points per trial', fontsize=16); ax.set_ylim(0, 0.65); ax.legend(frameon=False, fontsize=15)
ax.spines[['top', 'right']].set_visible(False)
ax.set_title('Points per trial by trial type (final model)', fontsize=17)
presentation_text(fig, base=26)
plt.tight_layout(); plt.show()
""")

md(r"""
**Interpretation.** Values are in the figures and table above.
- A race is mostly decided by geometry. A wall next to the peripheral agent's alcove takes it 1.84 s even when it must search for it, against the central agent's mean of 2.02 s, so it wins 0.89 of these races, at High (Figure A, a) or at Low (b). With both walls far it needs 3.80 s and never wins (c).
- The peripheral agent wins all trials with High at its own wall (1.00) and 0.96 with High next to it. It wins only 0.25-0.38 with High further away, all at Low (Figure B and its table).
- Head to head, the two score almost the same per second: 0.0527 against 0.0531. Both are below central vs central play (0.0591; Figure C, a).
- They get there by different routes. The peripheral agent wins 0.58 of races to the central agent's 0.42, but part of its points come from Low wins worth 0.4. So its points per trial are 0.416 against 0.419 (Figure C, b).
- The balance is an average over trial types. The central agent wins the 45° trials (0.50 against 0.35 points per trial; half of all trials). The peripheral agent wins the 90° trials (0.45 against 0.40) and the 180° trials (0.51 against 0.26).
""")

nb = nbf.v4.new_notebook()
nb['cells'] = cells
nb['metadata'] = {'kernelspec': {'display_name': 'octagon_analysis', 'language': 'python', 'name': 'octagon_analysis'},
                  'language_info': {'name': 'python'}}
nbf.write(nb, OUT)
print('written', OUT)
