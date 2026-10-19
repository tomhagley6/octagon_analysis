## General Points
- You are allowed to say 'I don't know'
- Use direct quotes for factual grounding
- Verify claims with citations
- Prompt with questions about implementation freely
- Code changes should have clear tests before implementing
- Important tests and validations should be stored in the tests or validations folder
- Check for and make clear any assumptions made in analyis or data processing, 
  and query the user on the validity of holding those assumptions
- Any recorded interpretations or conclusions on analysis should be immediately preceded 
  by general data values and comparisons used to reach those interpretations and conclusions

# octagon_analysis

Python repo for behavioural and statistical analysis, plotting, and visualisation of
data logged in the Octagon task by both human players and ML-Agents RL agents.

## Related repos

Two other local repos make up this project and should be kept in context:

- **`~/Unity/Octagon`** — the Unity project for the Octagon game: an octagonal arena
  reward task where players/agents choose between coloured walls (High vs Low reward)
  at varying angular separations. Holds the game logic, netcode for human two-player
  sessions, and the (local, netcode-free) RL agent training and inference. Agent
  behaviour lives in `TrialLogic/` (`OctagonAgent.cs`, `OctagonWallTrigger.cs`); trained
  models, run configs, and logged data land under its `results/` and `simulations/`.
  Unity 2022.3.13f1, ML-Agents; its Python side uses the conda env `mlagents`.

- **`~/repos/agent_training`** — Python tooling for batch training and inference of the
  Unity ML-Agents models at scale (config patching, parameter sampling, HPC/SLURM launch
  in `run_training_hpc.py` / `*.slurm`, and a `tournament/` for model-vs-model runs).
  Produces the trained models and inference logs that this repo then analyses.

The typical flow: `agent_training` trains/infers models in the `Octagon` environment,
which logs data under `~/Unity/Octagon/{results,simulations}`; `octagon_analysis` reads
those logs.

## Comment & writing style

Applies to code comments, docstrings, and notebook markdown. Prefer clear, tight,
low-jargon writing. When in doubt, cut words.

- **Shorthand means tight wording, not fewer comments.** Comment most steps, as the
  repo source does; just keep each comment short. Drop "we"/"you" and state the thing
  directly.
  - Good: `Agent 0 reward is kept as the model's efficiency (both agents are the same model)`
  - Avoid: `We keep agent 0's reward as the model's efficiency, since in self-play both agents run the same model`

- **Inline comments: lowercase, verb-first, describe the next step.** Start with a verb
  (`get`, `find`, `take`, `compute`, `plot`), lowercase the first word; later sentences
  in the same comment take normal capitalisation. One comment per operation.
  - `# get the player's trajectory`, `# take the square root to get the euclidean distance`

- **Docstrings: capitalised, verb-first, with a `Takes` line.** Open with `Return...` /
  `Find...` / `Plot...`, then name the inputs on a short second line.
  - `Return the direct distance between slice-onset and trigger locations. Takes a trial and player id.`

- **Parentheticals for clarification are fine and encouraged** mid-sentence
  (`(timepoints)`, `(e.g. wall_1 has index 0)`). This differs from *trailing* asides,
  which should still be cut.

- **No "clause: [restatement]" structure.** Do not name a thing and then restate it
  after a colon. Write it as a plain statement.
  - Avoid: `The characterisation axis: centrality at slice onset`
  - Avoid: `places every model on one axis: where it stands when the trial starts`
  - Good: `The characterisation axis is centrality at slice onset`
  - A colon is fine only to introduce a genuine list of distinct items
    (e.g. `shortlist: extremes, medoid, best scorer`).

- **Cut trailing asides and editorialising.** No tacked-on "— the characterisation
  axis", no "This is the spatial read of the axis: ...", no timing notes like
  "~1-2 min". Say the substance and stop.

- **Less jargon.** Plain words beat terms like "tidy tables", "gross-strategy
  summary", "variance decomposition", "second independent axis of individual
  difference". Name the actual thing.

- **One idea per sentence.** Break dense run-ons into short, sequential sentences.

- **Say "agent", not "seat"** (self-play has two agents, both the same model).

- Markdown cells may stay readable prose, but keep them concise. Code comments should
  be the shortest form that is still clear.

## Analysis notebook structure

Use this markdown structure in every analysis notebook, including ones with only
tables. The worked example is
`demos/strategy_characterisation/humans/mixed_strategy_characterisation.ipynb`
(sections 4 and 5).

- **Notebook top.** The question, a numbered plan of sections, the choices and
  assumptions, and which modules compute the numbers.
- **Top of each section or subsection.**
  - Motivation. What question the section answers, and why this analysis answers it.
  - Method, in short plain steps.
  - A definition of every term and measure. Say what it is, its range, and how its
    values read (e.g. `0 = no trend, 1 = every trial on the line`), with the column name.
  - Choices and assumptions, each with its source (data anchor, simulation,
    literature, or the user's choice).
- **Before each figure.**
  - "How to read the figure". Say what one point or row is, and explain every axis,
    line, colour and panel, and why points may be missing.
  - "Why it is plotted". Say what to look for, and what each pattern would mean.
- **With each table.** "How to read the table" for any column that is not self-evident.
- **Interpretation cell after the outputs.** Short bullets, each citing values shown in
  the notebook. Compute reference or expected values and show them beside the data.
  State limits plainly.
- **Validate before applying.** A method that labels or classifies is first tested on
  simulated data with a known answer, and that check is shown in the notebook.
- **Figures.** Axis labels say exactly what is plotted. Short panel titles may stay as
  intuitive names. Label reference lines inside their panel, not in the figure title.

## Environment

Run the pipeline and notebooks with the conda env python:
`/home/tom/miniconda3/envs/octagon_analysis/bin/python`. The repo-local `.venv`
lacks matplotlib/scipy and will fail imports. Notebooks use the `octagon_analysis`
Jupyter kernel.
