## General Points
- You are allowed to say 'I don't know'
- Use direct quotes for factual grounding
- Verify claims with citations
- Prompt with questions about implementation freely
- Code changes should have clear tests before implementing

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

## Environment

Run the pipeline and notebooks with the conda env python:
`/home/tom/miniconda3/envs/octagon_analysis/bin/python`. The repo-local `.venv`
lacks matplotlib/scipy and will fail imports. Notebooks use the `octagon_analysis`
Jupyter kernel.
