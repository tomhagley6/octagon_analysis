# Validation for the network compensation

Scripts that produce the tables in
[`READMEs/network_compensation.md`](../../READMEs/network_compensation.md).
They load the saved session pickles and take minutes, so they are not part of the unit test suite. The
fast checks live in `tests/test_network_compensation.py` and `tests/test_log_data_properties.py`.

Run from the repo root:

```
/home/tom/miniconda3/envs/octagon_analysis/bin/python validation/network_compensation/lambda_sweep.py
```

Each script writes a markdown table into `results/`. Those files are committed, so the write-up's numbers
can be traced back to the code that made them without rerunning anything.

| script | writes | answers |
|---|---|---|
| `forward_model_residual.py` | `results/forward_model_residual.md` | Does the modelled network path describe the real log, or is something missing from it? |
| `lambda_sweep.py` | `results/lambda_sweep.md` | What second-difference penalty should `solve_block` use? |
| `seat_comparison.py` | `results/seat_comparison.md` | How much of the gap between the two players does the recovery remove? |
| `assumption_errors.py` | `results/assumption_errors.md` | What does each shaky assumption in the forward path cost? |
| `identifiability.py` | `results/identifiability.md` | What can the recovery know at all, and is the extra turning in the data or in the penalty? |
| `totals_fidelity.py` | `results/totals_fidelity.md` | Recovery changes player 1's rotation total by about 11%. Which of the two is right? |
| `opponent_view_impact.py` | `results/opponent_view_impact.md` | How much does the network change whether a player could see their opponent? |

`_common.py` holds the shared loading and scoring helpers. `opponent_view_impact.py` reads the saved
opponent view table rather than building it; rebuild that first with
`human_characterisation.collect_opponent_views`.

## Why these three

The recovery undoes a known filter, so the obvious test is to run a known trace through that filter and
recover it. That test passes whatever the filter is, including a wrong one, and it also has no error for
the penalty to absorb, so it always picks the smallest penalty offered. All three scripts are built to
avoid depending on it.

`forward_model_residual.py` compares what the fit cannot explain on real player 1 frames against what it
cannot explain on a case where the path is exact by construction. A path missing a term shows up as a
residual the exact case never reaches.

`lambda_sweep.py` scores the penalty against the two time-axis errors the log really has, and against
held-out real frames, where no forward model stands between the answer and the data.

`seat_comparison.py` reports paired confidence intervals rather than p values. The two players do the same
task and show no difference in their solo sessions, so a gap in social is the logging path. A p value above
.05 only says a gap was not detected, which a noisy measure achieves by doing nothing. The interval says
what size of gap the data still allows, and that is what has to be held against the gap seen as logged.
