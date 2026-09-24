# What each player saw of the other

100 social sessions, measured at slice onset. `true` is where both players really were, `seen` is what that viewer's screen showed. Regenerate with
`validation/network_compensation/opponent_view_impact.py`.

| viewer | trials | visible, true | visible, on screen | angle error, median | angle error, p90 | screen disagrees |
|---|---|---|---|---|---|---|
| player 0 (host, renders 1 tick behind) | 13239 | 0.406 | 0.406 | 0.00 deg | 7.47 deg | 1.53% |
| player 1 (client, renders 2 ticks behind) | 13240 | 0.469 | 0.469 | 0.00 deg | 7.16 deg | 1.50% |

Where the screen and the truth disagree, they disagree both ways in about equal numbers, so the network does not bias how often an opponent appears visible:

| viewer | on screen but not really visible | really visible but not on screen |
|---|---|---|
| player 0 (host, renders 1 tick behind) | 96 | 107 |
| player 1 (client, renders 2 ticks behind) | 101 | 98 |

Paired over sessions, the extra tick the client carries does not make its screen measurably worse than the host's:

| measure | player 0 | player 1 | paired p |
|---|---|---|---|
| disagreement rate | 0.0151 | 0.0149 | 0.92 |
| median angle error, deg | 1.4712 | 1.0628 | 0.16 |

## What the error does to an analysis that conditions on the flag

The visible flag is a condition, so what matters is how often it is wrong, not whether the errors cancel in the marginal rate. A condition that is wrong on some trials pulls whatever is conditioned on it toward no effect, and the two error directions add to that rather than cancelling.

| viewer | sensitivity | specificity | attenuation factor | conditional effects shrink by |
|---|---|---|---|---|
| player 0 (host, renders 1 tick behind) | 0.9801 | 0.9878 | 0.9679 | 3.2% |
| player 1 (client, renders 2 ticks behind) | 0.9842 | 0.9856 | 0.9699 | 3.0% |

## The errors are not spread evenly

A disagreement needs the opponent to have moved between when their position was sent and when it was drawn, so every one of them sits on a trial where the opponent was already running at slice onset. Those are not ordinary trials.

| viewer | opponent speed at onset, disagreeing trials | agreeing trials | p |
|---|---|---|---|
| player 0 (host, renders 1 tick behind) | 9.20 units/s | 0.00 units/s | 9e-58 |
| player 1 (client, renders 2 ticks behind) | 9.92 units/s | 0.00 units/s | 7e-76 |

And they carry a different outcome with them, so the error is tied to the thing an analysis would be trying to explain:

| viewer | own-win rate, disagreeing trials | agreeing trials | p |
|---|---|---|---|
| player 0 (host, renders 1 tick behind) | 0.394 | 0.495 | 0.0055 |
| player 1 (client, renders 2 ticks behind) | 0.417 | 0.508 | 0.013 |

Slice onset is the most forgiving frame in the trial for this. Pooling both players, 60% of players are within half a unit per second of standing still at onset, and a filter applied to someone who is not moving changes nothing. That is why the median angle error is zero. The same measure taken later in the trial would not be as safe.

One asymmetry is worth keeping in mind when reading the table. For player 0 the `seen` side is the log itself, as the logged player 1 values already are the host's rendered copy, and the `true` side is the recovery. For player 1 it is the other way round: `true` is player 0 as logged and `seen` is simulated. The modelled side is not the same one for the two players.
