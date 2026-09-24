# Does the forward path describe the real log?

Yaw fit residual per frame, in degrees, over 60 s blocks from the first 300 s of sessions [0, 1, 2]. Regenerate with
`validation/network_compensation/forward_model_residual.py`.

| case | median | max | n blocks |
|---|---|---|---|
| real player 1 | 0.1067 | 0.2715 | 15 |
| player 0 through the path, exact times | 0.0085 | 0.0206 | 15 |
| player 0, millisecond timestamps | 0.0184 | 0.0331 | 15 |
| player 0, 59 Hz render clock | 0.0141 | 0.0276 | 15 |
