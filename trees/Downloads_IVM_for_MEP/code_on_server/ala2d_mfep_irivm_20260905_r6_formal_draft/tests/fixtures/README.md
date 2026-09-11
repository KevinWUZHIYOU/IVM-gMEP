# Test-only causal state fixture

`causal_initial_state0.npz` is a 9 KiB byte-identical copy of the historical
state-zero fixture used by the V17 core tests.  It exists only so the analytic,
zero-MD state-machine tests are self-contained.  It is not listed as a sealed
formal input, is not a restrained-MD result, and is not formal evidence.

The future activation must separately import and SHA-256-bind the authoritative
causal initial observation under `inputs/reference/`; it may not treat this
test fixture as satisfying that gate.
