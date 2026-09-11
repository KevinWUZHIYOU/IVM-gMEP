# Barnase--Barstar joint-E/F arc500 read-only acceptance checklist

This checklist is an acceptance gate, not an experiment authorization. A
result may be used in the densified cost-to-precision figure only when every
item below is verified from immutable files and all checks pass without an
OpenMM call.

## 1. Frozen identity and source path

- The evaluator, `bb_core.py`, topology, selection or full-state replay
  verification, and every selected source path are bound by path and SHA-256.
- ODE--NEB contains exactly the retained paths at optimization
  \(N_F=60,150,300,1320\) from the targeted R2 inventory.
- IVM contains exactly full-state replay calls 2, 5, 10, and 50 at optimization
  \(N_F=60,150,300,1329\). Each path is formally reportable and outside the
  quarantined terminal tail.
- R4 is rejected because its observer confused raw callback order with the
  deduplicated call axis. IVM accepts only the final canonical successor
  verification. That verification and the metric contract bind its identical
  seven-role closure: registration,
  authorization, reference result, replay result, replay runtime audit,
  capture manifest, and capture audit. R2, R3, and R4 are rejected.
- Every selected online shard retains its exact successor
  counter provenance: live schema-8, historical schema-8, or historical
  schema-7 with `inherited_cache_entries` explicitly unavailable and absent.

## 2. Original 32-image discrete metric

- ODE--NEB performs 30 fresh joint energy/full-force evaluations at the
  internal images. It performs no diagnostic evaluation at the two fixed
  endpoints.
- IVM reuses the already-paid 32-image path, energy, and full-force arrays from
  the verified full-state replay and performs zero new evaluation for this
  discrete metric.
- The stored internal forces reconstruct the manuscript definition of
  \(I_\perp\) through `bb_core.path_diagnostics`: central chord tangent and
  half-adjacent-chord-length weights.
- The path, energy, force, and reconstructed \(I_\perp\) agree exactly with
  their source hashes and registered values.

## 3. Uniform Cartesian arc-length-500 path

- Each original 32-image path is independently resampled at
  \(s_j=jL/499\), \(j=0,\ldots,499\), by piecewise-linear interpolation in
  unweighted Cartesian coordinates.
- The first and last dense coordinates are byte-identical to the two original
  endpoints.
- Every dense point receives exactly one fresh joint energy/full-force
  `getState` request. Thus each path stores arrays with shapes
  `dense_coordinates_A (500, 9396)`, `dense_energy_eV (500,)`, and
  `dense_force_eV_per_A (500, 9396)`.
- The densified \(\Delta E\) is reconstructed only from the 500 stored energy
  values. Dense forces are retained for traceability and do not enter either
  \(I_\perp\) or \(\Delta E\).

## 4. Durable physical-call and storage closure

- Each attempt has an fsynced `attempt_started` record before the physical
  request and a paired `attempt_succeeded` or `attempt_failed` record after it.
- A successful record binds the path, point index, coordinate SHA-256, energy
  value and scalar SHA-256, and full-force SHA-256.
- For every accepted path, all pointwise WAL records are checked against the
  authoritative shard coordinates, energies, and full forces.
- Failed, open, and repeated attempts are reported separately from the
  500-point metric lineage.
- A restart may reuse only a complete, atomically published, fully validated
  per-path shard. Individual successes from an incomplete path are never
  reused.
- The manifest binds each authoritative shard and its path, online or
  discrete energy/force, dense coordinate, dense energy, and dense force
  hashes. The aggregate is only a plotting index and does not duplicate dense
  coordinates or dense forces.

## 5. Optimization cost axis

- All work in this checklist is diagnostic: `optimization_N_F_contribution`
  must equal zero in the contract, shards, aggregate, manifest lineage, audit,
  and figure provenance.
- Dense endpoint calls are reported as diagnostic calls (two per dense path)
  but never added to optimization \(N_F\).
- The figure vertical axis continues to use the optimization `Number of
  Forces`; diagnostic calls are disclosed but excluded from that axis.

## PASS rule

The read-only validator must reject missing arrays, shape or value changes,
hash drift, an incomplete or conflicting WAL, an unverified IVM online state,
non-uniform or endpoint-changing densification, any nonzero optimization cost
contribution, or a figure contract that fails to bind the complete shard/WAL
dependency closure. Only an all-PASS result is eligible for plotting.
