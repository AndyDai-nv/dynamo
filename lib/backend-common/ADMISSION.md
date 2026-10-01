<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Controller-managed discovery admission

`WorkerConfig.controller_managed = true` requires explicit controller permission
before an observed engine joins discovery. This is pool membership, not an
engine pause flag. The default is false; engines without a serving observation
source cannot opt in. Engine adapters must implement `verify_serving_admission`
to re-read actual engine identity/version/readiness; the default fails closed.

The worker installs these **POST** routes on its runtime system server only in
managed mode, after normal startup finishes:

- `/engine/admission/status`, body `{}`.
- `/engine/admission/admit`, body shown below.
- `/engine/admission/withdraw`, body with only `session` and `revision`.

```json
{
  "session": "<copy from status>",
  "revision": 1,
  "expected": {
    "engine_instance_id": 42,
    "observation_epoch": 1,
    "weight_version": "17"
  }
}
```

The controller first reads status, validates its run/worker association, and
uses the observed incarnation and observation epoch, but its **own expected
weight version**. It must not blindly approve whatever version status reports.
The administrative listener must be restricted to trusted controllers; these
identity fields are fencing tokens, not authentication credentials.

Each mutation needs a positive, strictly increasing revision within the returned
sidecar session. Retries with the exact same operation/body are idempotent and
only report current state. A same-revision different command or older revision
is rejected. A valid newer command supersedes previous permission even if its
subsequent engine verification fails or is cancelled. Retry a failed verification
with a **new** revision after fixing the cause; reusing its revision does not
turn failure into permission. Coordinate a single revision writer per sidecar.

`admit` checks the observation, bounds the engine's live verification to ten
seconds, rechecks the observation, and records intent under the same mutation
lock used by discovery and shutdown. It does not call engine resume. `withdraw`
revokes intent but does not pause or drain execution. Only the shared membership
task writes discovery. Regular engine resume cannot create controller permission.

Responses/status distinguish:

- `admitted`: a current controller grant exists (not necessarily routable).
- `engine_ready`: the latest observed engine execution state.
- `published`: local discovery RPC outcome; null means unknown/shutting down.
- `revision`: latest accepted controller mutation.
- `applied_revision`: latest mutation successfully reconciled with discovery.
- `last_error`: most recent verification/reconciliation/invalidation error.
- `observed`: identity and version from the engine observation (or null).

An HTTP success is **not** an admission-success or router-convergence barrier.
After admit, wait for the intended revision to be applied, `admitted=true`, and
`published=true`; surface errors or newer commands instead of waiting forever.
After withdraw, wait for its applied revision and `published=false`. These are
local discovery acknowledgments, not acknowledgments from all frontend caches.

Pause/health changes withdraw membership without erasing a still-matching grant;
the same engine/version may rejoin after recovery. A changed engine identity,
version, missing observation, closed producer, or changed observation epoch
revokes permission and requires a fresh controller approval. Producers must
advance the epoch on reconnect so a coalesced disconnect cannot preserve an old
grant. Sidecar restart creates a new session and begins unapproved.

For weight updates, stop/resolve rollout submissions and withdraw first, then
perform the native engine pause/update/commit/resume protocol. Admit only after
the controller validates the resulting worker instance and version. New workers
may be healthy and unpaused while awaiting their first grant, matching a
controller's `PendingWeights` state. Engine controls, version labels, and resume
are not substitutes for that grant.

This API does not reject direct or stale-routed requests, implement a distributed
drain, prove tensor/TP-rank consistency, or enforce per-request policy versions.
Controllers still own in-flight request handling and rollout round boundaries.
The separate `--enable-rl` discovery endpoint is unsupported in observed mode.
