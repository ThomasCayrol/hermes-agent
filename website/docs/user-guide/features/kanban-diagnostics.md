---
sidebar_position: 13
title: "Kanban Diagnostics & Operator Attention"
description: "Operator Diagnostics Clarity — technical diagnostics separated from operator-attention severity, stranded-task classification, attention banner policy, recovery & concurrency states"
---

# Kanban Diagnostics & Operator Attention

> This page is the canonical audit matrix for the **Operator Diagnostics Clarity**
> increment on the Hermes fork. The engine lives in
> `hermes_cli/kanban_diagnostics.py`; CLI surfaces run it via
> `hermes kanban diagnostics` and the dashboard plugin API. Operator-facing copy
> is French with canonical English technical terms.

Every kanban diagnostic now carries **two orthogonal axes**:

- **Technical severity** — `warning | error | critical` (`SEVERITY_ORDER`),
  unchanged: triage order and legacy colouring.
- **Operator attention** — `NONE | INFO | WARNING | ACTION_REQUIRED | CRITICAL`
  (`ATTENTION_ORDER`), additive: what the operator should actually do.
  A diagnostic may exist with `NONE`/`INFO` attention (a healthy,
  legitimately-queued task); emission never implies an operator action.

Each `Diagnostic` exposes in `to_dict()`: `attention`, `owner_action`
(`NONE`/`REQUIRED` — REQUIRED exactly when attention is
ACTION_REQUIRED/CRITICAL), `system_action` (what Hermes does automatically),
`attention_banner` (engine-decided), `auto_recovery_state`
(`none|in_progress|succeeded|failed`), `classification`, and the French
operator message fields `operator_status` / `operator_cause` /
`operator_impact` (+ `operator_risk` for concurrency kinds, where RISK
replaces IMPACT). Section labels rendered by surfaces stay English:
`STATUS / CAUSE / IMPACT / OWNER ACTION / SYSTEM ACTION`.

## Attention-banner policy (decided once, by the engine)

The banner (“N tasks need attention”) counts **tasks** that carry ≥ 1
diagnostic with `attention_banner=true`, decided by the engine:

```text
attention_banner = true  ⇔  auto_recovery_state == FAILED
                        ∨  attention == CRITICAL
                        ∨  owner_action == REQUIRED
                        ∨  (attention == WARNING AND abnormal AND not auto-recoverable)
```

- **Never in the banner**: healthy queue/capacity (`LEGITIMATELY_QUEUED`,
  expected saturation = INFO/NONE), non-actionable diagnostics
  (`prose_phantom_refs`, `recovery_in_progress`).
- Surfaces filter/render the engine field; they never re-derive the policy.
- Healthy copy never sounds alarmist: “En attente légitime de capacité worker”,
  “Dispatcher sain — capacité worker compatible actuellement occupée”;
  never “dispatcher stuck” on a healthy dispatcher, never “no worker” on a
  legitimate queue.

## stranded_in_ready — 5-outcome classifier

A `ready` task whose `ready_since` age exceeds `stranded_threshold_seconds`
(30 min default), with an empty `claim_lock` and a non-empty assignee, is
**classified** instead of being flagged as a bare “no worker”:

| Outcome | Evidence pattern | Attention | Owner action | Banner |
|---|---|---|---|---|
| `LEGITIMATELY_QUEUED` | profile cap full, healthy running sibling (fresh heartbeat), dispatcher healthy, queue advancing / known position+priority | INFO | NONE | no |
| `READY_TOO_LONG_UNEXPLAINED` | capacity available, no claim, assignee valid, dispatcher healthy, no legitimate explanation (fail-safe default) | WARNING → ACTION_REQUIRED → CRITICAL by age | NONE (reco) / REQUIRED at escalation | yes |
| `NO_COMPATIBLE_WORKER` | assignee invalid / lane not spawnable (proven roster mismatch) | ACTION_REQUIRED | REQUIRED | yes |
| `DISPATCHER_UNHEALTHY` | dispatcher evidence only (`dispatcher.healthy=false`); never inferred from task age alone | ACTION_REQUIRED / CRITICAL (board impact) | REQUIRED | yes |
| `PROFILE_CAPACITY_SATURATED` | cap reached by running workers of the profile | expected INFO; abnormal (stale siblings, no dispatcher proof) WARNING | NONE | only abnormal |

Decision inputs (read-only, all optional — absence of evidence is never
treated as health): `dispatcher`, `profiles`, `lanes`, `profile_cap`,
`board_cap`, `running_by_assignee`, `queue_by_assignee`,
`queue_progressed_by_assignee`, `expected_slot_freed_by_assignee`,
`attempts_by_task`, `active_scope_tasks`, `out_of_band_writers`,
`superseded_scope_tasks`. The runtime collector
`kanban_diagnostics.build_board_context(conn, config=…)` gathers them from
stable columns at the base (no schema change); tests / Mission Control can
pass a context by hand. Fail-safe: when no positive legitimacy evidence
exists, the outcome defaults to conservative `READY_TOO_LONG_UNEXPLAINED`
warning — a missing signal is never dropped. Precedence: repeated failures /
crashes already explaining the task win (stranded cedes, no double flag).

## Audit matrix — every diagnostic kind

Technical severity reused; operator attention additive. Trigger | technical
meaning | attention | auto-recovery | OWNER ACTION | banner.

| Kind | Trigger | Attention | Auto-recovery | Owner | Banner |
|---|---|---|---|---|---|
| `hallucinated_cards` | `completion_blocked_hallucination` active | ACTION_REQUIRED | no (kernel doesn’t guess the route) | REQUIRED | yes |
| `triage_aux_unavailable` | triage task + unusable aux slot, no main model | ACTION_REQUIRED | no | REQUIRED | yes |
| `prose_phantom_refs` | completion summary cites unknown ids | INFO | yes (auto-clear on next clean completion) | NONE | no |
| `repeated_failures` | `consecutive_failures` ≥ threshold | ACTION_REQUIRED / CRITICAL (≥ 2×) | partial (dispatcher retries, then breaker) | REQUIRED | yes |
| `repeated_crashes` | trailing crashed runs ≥ threshold | ACTION_REQUIRED / CRITICAL (≥ 2×) | no (beyond retry policy) | REQUIRED | yes |
| `review_dependency_deadlock` | blocked `review-required:` + waiting `todo` children | ACTION_REQUIRED | no (graph + sticky block intact by design) | REQUIRED | yes |
| `stuck_in_blocked` | blocked > `blocked_stale_hours` without exchange — or **immediate** when the block carries an approval decision (`decision_class=APPROVAL_REQUIRED` / reason mentions approval) | ACTION_REQUIRED | no (blocked = human input) | REQUIRED | yes |
| `block_unblock_cycling` | ≥ N block→unblock cycles in window | ACTION_REQUIRED | no | REQUIRED | yes |
| `stranded_in_ready` | see classifier above | per outcome | per outcome | per outcome | per outcome |
| `recovery_in_progress` | auto-recovery lifecycle marker `recovery_started`/`recovering` | INFO | in progress | NONE | no |
| `recovery_failed` | auto-recovery failed marker | ACTION_REQUIRED / CRITICAL (board impact) | failed → escalate | REQUIRED | yes |
| `duplicate_implementation` | 2+ active tasks same repo+branch+scope | ACTION_REQUIRED (owner decision) / WARNING when supersession proven | auto ONLY on proven supersession | REQUIRED / NONE | yes / no |
| `concurrent_writer_risk` | out-of-band (non-kanban) writer on same checkout | ACTION_REQUIRED while risk active | no | REQUIRED | yes |

**Recovery states are honest.** `RECOVERING` is only shown when an
auto-recovery lifecycle marker is evidenced; a worker actually running is
`RUNNING` with `OWNER ACTION: NONE` (never a fabricated `STARTING` /
“progressing automatically”). The engine renders recovery states from events
(`recovery_started` / `recovering` / `recovery_succeeded` / `recovery_failed`)
and never executes recovery itself — bounded, non-overlapping with the stall
watchdog lane. History is always preserved.

## Action contract

Primary action labels describe the resulting action, in French; the raw CLI
command is only a secondary affordance (discreet copy icon). New engine kinds:
`run_diagnostics` (**Diagnostiquer le dispatcher** — executes a real read-only
diagnostics run, replacing the old “Check dispatcher status” copy hint),
`view_worker` (**Voir le worker actif**), `view_queue` (**Voir la file
{profile}**). Existing real actions: `reclaim` (Récupérer la tâche),
`reassign` (Réassigner), `unblock` (Débloquer), `comment` (Ajouter un
commentaire), `open_docs`, `cli_hint` (secondary only).

## Case map (A–H)

Tests live in `tests/hermes_cli/test_kanban_diagnostics.py`.

| Case | Normative result |
|---|---|
| A | ready 45m, cap 1/1 + healthy running sibling → `LEGITIMATELY_QUEUED`, INFO, no banner |
| B | ready 45m, capacity free, no claim → `READY_TOO_LONG_UNEXPLAINED`, WARNING, banner |
| C | dispatcher evidence dead → `DISPATCHER_UNHEALTHY`, ACTION_REQUIRED/CRITICAL, banner; never from age alone |
| D | auto-recovery markers → `recovery_in_progress` INFO (no banner) / `recovery_failed` ACTION_REQUIRED (banner); history preserved |
| E | approval block → immediate ACTION_REQUIRED + REQUIRED ACTION + WHY, banner |
| F | long-running worker, fresh heartbeat → no warning at all |
| G | same repo+branch+scope active duplicates / out-of-band writer → `duplicate_implementation` / `concurrent_writer_risk` with STATUS/CAUSE/RISK/OWNER ACTION |
| H | real action buttons execute the real read-only run or board action; CLI copy secondary |
