"""Live-path validation for kanban_external_ci (run manually, not in CI).

Uses the real authenticated gh CLI against ThomasCayrol/hermes-agent to prove
the collector parses actual GitHub REST payloads and the classifier behaves on
real evidence (PR #6: closed now, but its historical queued/steps=0 jobs are
still readable — the exact incident shape from the Product contract).

Read-only: only GET requests are issued; no rerun POST is attempted.
"""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, ".")
from hermes_cli import kanban_external_ci as kc  # noqa: E402


def main() -> int:
    print(f"gh available: {kc.gh_available()}")
    if not kc.gh_available():
        print("SKIP: gh CLI not on PATH")
        return 0

    # 1) Collector on the real PR #6 (closed since the Product snapshot).
    snap = kc.collect_external_ci_snapshot(
        "ThomasCayrol/hermes-agent", 6,
        head_sha="2207ea035fd23dec1d3fd1bda0946e68a2654359",
    )
    print(f"captured_at={snap.captured_at} repo={snap.repo} pr={snap.pr_number}")
    print(f"head_sha={snap.head_sha} required={snap.required} superseded={snap.superseded}")
    print(f"runs={len(snap.runs)} jobs={len(snap.jobs)}")
    print(f"evidence_complete={snap.evidence_complete()}")
    queued = [
        (str((j or {}).get("id")), (j or {}).get("name"), (j or {}).get("status"))
        for j in snap.jobs
        if kc.job_is_queued(j)
    ]
    print(f"queued_jobs={json.dumps(queued)}")
    started = [
        (str((j or {}).get("id")), (j or {}).get("started_at"))
        for j in snap.jobs
        if kc.job_is_queued(j)
    ]
    print(f"queued_jobs_started_at={json.dumps(started)}")
    # started_at non-null must NOT be execution proof.
    for j in snap.jobs:
        if kc.job_is_queued(j):
            assert not kc.job_execution_started(j), "queued job read as running"
            print("  queued job execution_started=False (started_at ignored) OK")

    result = kc.classify_external_ci_wait(snap, now=int(time.time()))
    print(f"classification={json.dumps(result, ensure_ascii=False)}")
    # PR #6 is now CLOSED -> required=False -> wait moot -> no stall alert.
    assert snap.required is False
    assert result["ci_state"] in (kc.COMPLETED, kc.CI_WAITING), result["ci_state"]
    print("classifier on closed PR: silent (no stall alert) OK")

    # 2) Same payload but forced required=True to exercise the real queued
    # evidence through the stall classifier (the historical incident state).
    snap.required = True
    result2 = kc.classify_external_ci_wait(snap, now=int(time.time()))
    print(f"forced-required classification={json.dumps(result2, ensure_ascii=False)}")
    print("live-path validation OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
