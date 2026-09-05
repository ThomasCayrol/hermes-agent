"""Live-path validation for kanban_external_ci (run manually, not in CI).

Uses the real authenticated gh CLI against ThomasCayrol/hermes-agent to prove
the collector parses REST job/run evidence plus authoritative GraphQL
CheckRun.isRequired data for the current PR head. PR #6's queued jobs were
optional; this probe proves they cannot authorize a retry.

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
    print(f"runs={len(snap.runs)} jobs={len(snap.jobs)} required_jobs={snap.required_job_ids}")
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
    # PR #6 is closed and its Action jobs are not required checks. The
    # authoritative rollup therefore makes the wait moot and non-retriable.
    assert snap.required_check_evidence is True
    assert snap.required_job_ids == []
    assert snap.required is False
    assert result["ci_state"] in (kc.COMPLETED, kc.CI_WAITING), result["ci_state"]
    print("optional/closed PR: silent and no retry authorization OK")
    print("live-path validation OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
