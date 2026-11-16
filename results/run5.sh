#!/bin/bash
# Run 5 of plans/REVISION_PLAN.md (tag plan-frozen-run5). Rerunning resumes.
cd "$(dirname "$0")/.."
step() { echo "=== $(date '+%a %-d %b %-I:%M %p') start: $*"; python experiments.py "$@"; echo "=== $(date '+%a %-d %b %-I:%M %p') end: $*"; }
step sr --equations all --arms full_diag,noent_diag --seeds 10 --jobs 10 --tag diag
step sr --equations all --arms best16,worst16 --seeds 30 --jobs 10 --tag oracle
echo "=== all done"
