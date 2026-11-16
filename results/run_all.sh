#!/bin/bash
# The plan's commands (plans/REVISION_PLAN.md at tag plan-frozen), in order. Rerunning resumes.
cd "$(dirname "$0")/.."
A=random,full,policy,heuristic,nocritic,full_noconst,random_noconst,full_m01,random_m01,ssc,sizefair,random_x3
step() { echo "=== $(date '+%a %-d %b %-I:%M %p') start: $*"; python experiments.py "$@"; echo "=== $(date '+%a %-d %b %-I:%M %p') end (exit $?): $*"; }
step sr --equations all --arms $A --seeds 30 --jobs 10 --tag main
step sr --equations all --arms random,full,ssc --seeds 30 --jobs 10 --noise 0.01 --tag noise01
step sr --equations all --arms random,full,ssc --seeds 30 --jobs 10 --noise 0.1 --tag noise10
step sr --equations nguyen --arms random,full,ssc --seeds 30 --jobs 10 --protocol orig --tag nguyen_orig
step circuits --arms swap,learned,uniform --targets 30 --seeds 10 --jobs 10
echo "=== all done"
