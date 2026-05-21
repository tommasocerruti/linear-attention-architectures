#!/bin/bash
#
# Submit GDN seq-length scaling jobs (1K-64K) for the attention-dominated ~7M model.
# Runs two variants to demonstrate GDN's O(N) vs softmax O(N^2) scaling:
#   - hybrid:  2:1 GDN:SDPA layer pattern (linear-attention-freq=3, default)
#   - pure:    all layers linear attention (PURE_GDN=1, no SDPA)
#
# Respects a max of 2 jobs (running + pending) in the debug partition at any time.
#
# Runs detached from your SSH session:
#   nohup bash _research/launch/submit-attn-dominated-gdn-seq-scaling.sh &> _research/results/runs/submit-attn-dominated-gdn-seq-scaling.log &
#
# Progress: tail -f _research/results/runs/submit-attn-dominated-gdn-seq-scaling.log

LAUNCH=_research/launch
SCRIPT=transformer-pp-attn-dominated-gdn-smoke.sbatch
SBATCH_TIME=00:45:00
MAX_JOBS=2
POLL_INTERVAL=10

echo "[$(date '+%H:%M:%S')] Submitting GDN seq-length scaling jobs (hybrid + pure, 1K-64K)."

wait_for_slot() {
    while true; do
        local count
        count=$(squeue --me --partition=debug --states=RUNNING,PENDING --noheader | wc -l)
        if [ "$count" -lt "$MAX_JOBS" ]; then
            return
        fi
        echo "[$(date '+%H:%M:%S')] $count/$MAX_JOBS slots used, waiting ${POLL_INTERVAL}s..."
        sleep $POLL_INTERVAL
    done
}

submit() {
    local exp=$1
    local script=$2
    local extra=$3
    wait_for_slot
    echo "[$(date '+%H:%M:%S')] Submitting $exp ..."
    sbatch --time=$SBATCH_TIME --export=ALL,${extra},EXP_NAME=${exp} $LAUNCH/${script}
}

# GDN hybrid (2:1 GDN:SDPA, linear-attention-freq=3)
# submit attn-dominated-gdn-smoke-1k    $SCRIPT "SEQ_LEN=1024,MBS=128,GBS=4096"   # already ran
# submit attn-dominated-gdn-smoke-2k    $SCRIPT "SEQ_LEN=2048,MBS=64,GBS=2048"    # already ran
# submit attn-dominated-gdn-smoke-4k    $SCRIPT "SEQ_LEN=4096,MBS=32,GBS=1024"    # already ran
submit attn-dominated-gdn-smoke-8k    $SCRIPT "SEQ_LEN=8192,MBS=16,GBS=512"
submit attn-dominated-gdn-smoke-16k   $SCRIPT "SEQ_LEN=16384,MBS=8,GBS=256"
submit attn-dominated-gdn-smoke-32k   $SCRIPT "SEQ_LEN=32768,MBS=4,GBS=128"
submit attn-dominated-gdn-smoke-64k   $SCRIPT "SEQ_LEN=65536,MBS=1,GBS=64"

# GDN pure (all layers linear attention, no SDPA)
submit attn-dominated-gdn-pure-smoke-1k   $SCRIPT "PURE_GDN=1,SEQ_LEN=1024,MBS=128,GBS=4096"
submit attn-dominated-gdn-pure-smoke-2k   $SCRIPT "PURE_GDN=1,SEQ_LEN=2048,MBS=64,GBS=2048"
submit attn-dominated-gdn-pure-smoke-4k   $SCRIPT "PURE_GDN=1,SEQ_LEN=4096,MBS=32,GBS=1024"
submit attn-dominated-gdn-pure-smoke-8k   $SCRIPT "PURE_GDN=1,SEQ_LEN=8192,MBS=16,GBS=512"
submit attn-dominated-gdn-pure-smoke-16k  $SCRIPT "PURE_GDN=1,SEQ_LEN=16384,MBS=8,GBS=256"
submit attn-dominated-gdn-pure-smoke-32k  $SCRIPT "PURE_GDN=1,SEQ_LEN=32768,MBS=4,GBS=128"
submit attn-dominated-gdn-pure-smoke-64k  $SCRIPT "PURE_GDN=1,SEQ_LEN=65536,MBS=1,GBS=64"

echo "[$(date '+%H:%M:%S')] All remaining jobs submitted."
