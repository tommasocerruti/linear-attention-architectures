#!/bin/bash
#
# Submit 8-job LR ablation sequentially, respecting a max of 2 jobs
# (running + pending) in the debug partition at any time.
#
# Runs detached from your SSH session:
#   nohup bash _research/launch/submit-lr-ablation.sh &> _research/results/runs/submit-lr-ablation.log &
#
# Progress: tail -f _research/results/runs/submit-lr-ablation.log

LAUNCH=_research/launch
DATE=$(date +%Y%m%d)
MAX_JOBS=2
POLL_INTERVAL=10  # seconds between squeue checks

COMMON="SEQ_LEN=4096,GBS=128,MBS=16,TRAIN_ITERS=500,EXIT_DURATION_IN_MINS=40,EVAL_INTERVAL=50,EVAL_ITERS=1,WANDB_GROUP=lr-ablation-$DATE"

wait_for_slot() {
    while true; do
        # count jobs in debug partition that are running or pending
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
    sbatch --export=ALL,${COMMON},${extra},EXP_NAME=${exp} $LAUNCH/${script}
}

# AdamW LR=3e-4 (baseline)
submit ablation-softmax-adamw-lr3e4  transformer-pp-350m-adamw-smoke.sbatch    "LR=3e-4,MIN_LR=3e-5"
submit ablation-gdn-adamw-lr3e4      transformer-pp-350m-gdn-smoke.sbatch      "LR=3e-4,MIN_LR=3e-5"
submit ablation-dn-adamw-lr3e4       transformer-pp-350m-deltanet-smoke.sbatch "LR=3e-4,MIN_LR=3e-5"

# AdamW LR=1e-3
submit ablation-softmax-adamw-lr1e3  transformer-pp-350m-adamw-smoke.sbatch    "LR=1e-3,MIN_LR=1e-4"
submit ablation-gdn-adamw-lr1e3      transformer-pp-350m-gdn-smoke.sbatch      "LR=1e-3,MIN_LR=1e-4"
submit ablation-dn-adamw-lr1e3       transformer-pp-350m-deltanet-smoke.sbatch "LR=1e-3,MIN_LR=1e-4"

# Muon MUON_LR=1e-4
submit ablation-softmax-muon-lr1e4   transformer-pp-350m-muon-smoke.sbatch     "MUON_LR=1e-4"

# Muon MUON_LR=5e-4
submit ablation-softmax-muon-lr5e4   transformer-pp-350m-muon-smoke.sbatch     "MUON_LR=5e-4"

echo "[$(date '+%H:%M:%S')] All 8 jobs submitted."
