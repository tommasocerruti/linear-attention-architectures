#!/bin/bash
#
# Submit 19-job LR ablation v2 (2000 iters) sequentially, respecting a max of 2 jobs
# (running + pending) in the debug partition at any time.
#
# Submit the retained LR ablation grid with missing LR values filled in:
#   - Reruns all 4 AdamW x softmax (1e-2,1e-3,1e-4,1e-5): original runs were
#     broken (LR hardcoded to 3e-4 in the sbatch script, now fixed).
#   - AdamW in-between LRs (3e-3,3e-4,3e-5) for all 3 architectures.
#   - Muon  in-between LRs (1e-3,3e-5)       for all 3 architectures.
#
# Runs detached from your SSH session:
#   nohup bash _research/launch/submit-lr-ablation-2000-v2.sh &> _research/results/runs/submit-lr-ablation-2000-v2.log &
#
# Progress: tail -f _research/results/runs/submit-lr-ablation-2000-v2.log

LAUNCH=_research/launch
DATE=$(date +%Y%m%d)
MAX_JOBS=2
POLL_INTERVAL=10

# 2000 iters ~4x 500-iter runs (max observed: 9:40). Budget 50 min with margin.
SBATCH_TIME=00:50:00
COMMON="SEQ_LEN=4096,GBS=128,MBS=16,TRAIN_ITERS=2000,EXIT_DURATION_IN_MINS=45,EVAL_INTERVAL=100,EVAL_ITERS=1,WANDB_GROUP=lr-ablation-2000-v2-$DATE"

echo "[$(date '+%H:%M:%S')] LR ablation v2."
echo "[$(date '+%H:%M:%S')] Adds missing LR values (adamw in-between: 3e-3/3e-4/3e-5; muon in-between: 1e-3/3e-5)"
echo "[$(date '+%H:%M:%S')] and reruns softmax x adamw with the LR bug fixed. 19 jobs total."

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
    sbatch --time=$SBATCH_TIME --export=ALL,${COMMON},${extra},EXP_NAME=${exp} $LAUNCH/${script}
}

# AdamW x softmax — rerun all 4 (original runs used wrong LR due to bug in sbatch script)
submit ablation2kv2-softmax-adamw-lr1e2  transformer-pp-350m-adamw-smoke.sbatch        "LR=1e-2,MIN_LR=1e-3"
submit ablation2kv2-softmax-adamw-lr1e3  transformer-pp-350m-adamw-smoke.sbatch        "LR=1e-3,MIN_LR=1e-4"
submit ablation2kv2-softmax-adamw-lr1e4  transformer-pp-350m-adamw-smoke.sbatch        "LR=1e-4,MIN_LR=1e-5"
submit ablation2kv2-softmax-adamw-lr1e5  transformer-pp-350m-adamw-smoke.sbatch        "LR=1e-5,MIN_LR=1e-6"

# AdamW x softmax — in-between LRs
submit ablation2kv2-softmax-adamw-lr3e3  transformer-pp-350m-adamw-smoke.sbatch        "LR=3e-3,MIN_LR=3e-4"
submit ablation2kv2-softmax-adamw-lr3e4  transformer-pp-350m-adamw-smoke.sbatch        "LR=3e-4,MIN_LR=3e-5"
submit ablation2kv2-softmax-adamw-lr3e5  transformer-pp-350m-adamw-smoke.sbatch        "LR=3e-5,MIN_LR=3e-6"

# AdamW x GDN — in-between LRs
submit ablation2kv2-gdn-adamw-lr3e3      transformer-pp-350m-gdn-smoke.sbatch          "LR=3e-3,MIN_LR=3e-4"
submit ablation2kv2-gdn-adamw-lr3e4      transformer-pp-350m-gdn-smoke.sbatch          "LR=3e-4,MIN_LR=3e-5"
submit ablation2kv2-gdn-adamw-lr3e5      transformer-pp-350m-gdn-smoke.sbatch          "LR=3e-5,MIN_LR=3e-6"

# AdamW x DeltaNet — in-between LRs
submit ablation2kv2-dn-adamw-lr3e3       transformer-pp-350m-deltanet-smoke.sbatch     "LR=3e-3,MIN_LR=3e-4"
submit ablation2kv2-dn-adamw-lr3e4       transformer-pp-350m-deltanet-smoke.sbatch     "LR=3e-4,MIN_LR=3e-5"
submit ablation2kv2-dn-adamw-lr3e5       transformer-pp-350m-deltanet-smoke.sbatch     "LR=3e-5,MIN_LR=3e-6"

# Muon x softmax — in-between LRs
submit ablation2kv2-softmax-muon-lr1e3   transformer-pp-350m-muon-smoke.sbatch         "MUON_LR=1e-3"
submit ablation2kv2-softmax-muon-lr3e5   transformer-pp-350m-muon-smoke.sbatch         "MUON_LR=3e-5"

# Muon x GDN — in-between LRs
submit ablation2kv2-gdn-muon-lr1e3       transformer-pp-350m-gdn-muon-smoke.sbatch     "MUON_LR=1e-3"
submit ablation2kv2-gdn-muon-lr3e5       transformer-pp-350m-gdn-muon-smoke.sbatch     "MUON_LR=3e-5"

# Muon x DeltaNet — in-between LRs
submit ablation2kv2-dn-muon-lr1e3        transformer-pp-350m-deltanet-muon-smoke.sbatch "MUON_LR=1e-3"
submit ablation2kv2-dn-muon-lr3e5        transformer-pp-350m-deltanet-muon-smoke.sbatch "MUON_LR=3e-5"

echo "[$(date '+%H:%M:%S')] All 19 jobs submitted."
