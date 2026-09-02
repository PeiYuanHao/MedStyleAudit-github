#!/bin/bash

cd /root/MedStyleAudit-github || return 1

export MEDSTYLE_DATA_ROOT=/root/autodl-tmp/datasets
export MEDSTYLE_OUTPUT_ROOT=/root/autodl-tmp/medstyleaudit-experiments
export PYTHONUNBUFFERED=1

ROOT=/root/autodl-tmp/medstyleaudit-experiments

BALANCED_TRIPLETS=$ROOT/subsets/hospital2_triplets.parquet
RANDOM_TRIPLETS=$ROOT/audits/hospital2/random_paired/triplets.parquet

LOGROOT=/root/autodl-tmp/h2_remaining_logs

mkdir -p "$LOGROOT"


run_primary () {
    seed=$1
    setting=$2
    triplets=$3
    extra=$4

    SEED4=$(printf "%04d" "$seed")

    OUT=$ROOT/audits/hospital2/$setting/resnet50/seed_$SEED4

    mkdir -p "$OUT"

    echo
    echo "============================================================"
    echo "START seed=$seed setting=$setting"
    echo "TIME: $(date)"
    echo "TRIPLETS: $triplets"
    echo "============================================================"

    python -u scripts/06_run_primary_audit.py \
      --config configs/audit/primary.yaml \
      --model-config configs/models/resnet50_hf.yaml \
      --checkpoint "$ROOT/checkpoints/resnet50/seed_$SEED4/best.ckpt" \
      --id-logits "$ROOT/predictions/resnet50/seed_$SEED4/id_val.parquet" \
      --triplets "$triplets" \
      --split test \
      --seed "$seed" \
      --device cuda:0 \
      --output-dir "$OUT" \
      --allow-final-test \
      --overwrite \
      $extra

    RC=$?

    if [ "$RC" -ne 0 ]; then
        echo "FAILED seed=$seed setting=$setting rc=$RC"
        return "$RC"
    fi

    echo "PASS seed=$seed setting=$setting"
}


run_robustness () {
    seed=$1
    setting=$2
    config=$3

    SEED4=$(printf "%04d" "$seed")

    OUT=$ROOT/audits/hospital2/$setting/resnet50/seed_$SEED4

    mkdir -p "$OUT"

    echo
    echo "============================================================"
    echo "START seed=$seed setting=$setting"
    echo "TIME: $(date)"
    echo "============================================================"

    python -u scripts/08_run_robustness.py \
      --config "$config" \
      --model-config configs/models/resnet50_hf.yaml \
      --checkpoint "$ROOT/checkpoints/resnet50/seed_$SEED4/best.ckpt" \
      --id-logits "$ROOT/predictions/resnet50/seed_$SEED4/id_val.parquet" \
      --triplets "$BALANCED_TRIPLETS" \
      --split test \
      --seed "$seed" \
      --device cuda:0 \
      --output-dir "$OUT" \
      --allow-final-test \
      --overwrite

    RC=$?

    if [ "$RC" -ne 0 ]; then
        echo "FAILED seed=$seed setting=$setting rc=$RC"
        return "$RC"
    fi

    echo "PASS seed=$seed setting=$setting"
}


run_seed () {
    seed=$1

    echo "############################################################"
    echo "SEED $seed — PRIMARY ALREADY COMPLETE"
    echo "REMAINING 4 AUDITS START"
    echo "############################################################"

    run_primary \
        "$seed" \
        random_paired \
        "$RANDOM_TRIPLETS" \
        "" || return 1

    echo "SEED $seed REMAINING PROGRESS 1/4"

    run_primary \
        "$seed" \
        roi_only \
        "$BALANCED_TRIPLETS" \
        "--roi-only-control" || return 1

    echo "SEED $seed REMAINING PROGRESS 2/4"

    run_robustness \
        "$seed" \
        buffer_r8 \
        configs/audit/source_buffer.yaml || return 1

    echo "SEED $seed REMAINING PROGRESS 3/4"

    run_robustness \
        "$seed" \
        hard_boundary \
        configs/audit/seam.yaml || return 1

    echo "SEED $seed REMAINING PROGRESS 4/4"

    echo
    echo "############################################################"
    echo "SEED $seed REMAINING 4 AUDITS: PASS"
    echo "############################################################"
}


# ------------------------------------------------------------
# Preflight
# ------------------------------------------------------------

echo "============================================================"
echo "H2 REMAINING 12 AUDITS PREFLIGHT"
echo "============================================================"

echo "Balanced:"
ls -lh "$BALANCED_TRIPLETS"

echo
echo "Random Paired:"
ls -lh "$RANDOM_TRIPLETS"

python - <<'PY'
import pandas as pd

balanced = pd.read_parquet(
    "/root/autodl-tmp/medstyleaudit-experiments/subsets/hospital2_triplets.parquet"
)

random = pd.read_parquet(
    "/root/autodl-tmp/medstyleaudit-experiments/audits/hospital2/random_paired/triplets.parquet"
)

assert len(balanced) == 90000
assert len(random) == 90000

assert balanced["source_id"].nunique() == 10000
assert random["source_id"].nunique() == 10000

print("BALANCED TRIPLETS: PASS")
print("RANDOM TRIPLETS:   PASS")
PY

if [ $? -ne 0 ]; then
    echo "PREFLIGHT FAILED"
    return 1
fi


echo
echo "START 3 PARALLEL SEEDS: $(date)"


(run_seed 11) \
> "$LOGROOT/seed_11.log" 2>&1 &
P11=$!

(run_seed 42) \
> "$LOGROOT/seed_42.log" 2>&1 &
P42=$!

(run_seed 101) \
> "$LOGROOT/seed_101.log" 2>&1 &
P101=$!


echo "$P11" > "$LOGROOT/seed_11.pid"
echo "$P42" > "$LOGROOT/seed_42.pid"
echo "$P101" > "$LOGROOT/seed_101.pid"


wait "$P11"
R11=$?

wait "$P42"
R42=$?

wait "$P101"
R101=$?


echo
echo "============================================================"
echo "FINAL RETURN CODES"
echo "seed11  = $R11"
echo "seed42  = $R42"
echo "seed101 = $R101"
echo "============================================================"

if [ "$R11" -eq 0 ] && \
   [ "$R42" -eq 0 ] && \
   [ "$R101" -eq 0 ]; then

    echo "ALL REMAINING 12 H2 AUDITS: PASS"

else

    echo "H2 REMAINING AUDITS: FAILED"

fi

echo "FINISHED: $(date)"
