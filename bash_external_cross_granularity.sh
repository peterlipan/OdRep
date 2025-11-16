#!/bin/bash
set -euo pipefail

# ====== Fixed setup ======
CONFIG="external"
METHODS=("deephit")     # keep all three
SEED=42
VISIBLE_GPUS="0"

# Fixed MLP encoder hyperparams (no grid)
N_LAYERS=2
D_HID=2048
ACTIVATION="leaky_relu"

# ====== Time steps (days) ======
# Julian year: 365.25; month≈30.4375; quarters/halves for convenience
FINE_TO_COARSE_STEPS=("7.0" "30.4375" "91.3125" "182.625" "365.25")
COARSE_TO_FINE_STEPS=("365.25" "182.625" "91.3125" "30.4375" "7.0" "1.0")

run_sweep () {
  local TRAIN_STEP="$1"; shift
  local -n TEST_STEPS_REF="$1"  # name-ref to array

  for method in "${METHODS[@]}"; do
    for eval_step in "${TEST_STEPS_REF[@]}"; do
      echo "[RUN] cfg=${CONFIG} method=${method} train_step=${TRAIN_STEP} eval_step=${eval_step} "\
           "layers=${N_LAYERS} d_hid=${D_HID} act=${ACTIVATION} seed=${SEED}"
      python3 main.py \
        --debug \
        --config "${CONFIG}" \
        --method "${method}" \
        --train_step "${TRAIN_STEP}" \
        --eval_step "${eval_step}" \
        --n_layers "${N_LAYERS}" \
        --d_hid "${D_HID}" \
        --activation "${ACTIVATION}" \
        --seed "${SEED}" \
        --visible_gpus "${VISIBLE_GPUS}"
    done
  done
}

# ====== Scenario A: fine → coarse (train at 1.0d; eval from 1.0 → 365.25) ======
# run_sweep "7.0" FINE_TO_COARSE_STEPS

# ====== Scenario B: coarse → fine (train at 365.25d; eval from 365.25 → 1.0) ======
run_sweep "365.25" COARSE_TO_FINE_STEPS
