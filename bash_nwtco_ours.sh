#!/bin/bash
set -euo pipefail

# ====== Datasets / Methods ======
configs=("nwtco")
methods=("clisurv-po" "clisurv-ph" "clisurv-gen")

# ====== Professional time steps (in days) ======
# Using the Julian year: 365.25 days -> 1 month = 365.25/12 = 30.4375 days
# Adds 3-month and 6-month bins; adjusts multi-year bins accordingly.
steps=(1 7 30.4375 91.3125 182.625 365.25 1095.75 1826.25)
# Note: 109.575e1 == 1095.75 (3 years). Written this way to avoid locale float parsing quirks; use 1095.75 if you prefer.

# ====== MLP encoder hyperparams ======
mlp_layers=(1 2 3 4)
mlp_hidden_dims=(256 512 1024 2048)
activations=("relu" "gelu" "tanh" "elu" "leaky_relu")

# Optional: set a seed list if you want repeats per config for stability
seeds=(42)

# Optional: choose GPU visibility once here
VISIBLE_GPUS="0"

#!/bin/bash
set -euo pipefail

# ... arrays ...

for config in "${configs[@]}"; do
  for method in "${methods[@]}"; do
    for step in "${steps[@]}"; do
      for layers in "${mlp_layers[@]}"; do
        for hidden in "${mlp_hidden_dims[@]}"; do
          for act in "${activations[@]}"; do
            for seed in "${seeds[@]}"; do

              echo "[RUN] cfg=${config} method=${method} step=${step} layers=${layers} hidden=${hidden} act=${act} seed=${seed}"

              # If python fails, print a message but continue looping
              python3 main.py \
                --debug \
                --config "${config}" \
                --method "${method}" \
                --step "${step}" \
                --n_layers "${layers}" \
                --d_hid "${hidden}" \
                --activation "${act}" \
                --seed "${seed}" \
                --visible_gpus "${VISIBLE_GPUS}" \
              || echo "❌ Failed run (but continuing)."
            done
          done
        done
      done
    done
  done
done
