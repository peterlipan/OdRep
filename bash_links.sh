#!/bin/bash
set -euo pipefail

# ====== Datasets / Methods ======
configs=("links")
methods=("clisurv-po" "clisurv-ph" "clisurv-gen")
links=("ph" "po" "gen")


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
    for link in "${links[@]}"; do
      for layers in "${mlp_layers[@]}"; do
        for hidden in "${mlp_hidden_dims[@]}"; do
          for act in "${activations[@]}"; do
            for seed in "${seeds[@]}"; do

              echo "[RUN] cfg=${config} method=${method} link=${link} layers=${layers} hidden=${hidden} act=${act} seed=${seed}"

              # If python fails, print a message but continue looping
              python3 main.py \
                --debug \
                --config "${config}" \
                --method "${method}" \
                --link "${link}" \
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
