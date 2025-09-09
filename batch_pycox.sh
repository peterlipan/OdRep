#!/bin/bash

configs=("metabric" "support" "gbsg" "flchain" "nwtco")
methods=("ordsurv" "deepsurv" "deephit" "nll" "lassocox" "coxtime")
bins=(10 100 1000 10000)

for config in "${configs[@]}"; do
  for method in "${methods[@]}"; do
    for bin in "${bins[@]}"; do
      python3 main.py --debug --config "$config" --method "$method" --n_bins "$bin"
    done
  done
done
