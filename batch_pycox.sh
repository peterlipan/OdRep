#!/bin/bash

configs=("metabric" "support" "gbsg" "flchain" "nwtco")
methods=("ordsurv" "deepsurv" "deephit" "nll" "lassocox" "coxtime")
steps=(1.0 30.0 180.0 365.0 1825.0)

for config in "${configs[@]}"; do
  for method in "${methods[@]}"; do
    for step in "${steps[@]}"; do
      python3 main.py --debug --config "$config" --method "$method" --step "$step"
    done
  done
done
