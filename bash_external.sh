#!/bin/bash

configs=("external")
methods=("decouple" "ordsurv" "deepsurv" "deephit" "nll")
steps=(1.0 7.0 30.0 180.0 365.0 1825.0)

for config in "${configs[@]}"; do
  for method in "${methods[@]}"; do
    for step in "${steps[@]}"; do
      python3 main.py --debug --config "$config" --method "$method" --step "$step" --visible_gpus "0"
    done
  done
done
