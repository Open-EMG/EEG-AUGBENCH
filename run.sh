#!/bin/bash
set -e  # 遇错即停
trap 'echo "❌ Error at line $LINENO"; exit 1' ERR

AUG_METHODS=("no" "jitter" "scaling" "permutation" "magwarp" "timewarp" "windowslice" \
             "windowwarp" "rgw" "rgws" "scaling_multi" "windowwarp_multi")

CLS_METHODS=("knn" "LRC" "RFC" "DTC" "AdaBoost")

for aug in "${AUG_METHODS[@]}"; do
  for cls in "${CLS_METHODS[@]}"; do
    echo "▶ Running with aug_method=${aug}, cls_method=${cls}"
    python main.py \
      --aug_ratio 10 \
      --aug_method "${aug}" \
      --cls_method "${cls}" \
      --input_dir ./data/chb-mit/ \
      --output_augmented_dir ./output/chb-mit/augmented/ \
      --output_feature_dir ./output/chb-mit/features/ \
      --output_result_dir ./output/chb-mit/results/
  done
done
