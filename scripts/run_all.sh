#!/usr/bin/env bash
# Reproduces the full study: data subsets, DAMSM encoders, all models x 3 datasets x 3 seeds, the CUB
# ablations, sampling and evaluation on the held-out test split, and the result tables.
# Runs sequentially on one GPU (select it with CUDA_VISIBLE_DEVICES). Every training run resumes from
# runs/<dataset>/<model>/seed_<k>/ckpt/last.pt if it was interrupted.
#
#   bash scripts/run_all.sh                 # everything
#   DATASETS="cub" SEEDS="0" bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
DATASETS=${DATASETS:-"cub flowers coco"}
SEEDS=${SEEDS:-"0 1 2"}
MODELS=${MODELS:-"gamf dfgan attngan dmgan stylegan_t"}
ABLATIONS=${ABLATIONS:-"gamf_no_clip gamf_no_attn gamf_no_mem"}   # CUB-200-2011 only

$PY scripts/prefetch_weights.py

cfgs() {  # model configs for a model name (ablations are applied on top of gamf.yaml)
  case $1 in gamf_*) echo "configs/models/gamf.yaml configs/models/$1.yaml" ;;
             *)      echo "configs/models/$1.yaml" ;; esac
}

run_model() {  # dataset model seed
  local ds=$1 m=$2 s=$3 run=runs/$1/$2/seed_$3
  [ -f "$run/train_stats.json" ] || \
    $PY -m gamf.train --config configs/base.yaml configs/datasets/$ds.yaml $(cfgs "$m") --set seed=$s
}

for ds in $DATASETS; do
  $PY -m gamf.data.prepare --config configs/base.yaml configs/datasets/$ds.yaml
  $PY -m gamf.train_damsm --config configs/base.yaml configs/datasets/$ds.yaml
  for m in $MODELS; do for s in $SEEDS; do run_model "$ds" "$m" "$s"; done; done
done
if [[ " $DATASETS " == *" cub "* ]]; then
  for m in $ABLATIONS; do for s in $SEEDS; do run_model cub "$m" "$s"; done; done
fi

# held-out test split: only after every model is trained (checkpoints were selected by validation FID)
for run in runs/*/*/seed_*; do
  [ -f "$run/train_stats.json" ] && [ ! -f "$run/metrics.json" ] || continue
  $PY -m gamf.sample --run "$run" --split test
  $PY -m gamf.evaluate --config "$run/config.yaml" --gen-dir "$run/samples/test" --split test \
      --out "$run/metrics.json"
done

$PY -m gamf.analysis.aggregate --runs runs --out results/final_test --models $MODELS
$PY -m gamf.analysis.aggregate --runs runs --out results/final_ablation --kind ablation \
    --models gamf $ABLATIONS
$PY scripts/final_tables.py --out results/final_tables.tex
$PY scripts/fig_val_curves.py --out results/figures/fig_val_curves.svg
