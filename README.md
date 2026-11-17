# Domain Adaptation Experiments

Public implementation of the synthetic, CIFAR-10-C, and Causal Chamber experiments in Chapters 3 and 4 of the thesis. ICU data, ICU foundation-model embeddings, trained models, and ICU-specific estimators are intentionally excluded.

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy scikit-learn pandas matplotlib torch Pillow
```

Run the commands below from the repository root. Results are written to caller-specified directories and are excluded from Git.

## Anti-Causal Synthetic Experiment

`synthetic_anti_causal/` implements Section 4.1: 20 covariates, 10 source environments with 1,000 labelled observations each, 100 target environments, five stable and 15 unstable coordinates, six target budgets `100, 200, 400, 800, 1200, 1500`, a 400-observation target test set, five structural draws, and a 2,000-replicate hierarchical paired bootstrap.

```bash
python -m synthetic_anti_causal.run_penalty_effect_sweep \
  --output-dir results/synthetic_anti_causal
```

Run the unlabeled-target-budget sensitivity analysis with:

```bash
python -m synthetic_anti_causal.run_unlabeled_budget_sweep \
  --output-dir results/synthetic_anti_causal_unlabeled_budget
```

## Shared-Support Synthetic Experiment

`synthetic_shared_support/` implements Section 4.4: source-only OLS, target-only OLS, mixed estimation, and Pseudo Lasso. Defaults are eight source environments with 1,000 observations each; 40 predictors; 20 shared, five source-specific, and five target-specific coefficients; 50 draws; a 1,000-observation test set; a 2,000-observation full-target OLS reference; and budgets `5, 10, 20, 30, 50, 100, 200, 400, 800`.

```bash
python -m synthetic_shared_support.run_experiment \
  --output-json results/synthetic_shared_support/results.json \
  --output-dir results/synthetic_shared_support/reported
```

The support screen requires source-coefficient sign agreement and relative standard error below `0.35`; the target correction penalty is selected by five-fold target cross-validation. Replot saved output with:

```bash
python -m synthetic_shared_support.replot_results \
  --results-json results/synthetic_shared_support/results.json \
  --output-dir results/synthetic_shared_support/reported
```

## Environment-Disjoint CIFAR-10-C

`CIFAR-10-C/` assigns every underlying CIFAR identity to exactly one of 19 corruption environments. IDs are shared across severity blocks only within their assigned environment. The split used for the reported experiments is class-stratified **70%/15%/15% train/validation/test**, yielding the recorded `370/80/80` or `360/80/80` base-image counts per environment.

This differs from the thesis text, which says a fixed 20% test split. The code below intentionally preserves the actual 70/15/15 experimental design, so the thesis should be corrected rather than regenerating the reported experiments.

The retained methods are source OLS and the anti-causal moment penalty, evaluated as ten one-vs-rest tasks at severity 5. Pixel inputs and frozen 384-dimensional DINOv2 ViT-S/14 embeddings are each reduced to PCA dimension 20. Unlabelled target budgets are `50, 100, 200, 300, all` (`-1`).

Create the reusable split:

```bash
python CIFAR-10-C/create_disjoint_environment_splits.py \
  --data-dir /path/to/CIFAR-10-C \
  --output results/cifar10c_disjoint_splits_seed_42.npz \
  --seed 42
```

Run the pixel PCA-20 source baseline and anti-causal penalty:

```bash
CIFAR-10-C/disjoint_source_baseline/run_source_baseline.sh \
  --data-dir /path/to/CIFAR-10-C \
  --split-file results/cifar10c_disjoint_splits_seed_42.npz \
  --severity 5 \
  --n-components 20 \
  --output-root results/cifar_pixel_pca20_source

CIFAR-10-C/disjoint_anti_causal_penalty/run_anti_causal_penalty.sh \
  --data-dir /path/to/CIFAR-10-C \
  --split-file results/cifar10c_disjoint_splits_seed_42.npz \
  --severity 5 \
  --n-components 20 \
  --budget 50,100,200,300,-1 \
  --baseline-root results/cifar_pixel_pca20_source \
  --output-root results/cifar_pixel_pca20_penalty
```

Extract DINO embeddings once. On an offline compute node, pass a local DINOv2 clone and checkpoint:

```bash
CIFAR-10-C/run_extract_dinov2_vits14_embeddings.sh \
  --data-dir /path/to/CIFAR-10-C \
  --output-root results/dinov2_vits14_embeddings \
  --model-repo /path/to/dinov2 \
  --weights-path /path/to/dinov2_vits14_pretrain.pth \
  --batch-size 512
```

Run DINO PCA-20 source baseline and penalty:

```bash
CIFAR-10-C/disjoint_source_baseline/run_source_baseline.sh \
  --data-dir /path/to/CIFAR-10-C \
  --split-file results/cifar10c_disjoint_splits_seed_42.npz \
  --severity 5 \
  --embedding-root results/dinov2_vits14_embeddings \
  --embedding-pca-components 20 \
  --output-root results/cifar_dinov2_pca20_source

CIFAR-10-C/disjoint_anti_causal_penalty/run_anti_causal_penalty.sh \
  --data-dir /path/to/CIFAR-10-C \
  --split-file results/cifar10c_disjoint_splits_seed_42.npz \
  --severity 5 \
  --embedding-root results/dinov2_vits14_embeddings \
  --embedding-pca-components 20 \
  --budget 50,100,200,300,-1 \
  --baseline-root results/cifar_dinov2_pca20_source \
  --output-root results/cifar_dinov2_pca20_penalty
```

Create aggregate curves and the task-by-environment heatmap with:

```bash
CIFAR-10-C/plot_anti_causal_penalty.sh \
  --data-root results \
  --baseline-dir cifar_pixel_pca20_source \
  --penalty-dir cifar_pixel_pca20_penalty \
  --penalty-label "Anti-causal penalty" \
  --aggregate-only
```

Use the DINO output names for the embedding plots. Add `--metric mse` for MSE summaries.

## Causal Chamber

`causal_chamber/` implements Section 4.5. It evaluates responses `ir_1`, `ir_2`, and `ir_3` over the intervention environments. Data are split 60%/20%/20% train/validation/test; train plus validation supply the 800 labelled target observations, and test contains 200 observations. Target budgets are `30, 50, 100, 200, 400, 800`, with three target-subset seeds.

The final implementation excludes all `ir_1`, `ir_2`, and `ir_3` columns from predictors. This differs from the thesis wording that says only the response is excluded and therefore reports 39 predictors. Update that wording before publication.

```bash
DATA_DIR=/path/to/lt_interventions_standard_v1
OUT_DIR=results/causal_chamber
MANIFEST="$OUT_DIR/splits.json"

for RESPONSE in ir_1 ir_2 ir_3; do
  python -m causal_chamber.run_experiment \
    --data-dir "$DATA_DIR" \
    --manifest "$MANIFEST" \
    --response "$RESPONSE" \
    --budgets 30 50 100 200 400 800 \
    --methods source_only_ols target_only_ols pseudo_lasso_centered mixed_estimator \
    --ranked-feature-counts 5 10 15 \
    --cv-folds 5 \
    --subset-seeds 0 1 2 \
    --output-root "$OUT_DIR" \
    --experiment-name "${RESPONSE}_3seeds"
done
```

The centered Pseudo Lasso and mixed estimator select their hyperparameters through five-fold target-data cross-validation. Pool the three response experiments with:

```bash
python -m causal_chamber.plot_results \
  --experiment-dir \
    results/causal_chamber/ir_1_3seeds \
    results/causal_chamber/ir_2_3seeds \
    results/causal_chamber/ir_3_3seeds \
  --pool-tasks \
  --integration-min-budget 30 \
  --output-dir results/causal_chamber/pooled
```
