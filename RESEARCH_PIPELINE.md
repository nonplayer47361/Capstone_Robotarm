# YOLO Research Pipeline

Use `RUN_RESEARCH_PIPELINE.bat` for blueberry and strawberry clay-object datasets.

The launcher intentionally keeps the main menu broad:

```text
Scope: blueberry only / strawberry only / both sequential
Action: fresh start, seed/manual labeling, full pipeline, holdout labeling, status, evaluation
```

There is also a fourth scope option, `continue last saved session`, which resumes from the last stop point saved by the launcher.

When `both sequential` is selected, the order is:

```text
blueberry seed/manual labeling -> strawberry seed/manual labeling
-> blueberry Stage 1 + review -> strawberry Stage 1 + review
-> blueberry Stages 2-3-final -> strawberry Stages 2-3-final
```

Stage-specific reruns are handled with direct `research_pipeline.py` commands instead of extra BAT menu items.

The BAT saves its resume state to `research_runs/last_session.cmd`:

```text
stage1   seed/manual labels are ready; next is Stage 1 training + review
stage2   Stage 1 review is done; next is Stages 2-3-final
complete full phased pipeline completed
```

## Goal

The pipeline turns a small manual seed set into a reviewed YOLO dataset:

1. Create the manual seed labels (50 images).
2. Train a seed model.
3. Auto-label the remaining images.
4. Open the review gallery so incorrect boxes can be fixed.
5. Retrain and relabel the full image pool through repeated review stages.
6. Train the final model.
7. Evaluate on the held-out 50 images and write a report.

Review stages are intentionally blocking. When review is enabled, every shown item must be checked and saved before the next training stage continues.

Review windows open in detailed sequential mode by default. Use mouse drag to add or replace a box, click inside a box to select it, then use arrow keys for 1-pixel nudging or Shift+Arrow for 5-pixel nudging. Use Enter, Ctrl+Right/Down, or PageDown to save the current image, mark it reviewed, and move to the next one. Use Ctrl+Left/Up or PageUp to go back.

## Review Minimization

Stage 1 remains a required human checkpoint because it is trained from only the seed labels. After Stage 1, the pipeline reduces human review by auto-accepting high-quality predictions before opening the gallery.

Auto-accept requires high confidence plus basic box sanity checks: single detection, valid area ratio, valid aspect ratio, and optional flip-consistency IoU. Accepted labels are written into `review_status.json` as already reviewed. The gallery is launched with `--unchecked-only`, so people see only uncertain labels and the configured audit sample.

Key config fields:

```json
{
  "auto_accept_confidence": 0.85,
  "auto_accept_min_area_ratio": 0.0003,
  "auto_accept_max_area_ratio": 0.35,
  "auto_accept_min_aspect_ratio": 0.55,
  "auto_accept_max_aspect_ratio": 1.75,
  "auto_accept_require_single_detection": true,
  "auto_accept_audit_ratio": 0.05,
  "auto_accept_consistency_iou": 0.0
}
```

Set `auto_accept_confidence` to `null` for full manual review. Each auto-label stage writes `auto_accept_report.json` into the dataset folder.

## Main Files

```text
RUN_RESEARCH_PIPELINE.bat   menu launcher with logging
research_pipeline.py        generic research workflow
active_learning_core.py     train/predict/dataset utilities
review_label_gallery.py     checkbox review and label edit tool
start_labeling.py           manual seed labeler launcher
research_configs/           per-target settings
```

## Target Config

Each target has a JSON config. All fields:

```json
{
  "target_name": "blueberry",
  "source_images": "blueberry_jpg",
  "manual_seed_count": 50,
  "class_id": 0,
  "class_names": ["blueberry", "strawberry"],
  "base_model": "yolo11n.pt",
  "freeze": 10,
  "confidence": 0.35,
  "fallback_confidence": 0.01,
  "manual_seed_step": 25,
  "max_manual_seed": 150,
  "min_stage1_auto_labels": 30,
  "test_images": "blueberry_test_images",
  "holdout_source": null,
  "require_review_complete": true
}
```

`class_id` must be set to 0 (blueberry) or 1 (strawberry). The review tool locks the class selector to this value to prevent cross-class labeling errors.

`freeze` controls how many backbone layers are frozen during training (0 = full fine-tuning, 10 = head-only). The ablation configs vary this value while keeping everything else identical.

`holdout_source` enables ablation variants to reuse holdout labels from the base target instead of labeling the same images again. Set to the base target name (e.g. `"blueberry"`). If `null`, the holdout labeler is opened.

`test_images` points to images excluded from all training stages. Set to `null` to skip holdout evaluation.

## Output Layout

```text
research_runs/<target>/
  00_sources/
  01_manual_seed_dataset/
  02_stage1_auto_dataset/
  03_stage1_reviewed_dataset/
  04_stage2_relabel_dataset/
  05_stage2_reviewed_dataset/
  06_stage3_relabel_dataset/
  07_stage3_reviewed_dataset/
  08_missed_images/
  09_final_dataset/
  10_detection_eval/
  11_reports/
  12_holdout_test_dataset/
  13_holdout_eval_dataset/
  14_holdout_eval/
  runs/
  logs/
  archive/
```

Every BAT run writes a terminal log into `research_runs/<target>/logs/`.

## Holdout Test

`test_images` in each config points to images excluded from training. They are not used by the active-learning stages. To get real mAP/precision/recall, label them with BAT action `4`, then evaluate the latest model with action `6`.

The holdout evaluation copies every labeled holdout image into an all-validation dataset before calling Ultralytics `val`, so all 50 excluded images are measured together.

The full source-pool evaluation is available as BAT action `7`. It runs object-detection success/failure over every source image. Treat it as a coverage check, not as the final research metric.

For a fresh run, use action `1`. It resets selected outputs, opens holdout labeling first, then opens seed/manual labeling, and then asks whether to continue the phased reviewed pipeline. If holdout labels are ready before training finishes, the final training stage automatically runs holdout evaluation. If holdout labels are missing, training still completes, but holdout evaluation is skipped until action `4` and then action `6` are run.

## Freeze Ablation

Three freeze values are compared per class (0, 5, 10). Each value runs the full pipeline independently. The base config (freeze=10) must be run first; ablation variants automatically copy the base holdout labels via `holdout_source`.

Freeze ablation is no longer part of the main BAT menu. Use direct commands after the base blueberry/strawberry runs are complete:

```text
python research_pipeline.py --config research_configs/blueberry_f5.json --run
python research_pipeline.py --config research_configs/blueberry_f0.json --run
python research_pipeline.py --config research_configs/strawberry_f5.json --run
python research_pipeline.py --config research_configs/strawberry_f0.json --run
```

Note: because each freeze value runs an independent pipeline (separate seed labeling, auto-labeling, and review), this comparison measures end-to-end pipeline performance under each freeze setting, not an isolated weight-initialization effect. If the goal is to isolate freeze as a single variable on identical training data, a shared-dataset variant would be needed.

## When Auto-Labeling Fails

If stage 1 produces too few automatic labels, the pipeline first retries with `fallback_confidence`. If that is still not enough, it opens the manual labeler for more seed images in `manual_seed_step` chunks until `max_manual_seed` is reached.

This keeps the research loop practical while avoiding silent progress with an unusable model.

Manual seed images are selected evenly across the full source image list, not simply from the first 50 files. This makes the initial seed more representative after new photos are added.
