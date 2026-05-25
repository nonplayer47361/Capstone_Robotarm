# Berry YOLO Labeling Tools

This folder contains the labeling and research pipeline for clay-model object detection.

Class map:

```text
0 = blueberry
1 = strawberry
```

## Recommended Entry Point

Run:

```bat
RUN_RESEARCH_PIPELINE.bat
```

Menu targets:

```text
1. blueberry only
2. strawberry only
3. both sequential  (blueberry -> strawberry)
4. continue last saved session
```

Main actions:

```text
1. fresh start: reset, label holdout, then seed/manual labeling
2. seed/manual labeling, then optionally continue phased reviewed pipeline
3. run phased reviewed pipeline from seed/manual labels
4. label excluded 50 holdout images
5. status
6. evaluate latest model on holdout images
7. evaluate latest model on full source image pool
```

The BAT checks whether required packages are already installed. If they are missing, it creates `.venv_research` and installs `requirements.txt` there.

Every run writes logs to:

```text
research_runs/<target>/logs/
```

When scope `3` is selected, the two independent models are processed in this order:

```text
blueberry seed/manual labeling -> strawberry seed/manual labeling
-> blueberry Stage 1 + review -> strawberry Stage 1 + review
-> blueberry Stages 2-3-final -> strawberry Stages 2-3-final
```

This keeps the models separate while allowing one continuous research session.

The launcher saves the latest stop point to:

```text
research_runs/last_session.cmd
```

Saved resume points:

```text
stage1   seed/manual labels are ready; next is Stage 1 training + review
stage2   Stage 1 review is done; next is Stages 2-3-final
complete full phased pipeline completed
```

Choose scope `4` to continue from the saved point without remembering stage commands.

## Advanced Commands

Use the BAT for the normal research flow. Use direct commands only when resuming or rerunning a specific stage.

Examples:

```text
cd /d "C:\Users\dhtmd\OneDrive\바탕 화면\robotarm\yolo\labeling_tools"

python research_pipeline.py --config research_configs/blueberry.json --run --stage1-only
python research_pipeline.py --config research_configs/blueberry.json --run --start-stage 2
python research_pipeline.py --config research_configs/blueberry.json --run --start-stage 2 --no-review
python research_pipeline.py --config research_configs/strawberry.json --run --start-stage 3
python research_pipeline.py --config research_configs/strawberry.json --eval-holdout
```

For freeze comparison, run the variant configs directly:

```text
python research_pipeline.py --config research_configs/blueberry_f5.json --run
python research_pipeline.py --config research_configs/blueberry_f0.json --run
python research_pipeline.py --config research_configs/strawberry_f5.json --run
python research_pipeline.py --config research_configs/strawberry_f0.json --run
```

Stage 1 is the most important human review checkpoint. Stages 2-3 use an already-reviewed model and auto-accept rules, so `--no-review` is only recommended after Stage 1 has been carefully reviewed.

## Review Minimization

After Stage 1 has been reviewed by a person, Stage 2/3 use automatic review reduction.
Predictions that pass all configured quality checks are marked reviewed before the gallery opens:

```text
confidence >= auto_accept_confidence
exactly one detection, when auto_accept_require_single_detection=true
box area ratio within auto_accept_min/max_area_ratio
box aspect ratio within auto_accept_min/max_aspect_ratio
optional flip-consistency IoU, when auto_accept_consistency_iou > 0
```

The review gallery opens with `--unchecked-only`, so auto-accepted labels are hidden and only uncertain labels plus audit samples are shown.
`auto_accept_audit_ratio` keeps a small deterministic sample of accepted labels visible for human spot-checking.

Auto-accept reports are written to:

```text
research_runs/<target>/<stage_dataset>/auto_accept_report.json
```

Set `auto_accept_confidence` to `null` in a config to disable automatic acceptance and return to full review.

## Holdout Evaluation

The folders below are excluded from training and used for final model evaluation:

```text
blueberry_test_images/   (50 images)
strawberry_test_images/  (50 images)
```

For true mAP/precision/recall, these 50 images must have human-made ground-truth labels. Use action `8` for each target first. After the final model exists, use action `9` to evaluate it.

Action `7` runs a detection smoke test on the full source image pool. It reports detection success/failure, but it is not an unbiased accuracy score.

Recommended fresh-research order:

```text
1. Action 1: reset, label excluded 50 holdout images, then create seed/manual labels
2. Answer Y when asked to continue the phased reviewed pipeline
3. After Stage 1 review finishes, answer Y to continue stages 2-3-final
4. Action 6: rerun holdout evaluation if needed
```

If holdout labeling is completed before training finishes, the pipeline automatically runs holdout evaluation at the end. If skipped, action `4` and then action `6` must be run later.

Holdout labels and evaluation outputs are stored under:

```text
research_runs/<target>/12_holdout_test_dataset/
research_runs/<target>/13_holdout_eval_dataset/
research_runs/<target>/14_holdout_eval/
research_runs/<target>/11_reports/holdout_eval_report.md
```

## Freeze Ablation

To compare freeze=10 vs freeze=5 vs freeze=0 on the same source images:

```text
1. Run the normal BAT flow for blueberry/strawberry freeze=10.
2. Run `blueberry_f5`, `blueberry_f0`, `strawberry_f5`, and `strawberry_f0` with direct commands.
3. The variant configs reuse/copy the base holdout labels when available.
```

Holdout labeling is now included in Action 1 for fresh starts. Action 4 remains available when holdout labels need to be created or fixed separately.
Freeze variant configs copy the base holdout labels automatically when available, so they do not need separate holdout labeling.

For a full ablation run, use this order:

```text
1. BAT scope 3 + Action 1  - reset, label holdout, create seed labels, and run the base phased reviewed pipelines
3. Direct commands for freeze variants
```

Running without Action 4 will still train, but holdout reports will be skipped until the holdout labels are created.

Ablation results are in:

```text
research_runs/blueberry/11_reports/holdout_eval_report.md    (freeze=10)
research_runs/blueberry_f5/11_reports/holdout_eval_report.md (freeze=5)
research_runs/blueberry_f0/11_reports/holdout_eval_report.md (freeze=0)
```

## Manual Labeling Only

For simple manual labeling without the full research loop:

```bat
RUN_LABELING.bat
```

or:

```bash
python start_labeling.py --images <image_folder> --dataset-dir <dataset_folder> --target strawberry
python start_labeling.py --images <image_folder> --dataset-dir <dataset_folder> --target blueberry
```

## Review Tool

The research pipeline opens the review tool automatically when review is enabled.

Manual launch example:

```bash
python review_label_gallery.py --dataset-dir research_runs/blueberry/02_stage1_auto_dataset --all --sequential
```

The review tool supports detailed one-by-one inspection. In the large review window:

```text
Mouse drag              = replace or add a box
Click inside a box      = select that box
Arrow keys              = move selected box by 1 pixel
Shift + Arrow keys      = move selected box by 5 pixels
Enter                   = save, mark OK, next image
Ctrl + Right / Down     = save, mark OK, next image
Ctrl + Left / Up        = previous image
PageDown / PageUp       = next / previous image
Ctrl+S                  = save without moving
Z                       = undo last box
Esc                     = cancel current drag
```

When reviewing a single-class model (blueberry or strawberry), the class selector is locked to the target class. All saved boxes are automatically forced to that class, preventing accidental cross-class labels.

The manual labelers also support fine adjustment after drawing a box. In the canvas labeler, arrow keys move the selected box by 1 pixel and Shift+Arrow moves it by 5 pixels. In the blueberry OpenCV labeler, arrow keys move by 1 pixel and `i`/`j`/`k`/`l` move the selected box by 5 pixels.

## Center Re-inference Report

For robot-arm experiments, center-point accuracy can be more useful than tight-box IoU. Use:

```bat
RUN_CENTER_REINFER_REPORT.bat
```

or run directly:

```bash
python center_reinfer_report.py --target both --model-set both --conf 0.25
python center_reinfer_report.py --target blueberry --model-set fullauto
python center_reinfer_report.py --target strawberry --model-set custom --custom-model path\to\best.pt --custom-name robot_camera_v1
```

Reports are saved under:

```text
fullauto/comparison_reports/
```

The report includes detection count, mean/median/p90 center error in pixels, normalized center error, center-inside-label count, confidence, and IoU. For deployment, convert the pixel center error to paper/world units using the fixed robot-camera pose.

## Research Configs

Per-target settings live in `research_configs/`. Key fields:

```text
blueberry.json       freeze=10, holdout: blueberry_test_images
strawberry.json      freeze=10, holdout: strawberry_test_images
blueberry_f5.json    freeze=5,  holdout: reused from blueberry
blueberry_f0.json    freeze=0,  holdout: reused from blueberry
strawberry_f5.json   freeze=5,  holdout: reused from strawberry
strawberry_f0.json   freeze=0,  holdout: reused from strawberry
```

## YOLO Dataset Layout

All outputs use this structure:

```text
dataset/
  train/images/
  train/labels/
  val/images/
  val/labels/
  dataset.yaml
```

The train/val split is deterministic from the image filename.

For a fresh research run, the first manual seed set is sampled evenly across the full source image folder so old and newly added photos are both represented.
