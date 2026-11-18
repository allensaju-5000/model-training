---
name: label-split
description: Label images with bounding boxes by eye and split them into train/val/test by scene, using the kiln CLI. Use when the user runs /label-split or asks Claude to box-label a folder of images (unlabelled, class-folder, or pre-labelled to review) and optionally split it.
argument-hint: <source folder> [train val test, e.g. 70 15 15] [--classes a,b] [--guide file.md]
---

# label-split

You label every image in a folder with boxes by looking at it, and in the same pass
put each image in a scene (same place, camera, event or video). If ratios are given,
whole scenes are then split into train/val/test, so near-identical frames never
leak between train and test. The
`kiln label` CLI does all file work; you do the looking and judging. Run every
command as `kiln …` from the project that has kiln installed (prefix `uv run`
when kiln lives in a uv project and is not on PATH).

Output is a YOLO dataset the user then reviews with
`kiln data load <out> --name <n>` (which opens the FiftyOne app), and exports with
`kiln data export`.

## 0. Arguments and setup

- Source folder: required. Ratios: optional three numbers summing to 100; no
  ratios means label only, no split.
- Workspace `--work`: default `<source parent>/<source name>-kiln-work`.
  Output `--out`: default `<source parent>/<source name>-labelled`.
- If the workspace already has a `manifest.json`, **resume**: skip to step 3 and
  run `kiln label check -w <work>` to see what is left. Never delete or re-init a
  workspace with work in it.

## 1. Init

```
kiln label init <source> -w <work> [--classes a,b] [--guide <file>] [--no-grouping]
```

- Grouping is on by default. Pass `--no-grouping` only when the user says not to
  group; then lines carry no `@scene` and skip everything about scenes below.

- Classes come from `--classes`, a `kiln.toml` beside the source, or the source's
  own labels. If none are found, **ask the user** for the class list and order
  (order = class ids, keep it consistent with their other datasets).
- Existing labels are prefilled into `boxes.txt` (review them, do not trust them
  blindly). `negative_folders` from kiln.toml are prefilled as `NEG`.

## 2. Labelling guide

Read the guide (`init` prints its path). It says what each class covers, where
boxes start and stop, and what to skip. If there is no guide, **ask the user**
what to box and the edge cases, then write `<work>/guide.md` from the answers and
follow it. Do not invent conventions silently; when you meet a case the guide does
not cover, decide, then add a line to the guide so later images stay consistent,
and mention it in your final report.

## 3. Label and group (one pass)

```
kiln label sheet -w <work> --todo 12
```

Each tile shows the id, file name, and a grid labelled in **original image pixels**.
Boxes already in `boxes.txt` (prefilled labels) are drawn on the tile. Below the
grid, a strip shows a reference picture of the known scenes most like these
images (also printed as text with their descriptions). For each image write one
line to `<work>/boxes.txt`, ending in its scene:

```
0137: fire 100 60 260 190; smoke 0 0 400 120 @lava_night
0138: NEG @forest_cam2
0139: SKIP    # not this dataset's domain (say why in a comment)
```

Boxes:
- `x0 y0 x1 y1` = top-left and bottom-right corners, read off the grid.
  Estimate to the nearest ~5 px; tight boxes matter more than exact numbers.
- One box per distinct object/region as the guide defines it. Prefilled lines:
  correct them in place (and add the `@scene`) rather than adding a second line.
- Images of one scene look alike: label the first carefully, then adjust for the
  next. Still look at every image; things move.

Scenes:
- **Same scene** = same place, camera, event or video, even if angle, zoom,
  brightness or the fire itself changed. If an image matches a scene in the strip
  (or one listed in `scenes.txt` that you remember), end its line with `@<name>`.
- **New scene**: pick a short name (letters, digits, `_ . -`, e.g. `lava_night`,
  `kitchen_cam2`) and add a line `<name>: <what it shows>` to `<work>/scenes.txt`
  *before* using it. Make the description specific enough to recognise the
  scene again later (place, camera view, lighting, distinctive objects).
- Images on one sheet can start a scene that the next image on the same sheet
  joins; the strip only shows scenes that existed before the sheet.
- **When unsure, join** the closest scene rather than starting a new one. A scene
  split across train and test inflates test scores; a slightly large scene costs
  almost nothing. Unrelated stock photos with no shared scene each get their own
  scene (e.g. `stock_0137`), or share one only if they truly are one shoot.
- SKIP lines need no scene.

Write each sheet's lines with heredocs (`cat >> <work>/boxes.txt <<'EOF' … EOF`,
and the same for new scenes in `scenes.txt`) one sheet at a time, so progress is
saved continuously and work can resume after a break. Run
`kiln label check -w <work>` every few sheets; it catches typos in scene
names and boxes.

## 4. Verify

Boxes:

```
kiln label sheet -w <work> --ids <ids> --overlay
```

Check every image with 3+ boxes, every image you were unsure about, and a sample
of at least 1 in 10 of the rest. Fix lines in `boxes.txt` that are loose, shifted,
or missing an object.

Scenes (one row per scene with a few frames):

```
kiln label sheet -w <work> --review-scenes all
kiln label merge -w <work> <keep> <other> [<other> …]   # same scene, two names
```

Read every page. Merge scenes that are really one. If a scene mixes different
places, change the `@scene` of the stray lines to another (or a new) scene. Then:

```
kiln label check -w <work>       # must end with "boxes.txt OK" and 0 to do
```

## 5. Build

```
kiln label build -w <work> -o <out> [--ratios "70 15 15"]
```

Read the stratum table: each kind of image (folder | classes) should be near the
requested ratios in every split. If a stratum is badly off (usually one huge scene),
consider splitting that scene if it is genuinely several, or re-run with
another `--seed`. Use `--overwrite` to rebuild the same output folder.

## 6. Report to the user

Briefly: images labelled, boxes per class, NEG and SKIP counts (and why skipped),
scenes (count and largest), the split table, any conventions you added to the guide, anything you were
unsure about (list ids). Then give the review commands:

```
kiln data load <out> --name <name>      # also opens the FiftyOne app
kiln data export <name> -o <export dir> [-f coco]
```

## Pacing

Each sheet is one image read. For large folders (≈500+ images) tell the user the
expected size of the job before starting, and offer to do it in sessions;
the workspace makes stopping and resuming safe (`kiln label check` shows what is left).
