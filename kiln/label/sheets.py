"""Contact sheets for reading boxes off images, checking boxes, and reviewing groups.

Label sheet: up to 12 images in a 4x3 grid. Each image is scaled to fit a tile and
overlaid with a coordinate grid labelled in ORIGINAL image pixels, so a box can be
read straight off the sheet as x0 y0 x1 y1 for boxes.txt. Boxes already in boxes.txt
(e.g. prefilled from existing labels) are drawn on top; overlay=True also marks
unlabelled images, for checking. With grouping on, a strip below the grid shows a
reference picture of each candidate scene (see kiln.label.match).

Scene sheet: one row per scene with a few frames spread across it, to spot two
scenes that should be merged, or a scene that mixes several.
"""

from __future__ import annotations

from pathlib import Path

from kiln.label.workspace import NEG, SHEETS, SKIP, Entry, Workspace

COLS, ROWS = 4, 3
TILE = 300  # px the longer image side is scaled to
MARGIN_L, HEADER, TICKS, PAD = 34, 18, 14, 10
GRID_STEPS = (5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000)
PALETTE = [(255, 64, 64), (64, 160, 255), (255, 200, 0), (0, 210, 120), (220, 90, 255),
           (255, 140, 0), (0, 220, 220), (255, 105, 180), (160, 200, 60), (180, 140, 100)]

GROUP_ROWS, GROUP_FRAMES, GROUP_TILE, GROUP_LABEL_W = 8, 5, 150, 150
CAND_TILE, CAND_GAP = 140, 8


def _font(size: int = 13):
    from PIL import ImageFont

    return ImageFont.load_default(size=size)


def grid_step(longest: int) -> int:
    """Smallest round step giving at most ~10 grid lines across the image."""
    return next((s for s in GRID_STEPS if longest / s <= 10), GRID_STEPS[-1])


def _fit(w: int, h: int, tile: int) -> tuple[int, int, float]:
    scale = tile / max(w, h)
    return max(1, round(w * scale)), max(1, round(h * scale)), scale


def _tile(ws: Workspace, i: str, entry: Entry | None, scene: str | None, overlay: bool):
    from PIL import Image, ImageDraw

    img = ws.images[i]
    tw, th, scale = _fit(img.w, img.h, TILE)
    cell = Image.new("RGB", (MARGIN_L + TILE + PAD, HEADER + TICKS + TILE + PAD), "white")
    with Image.open(ws.path_of(i)) as src:
        pic = src.convert("RGB").resize((tw, th))
    ox, oy = MARGIN_L, HEADER + TICKS
    cell.paste(pic, (ox, oy))

    d = ImageDraw.Draw(cell, "RGBA")
    f, small = _font(13), _font(10)
    name = Path(img.rel).name
    title = f"{i}  {name[:26] + '…' if len(name) > 27 else name}"
    if scene:
        title += f"  @{scene}"
    d.text((2, 1), title, fill="black", font=f)

    # coordinate grid in original pixels
    step = grid_step(max(img.w, img.h))
    for v in range(0, img.w + 1, step):
        x = ox + round(v * scale)
        d.line([(x, oy), (x, oy + th)], fill=(0, 255, 255, 90), width=1)
        if v % (step * 2) == 0 or img.w // step <= 6:
            lx = min(x - 6, ox + tw - 7 * len(str(v)))  # keep the last label inside the tile
            d.text((lx, HEADER), str(v), fill=(0, 90, 160), font=small)
    for v in range(0, img.h + 1, step):
        y = oy + round(v * scale)
        d.line([(ox, y), (ox + tw, y)], fill=(0, 255, 255, 90), width=1)
        if v % (step * 2) == 0 or img.h // step <= 6:
            d.text((2, y - 6), str(v), fill=(0, 90, 160), font=small)
    d.text((ox + tw - 60, oy + th + 1), f"{img.w}x{img.h}", fill="gray", font=small)

    if entry is not None:
        if entry.kind in (NEG, SKIP):
            d.rectangle([ox, oy, ox + 52, oy + 18], fill=(0, 0, 0, 170))
            d.text((ox + 4, oy + 2), entry.kind, fill="white", font=f)
        for cls, x0, y0, x1, y1 in entry.boxes:
            col = PALETTE[ws.classes.index(cls) % len(PALETTE)]
            box = [ox + x0 * scale, oy + y0 * scale, ox + x1 * scale, oy + y1 * scale]
            d.rectangle(box, outline=col + (255,), width=2)
            d.rectangle([box[0], box[1], box[0] + 7 * len(cls) + 6, box[1] + 14], fill=col + (200,))
            d.text((box[0] + 3, box[1]), cls, fill="black", font=small)
    elif overlay:
        d.rectangle([ox, oy, ox + 90, oy + 18], fill=(255, 0, 0, 170))
        d.text((ox + 4, oy + 2), "unlabelled", fill="white", font=f)
    return cell


def _pages(items: list, per_page: int) -> list[list]:
    return [items[k:k + per_page] for k in range(0, len(items), per_page)]


def _candidate_strip(ws: Workspace, cands: list[tuple[str, str]], width: int):
    """One reference thumbnail per candidate scene, named, to match sheet images against."""
    from PIL import Image, ImageDraw

    per_row = max(1, (width - CAND_GAP) // (CAND_TILE + CAND_GAP))
    cell_h = CAND_TILE + 32
    rows = (len(cands) + per_row - 1) // per_row
    strip = Image.new("RGB", (width, 24 + rows * cell_h), (232, 232, 232))
    d = ImageDraw.Draw(strip)
    f, small = _font(13), _font(10)
    d.text((CAND_GAP, 4), "Known scenes nearest these images: end a line with @<name>, or add a new scene",
           fill="black", font=f)
    chars = CAND_TILE // 7
    for n, (scene, ref) in enumerate(cands):
        x = CAND_GAP + (n % per_row) * (CAND_TILE + CAND_GAP)
        y = 24 + (n // per_row) * cell_h
        img = ws.images[ref]
        tw, th, _ = _fit(img.w, img.h, CAND_TILE)
        with Image.open(ws.path_of(ref)) as src:
            strip.paste(src.convert("RGB").resize((tw, th)), (x, y))
        d.text((x, y + th + 2), "@" + (scene if len(scene) <= chars else scene[:chars - 1] + "…"),
               fill="black", font=f)
        d.text((x, y + th + 17), f"ref {ref}", fill="gray", font=small)
    return strip


def label_sheets(ws: Workspace, ids: list[str], name: str, entries: dict[str, Entry],
                 scene_of: dict[str, str], overlay: bool = False,
                 cands_for=None) -> list[tuple[Path, list[tuple[str, str]]]]:
    """Render label sheets. cands_for(page ids) -> [(scene, ref id)] adds a candidate strip.
    Returns (path, candidates shown) per page."""
    from PIL import Image

    out_dir = ws.dir / SHEETS
    out_dir.mkdir(exist_ok=True)
    out = []
    pages = _pages(ids, COLS * ROWS)
    for k, page in enumerate(pages, 1):
        tiles = [_tile(ws, i, entries.get(i), scene_of.get(i), overlay) for i in page]
        cw, ch = tiles[0].size
        rows = (len(tiles) + COLS - 1) // COLS
        grid_w, grid_h = cw * min(COLS, len(tiles)), ch * rows
        cands = cands_for(page) if cands_for else []
        strip = _candidate_strip(ws, cands, grid_w) if cands else None
        sheet = Image.new("RGB", (grid_w, grid_h + (strip.height if strip else 0)), "white")
        for n, t in enumerate(tiles):
            sheet.paste(t, ((n % COLS) * cw, (n // COLS) * ch))
        if strip:
            sheet.paste(strip, (0, grid_h))
        p = out_dir / (f"{name}.png" if len(pages) == 1 else f"{name}-{k}.png")
        sheet.save(p)
        out.append((p, cands))
    return out


def scene_sheets(ws: Workspace, scenes: dict[str, list[str]], name: str) -> list[Path]:
    """One row per scene: name, size, and up to 5 frames spread across it."""
    from PIL import Image, ImageDraw

    out_dir = ws.dir / SHEETS
    out_dir.mkdir(exist_ok=True)
    f, small = _font(13), _font(10)
    row_h = GROUP_TILE + 18
    width = GROUP_LABEL_W + GROUP_FRAMES * (GROUP_TILE + 6)
    paths = []
    pages = _pages(list(scenes.items()), GROUP_ROWS)
    for k, page in enumerate(pages, 1):
        sheet = Image.new("RGB", (width, row_h * len(page)), "white")
        d = ImageDraw.Draw(sheet)
        for r, (g, ids) in enumerate(page):
            y = r * row_h
            d.line([(0, y), (width, y)], fill=(200, 200, 200))
            folder = ws.images[ids[0]].folder
            d.text((4, y + 4), "@" + (g if len(g) <= 19 else g[:18] + "…"), fill="black", font=f)
            d.text((4, y + 22), f"{len(ids)} image{'s' * (len(ids) != 1)}", fill="black", font=small)
            d.text((4, y + 36), ids[0] if len(ids) == 1 else f"{ids[0]} - {ids[-1]}", fill="gray", font=small)
            if folder:
                d.text((4, y + 50), folder[-22:], fill="gray", font=small)
            n = min(GROUP_FRAMES, len(ids))
            picks = [ids[round(j * (len(ids) - 1) / max(1, n - 1))] for j in range(n)]
            for c, i in enumerate(picks):
                img = ws.images[i]
                tw, th, _ = _fit(img.w, img.h, GROUP_TILE)
                with Image.open(ws.path_of(i)) as src:
                    pic = src.convert("RGB").resize((tw, th))
                x = GROUP_LABEL_W + c * (GROUP_TILE + 6)
                sheet.paste(pic, (x, y + 2))
                d.text((x, y + GROUP_TILE + 3), i, fill="black", font=small)
        p = out_dir / (f"{name}.png" if len(pages) == 1 else f"{name}-{k}.png")
        sheet.save(p)
        paths.append(p)
    return paths
