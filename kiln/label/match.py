"""Shortlist the scenes an image most likely belongs to, so a label sheet shows a few
candidate scenes instead of the whole catalogue. Claude makes the actual call.

Similarity is cheap and rough on purpose: a 64-bit difference hash (structure: edges,
layout) plus an 8x8 colour thumbnail. It only decides which reference pictures to show.
Features are computed once per image and cached in <work>/.features.json.
"""

from __future__ import annotations

import json

import numpy as np

from kiln.label.workspace import FEATURES, Entry, Workspace

PER_IMAGE = 3       # nearest scenes kept per sheet image
LIMIT = 8           # candidates shown per sheet
RECENT = 2          # scenes of the images just before the sheet, always shown
MEMBER_SAMPLE = 24  # members compared per scene (spread over the scene)


def dhash(path, size: int = 8) -> int:
    """64-bit difference hash: is each pixel of a tiny grey thumbnail brighter than its right neighbour?"""
    from PIL import Image

    with Image.open(path) as im:
        px = list(im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS).getdata())
    bits = 0
    for row in range(size):
        for col in range(size):
            a, b = px[row * (size + 1) + col], px[row * (size + 1) + col + 1]
            bits = (bits << 1) | (a > b)
    return bits


def colour_thumb(path, size: int = 8) -> list[int]:
    from PIL import Image

    with Image.open(path) as im:
        return [v for px in im.convert("RGB").resize((size, size), Image.Resampling.BOX).getdata() for v in px]


def features(ws: Workspace, ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(bits [n, 64] bool, colours [n, 192] float) for ids, computing and caching missing ones."""
    path = ws.dir / FEATURES
    try:
        cache = json.loads(path.read_text())
    except (OSError, ValueError):
        cache = {}
    missing = [i for i in ids if i not in cache]
    for i in missing:
        cache[i] = [dhash(ws.path_of(i)), colour_thumb(ws.path_of(i))]
    if missing:
        path.write_text(json.dumps(cache, separators=(",", ":")))
    bits = np.array([[(cache[i][0] >> (63 - b)) & 1 for b in range(64)] for i in ids], dtype=bool)
    cols = np.array([cache[i][1] for i in ids], dtype=np.float32)
    return bits, cols


def _sample(ids: list[str]) -> list[str]:
    if len(ids) <= MEMBER_SAMPLE:
        return ids
    step = len(ids) / MEMBER_SAMPLE
    return [ids[int(k * step)] for k in range(MEMBER_SAMPLE)]


def candidates(ws: Workspace, page: list[str], entries: dict[str, Entry],
               members: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Scenes to show beside a sheet of `page` ids, as (scene, reference image id), best first.

    The scenes of the labelled images just before the sheet come first (consecutive files
    are often one scene), then each image's nearest scenes by distance to their members.
    """
    if not members:
        return []
    owner, mids = [], []
    for s, ids in members.items():
        for j in _sample(ids):
            owner.append(s)
            mids.append(j)
    pb, pc = features(ws, page)
    mb, mc = features(ws, mids)
    dist = (pb[:, None, :] != mb[None, :, :]).mean(-1) + np.abs(pc[:, None, :] - mc[None, :, :]).mean(-1) / 128
    for r, i in enumerate(page):  # an image already in a scene must not match itself
        dist[r, [k for k, j in enumerate(mids) if j == i]] = np.inf

    best: dict[str, tuple[float, str]] = {}
    for r in range(len(page)):
        nearest: dict[str, tuple[float, str]] = {}
        for k in np.argsort(dist[r]):
            if not np.isfinite(dist[r, k]):
                break
            nearest.setdefault(owner[k], (float(dist[r, k]), mids[k]))
            if len(nearest) == PER_IMAGE:
                break
        for s, (d, j) in nearest.items():
            if s not in best or d < best[s][0]:
                best[s] = (d, j)

    order = list(ws.images)
    recent: dict[str, str] = {}
    for i in reversed(order[max(0, order.index(page[0]) - 50):order.index(page[0])]):
        e = entries.get(i)
        if e is not None and e.scene in members and e.scene not in recent:
            recent[e.scene] = i
            if len(recent) == RECENT:
                break

    out = [(s, best[s][1] if s in best else j) for s, j in recent.items()]
    out += [(s, j) for d, s, j in sorted((d, s, j) for s, (d, j) in best.items()) if s not in recent]
    return out[:LIMIT]
