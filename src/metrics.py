import csv
import math
from collections import defaultdict

import numpy as np
import torch.nn.functional as F


def flipped(s0, s1, tau, direction):
    """True if the pair crossed tau in the requested direction."""
    if direction == "accept_to_reject":
        return s0 >= tau and s1 < tau
    return s0 < tau and s1 >= tau


def l2(a, b):
    """||a - b||_2."""
    return float((a - b).norm())


def pesq_stoi(ref, deg, sr=16000):
    """Wideband PESQ and STOI, or (None, None) if unavailable."""
    if ref is None or deg is None:
        return None, None
    n = min(ref.shape[-1], deg.shape[-1])
    r = ref[..., :n].detach().cpu().float().numpy().squeeze()
    d = deg[..., :n].detach().cpu().float().numpy().squeeze()
    p = s = None
    try:
        from pesq import pesq as _pesq
        p = float(_pesq(sr, r, d, "wb"))
    except Exception:
        pass
    try:
        from pystoi import stoi as _stoi
        s = float(_stoi(r, d, sr, extended=False))
    except Exception:
        pass
    return p, s


def binarize(mask, tau_m=0.5):
    """Threshold a [0, 1] attribution map."""
    a = mask.abs()
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    return (a > tau_m).float()


def mask_cosine(a, b):
    """Cosine similarity of two flattened binarized masks."""
    x, y = binarize(a).flatten(), binarize(b).flatten()
    n = min(x.numel(), y.numel())
    return float(F.cosine_similarity(x[:n].unsqueeze(0), y[:n].unsqueeze(0)))


def mask_cosine_matrix(masks):
    """Pairwise cosine similarity dict for named binarized masks."""
    names = list(masks)
    out = {n: {} for n in names}
    for i, a in enumerate(names):
        for b in names[i:]:
            v = 1.0 if a == b else mask_cosine(masks[a], masks[b])
            out[a][b] = out[b][a] = v
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else None


def summarize(rows):
    """Group rows by baseline, strategy, direction; return SR, L2, |delta|, PESQ, STOI, iters."""
    g = defaultdict(list)
    for r in rows:
        g[(r.get("baseline"), r.get("strategy"), r.get("direction"))].append(r)
    out = {}
    for k, rs in g.items():
        bl, st, d = k
        out.setdefault(bl, {}).setdefault(st, {})[d] = {
            "n": len(rs),
            "success_rate": 100.0 * np.mean([bool(r.get("flipped")) for r in rs]),
            "l2": _mean([r.get("l2") for r in rs]),
            "abs_delta": _mean([abs(r["score_after"] - r["score_before"]) for r in rs if r.get("score_after") is not None]),
            "pesq": _mean([r.get("pesq") for r in rs]),
            "stoi": _mean([r.get("stoi") for r in rs]),
            "iters": _mean([r.get("iters") for r in rs]),
        }
    return out


def write_csv(path, rows):
    """Write experiment rows to CSV."""
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    """Load rows written by write_csv."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("score_before", "score_after", "l2", "pesq", "stoi", "iters", "tau"):
            if r.get(k) in (None, "", "None"):
                r[k] = None
            else:
                r[k] = float(r[k])
        r["flipped"] = str(r.get("flipped", "")).lower() == "true"
        r["same_speaker"] = str(r.get("same_speaker", "")).lower() == "true"
    return rows
