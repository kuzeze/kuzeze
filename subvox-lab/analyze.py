#!/usr/bin/env python3
"""Offline analysis for Subvox recordings (joeblau/subvox).

A faithful NumPy port of the app's decoding pipeline (trim -> features -> z-scale ->
DTW 1-NN -> leave-one-out + label-permutation test), plus the two things the app
cannot do on its own:

  * feature ablation over all 7 channel combinations in one run
  * cross-session evaluation (train on one recording, test on another), which is
    the honest test for "does it survive re-seating the AirPods / a new day?"

Inputs it understands
  dataset  ~/Library/Application Support/Subvox/dataset.json   (the Train screen takes)
  session  ~/Documents/Subvox/Sessions/<timestamp>-<condition>/ (Record Session output)

Examples
  python3 analyze.py dataset "~/Library/Application Support/Subvox/dataset.json"
  python3 analyze.py session ~/Documents/Subvox/Sessions/2026*-silent
  python3 analyze.py cross  ~/Documents/Subvox/Sessions/A-silent ~/Documents/Subvox/Sessions/B-silent
  python3 analyze.py selftest

Only NumPy is required. Add --json out.json to save every number for sharing.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

COLS = ["t", "ax", "ay", "az", "rx", "ry", "rz", "pitch", "roll", "yaw",
        "gx", "gy", "gz", "qw", "qx", "qy", "qz"]
ALL_FEATURES = ("accel", "rot", "att")


class Take:
    """One labelled segment: samples is an [n, 17] array in COLS order."""

    def __init__(self, label: str, samples: np.ndarray, source: str = ""):
        self.label = label
        self.samples = samples
        self.source = source

    @property
    def duration(self) -> float:
        return float(self.samples[-1, 0] - self.samples[0, 0]) if len(self.samples) > 1 else 0.0


# ----------------------------------------------------------------------------- loading

def load_dataset(path: str) -> list[Take]:
    p = Path(path).expanduser()
    data = json.loads(p.read_text())
    takes = []
    for tpl in data.get("templates", []):
        rows = [[float(s.get(c, 0.0)) for c in COLS] for s in tpl.get("samples", [])]
        if len(rows) >= 4:
            takes.append(Take(str(tpl["label"]), np.array(rows), f"dataset:{p.name}"))
    return takes


def load_session(path: str, kinds=("train", "segment")) -> tuple[list[Take], np.ndarray, dict]:
    d = Path(path).expanduser()
    motion_path = d / "motion.csv"
    with open(motion_path) as f:
        header = f.readline().strip().split(",")
    raw = np.loadtxt(motion_path, delimiter=",", skiprows=1, ndmin=2)
    if raw.size == 0:
        raise SystemExit(f"{motion_path} has no samples")
    col = {name: i for i, name in enumerate(header)}
    M = np.stack([raw[:, col[c]] for c in COLS], axis=1)

    takes = []
    events_path = d / "events.csv"
    if events_path.exists():
        with open(events_path) as f:
            for row in csv.DictReader(f):
                if row["kind"] not in kinds:
                    continue
                s, e = float(row["start"]), float(row["end"])
                seg = M[(M[:, 0] >= s - 1e-6) & (M[:, 0] <= e + 1e-6)]
                if len(seg) >= 4:
                    takes.append(Take(row["label"], seg, f"session:{d.name}"))
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    return takes, M, meta


def load_any(path: str, kinds) -> list[Take]:
    p = Path(path).expanduser()
    if p.is_dir():
        return load_session(p, kinds)[0]
    return load_dataset(p)


# ----------------------------------------------------------------------------- features
# Ported from Subvox/Decoding/MotionFeatures.swift.

def energy(S: np.ndarray) -> np.ndarray:
    rot = np.linalg.norm(S[:, 4:7], axis=1)
    acc = np.linalg.norm(S[:, 1:4], axis=1)
    return rot + 5.0 * acc


def trim(S: np.ndarray, padding: int = 2) -> np.ndarray:
    n = len(S)
    if n <= 6:
        return S
    e = energy(S)
    srt = np.sort(e)
    floor, peak = srt[n // 5], srt[-1]
    if not peak > floor * 1.5:
        return S
    thr = floor + 0.15 * (peak - floor)
    idx = np.nonzero(e > thr)[0]
    if len(idx) == 0:
        return S
    lo = max(0, int(idx[0]) - padding)
    hi = min(n - 1, int(idx[-1]) + padding)
    if hi - lo + 1 < 5:
        return S
    return S[lo:hi + 1]


def wrap(a: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a), np.cos(a))


def frames(S: np.ndarray, fs) -> np.ndarray:
    parts = []
    if "accel" in fs:
        parts.append(S[:, 1:4])
    if "rot" in fs:
        parts.append(S[:, 4:7])
    if "att" in fs:
        parts.append(wrap(S[:, 7:10] - S[0, 7:10]))
    F = np.concatenate(parts, axis=1)
    return F - F.mean(axis=0, keepdims=True)


def channel_scales(seqs: list[np.ndarray]) -> np.ndarray:
    allf = np.concatenate(seqs, axis=0)
    if len(allf) <= 1:
        return np.ones(allf.shape[1])
    return np.maximum(allf.std(axis=0), 1e-6)


# ----------------------------------------------------------------------------- DTW
# Ported from Subvox/Decoding/DTW.swift (Sakoe-Chiba band, normalised by n+m).

def dtw(a: np.ndarray, b: np.ndarray, band_fraction: float = 0.25) -> float:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return math.inf
    band = max(abs(n - m), int(max(n, m) * band_fraction))
    prev = np.full(m + 1, np.inf)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur = np.full(m + 1, np.inf)
        lo, hi = max(1, i - band), min(m, i + band)
        if lo <= hi:
            cost = np.linalg.norm(b[lo - 1:hi] - a[i - 1], axis=1)
            for j in range(lo, hi + 1):
                cur[j] = cost[j - lo] + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return float(prev[m] / (n + m))


def distance_matrix(seqs: list[np.ndarray], band: float) -> np.ndarray:
    n = len(seqs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = dtw(seqs[i], seqs[j], band)
    return D


# ----------------------------------------------------------------------------- evaluation

def nearest_per_label(D: np.ndarray, assign: np.ndarray, n_labels: int) -> np.ndarray:
    """best[i, l] = min distance from query i to any reference with label l. D may be [q, r]."""
    best = np.full((D.shape[0], n_labels), np.inf)
    for l in range(n_labels):
        mask = assign == l
        if mask.any():
            best[:, l] = D[:, mask].min(axis=1)
    return best


def evaluate(takes: list[Take], fs, permutations=500, band=0.25, seed=0):
    """Leave-one-out 1-NN DTW plus label-permutation test. Port of Evaluation.run."""
    labels = sorted({t.label for t in takes})
    counts = {l: sum(1 for t in takes if t.label == l) for l in labels}
    if len(labels) < 2 or min(counts.values()) < 2 or not fs:
        return None
    raw = [frames(trim(t.samples), fs) for t in takes]
    scales = channel_scales(raw)
    seqs = [r / scales for r in raw]
    n = len(seqs)
    D = distance_matrix(seqs, band)
    np.fill_diagonal(D, np.inf)  # leave-one-out: never match a take to itself
    idx = {l: k for k, l in enumerate(labels)}
    truth = np.array([idx[t.label] for t in takes])

    def loo(assign):
        return nearest_per_label(D, assign, len(labels)).argmin(axis=1)

    pred = loo(truth)
    acc = float((pred == truth).mean())
    conf = np.zeros((len(labels), len(labels)), dtype=int)
    for p, a in zip(pred, truth):
        conf[a, p] += 1

    rng = np.random.default_rng(seed)
    null = np.array([float((loo(s) == s).mean()) for s in (rng.permutation(truth) for _ in range(permutations))])
    null.sort()
    return {
        "labels": labels,
        "n": n,
        "counts": counts,
        "accuracy": acc,
        "chance": max(counts.values()) / n,
        "null_mean": float(null.mean()) if len(null) else 0.0,
        "null_p95": float(null[int((len(null) - 1) * 0.95)]) if len(null) else 0.0,
        "p_value": (int((null >= acc).sum()) + 1) / (permutations + 1),
        "permutations": permutations,
        "confusion": conf.tolist(),
        "features": list(fs),
    }


def evaluate_cross(train: list[Take], test: list[Take], fs, permutations=500, band=0.25, seed=0):
    """Train on one recording, test on another. Scales come from the training set only."""
    labels = sorted({t.label for t in train})
    missing = sorted({t.label for t in test} - set(labels))
    test = [t for t in test if t.label in labels]
    if len(labels) < 2 or not test or not fs:
        return None
    idx = {l: k for k, l in enumerate(labels)}
    raw_train = [frames(trim(t.samples), fs) for t in train]
    scales = channel_scales(raw_train)
    seq_train = [r / scales for r in raw_train]
    seq_test = [frames(trim(t.samples), fs) / scales for t in test]
    D = np.array([[dtw(q, r, band) for r in seq_train] for q in seq_test])
    y_train = np.array([idx[t.label] for t in train])
    y_test = np.array([idx[t.label] for t in test])

    def predict(assign):
        return nearest_per_label(D, assign, len(labels)).argmin(axis=1)

    pred = predict(y_train)
    acc = float((pred == y_test).mean())
    conf = np.zeros((len(labels), len(labels)), dtype=int)
    for p, a in zip(pred, y_test):
        conf[a, p] += 1
    rng = np.random.default_rng(seed)
    null = np.array([float((predict(rng.permutation(y_train)) == y_test).mean()) for _ in range(permutations)])
    null.sort()
    test_counts = {l: int((y_test == idx[l]).sum()) for l in labels}
    return {
        "labels": labels,
        "n_train": len(train),
        "n_test": len(test),
        "test_counts": test_counts,
        "labels_only_in_test": missing,
        "accuracy": acc,
        "chance": max(test_counts.values()) / len(test),
        "null_mean": float(null.mean()) if len(null) else 0.0,
        "null_p95": float(null[int((len(null) - 1) * 0.95)]) if len(null) else 0.0,
        "p_value": (int((null >= acc).sum()) + 1) / (permutations + 1),
        "permutations": permutations,
        "confusion": conf.tolist(),
        "features": list(fs),
    }


def feature_combos():
    for k in range(1, 4):
        for combo in itertools.combinations(ALL_FEATURES, k):
            yield combo


# ----------------------------------------------------------------------------- reporting

def pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def describe_takes(takes: list[Take]) -> None:
    labels = sorted({t.label for t in takes})
    print(f"{len(takes)} takes, {len(labels)} labels")
    print(f"  {'label':<16}{'n':>4}{'dur mean':>10}{'dur sd':>8}{'frames':>8}")
    for l in labels:
        ts = [t for t in takes if t.label == l]
        d = np.array([t.duration for t in ts])
        fr = np.array([len(t.samples) for t in ts])
        print(f"  {l:<16}{len(ts):>4}{d.mean():>9.2f}s{d.std():>7.2f}s{fr.mean():>8.1f}")


def describe_motion(M: np.ndarray, meta: dict, name: str) -> None:
    dt = np.diff(M[:, 0])
    dt = dt[dt > 0]
    rate = 1.0 / np.median(dt) if len(dt) else 0.0
    gaps = int((dt > 0.25).sum()) if len(dt) else 0
    cond = meta.get("condition", "?")
    print(f"session {name}: condition={cond!r} samples={len(M)} "
          f"duration={M[-1, 0] - M[0, 0]:.1f}s rate~{rate:.1f} Hz gaps>250ms={gaps}")


def print_eval(r: dict, title: str) -> None:
    verdict = "SIGNAL (p < 0.05)" if r["p_value"] < 0.05 else "not distinguishable from chance"
    print(f"\n{title}  features={'+'.join(r['features'])}")
    print(f"  accuracy {pct(r['accuracy'])}  chance {pct(r['chance'])}  "
          f"shuffled mean {pct(r['null_mean'])}  shuffled p95 {pct(r['null_p95'])}  "
          f"p={r['p_value']:.3f}  -> {verdict}")
    labels = r["labels"]
    w = max(6, max(len(l) for l in labels))
    print("  actual \\ predicted".ljust(w + 4) + "".join(l.rjust(w + 1) for l in labels))
    for l, row in zip(labels, r["confusion"]):
        print("  " + l.ljust(w + 2) + "".join(str(c).rjust(w + 1) for c in row))


def print_ablation(rows: list[dict]) -> None:
    print("\nFeature ablation (leave-one-out, same takes):")
    print(f"  {'features':<18}{'acc':>8}{'chance':>8}{'p95null':>9}{'p':>8}")
    for r in rows:
        print(f"  {'+'.join(r['features']):<18}{pct(r['accuracy']):>8}{pct(r['chance']):>8}"
              f"{pct(r['null_p95']):>9}{r['p_value']:>8.3f}")


# ----------------------------------------------------------------------------- selftest

def synthetic_take(freq: float, axis: int, n_frames: int = 30, noise: float = 0.01, rng=None,
                   rate: float = 25.0, label: str = "") -> Take:
    """Same construction as SubvoxTests.DecodingTests: a windowed rotation burst on one axis."""
    rng = rng or np.random.default_rng()
    S = np.zeros((n_frames, len(COLS)))
    S[:, 0] = np.arange(n_frames) / rate
    i = np.arange(n_frames)
    env = np.sin(np.pi * i / (n_frames - 1))
    v = 0.2 * env * np.sin(2 * np.pi * freq * i / n_frames)
    S[:, 4:7] = rng.uniform(-noise, noise, size=(n_frames, 3))
    S[:, 4 + axis] += v
    S[:, 12] = -1.0  # gravity z
    S[:, 13] = 1.0   # quaternion w
    return Take(label, S, "synthetic")


def selftest() -> int:
    rng = np.random.default_rng(1)
    real = [synthetic_take(2, 0, rng=rng, label="yes") for _ in range(8)] + \
           [synthetic_take(3, 2, rng=rng, label="no") for _ in range(8)]
    noise = [synthetic_take(0, 0, noise=0.05, rng=rng, label="yes") for _ in range(8)] + \
            [synthetic_take(0, 0, noise=0.05, rng=rng, label="no") for _ in range(8)]
    a = frames(synthetic_take(2, 0, noise=0).samples, ("rot",))
    stretched = frames(synthetic_take(2, 0, n_frames=40, noise=0).samples, ("rot",))
    other = frames(synthetic_take(2, 1, noise=0).samples, ("rot",))
    assert dtw(a, a) < 1e-12, "identical sequences must have zero distance"
    assert dtw(a, stretched) < dtw(a, other), "DTW must tolerate time stretch"

    sig = evaluate(real, ALL_FEATURES, permutations=200)
    nul = evaluate(noise, ALL_FEATURES, permutations=200)
    print_eval(sig, "selftest: real signal")
    print_eval(nul, "selftest: pure noise")
    assert sig["accuracy"] > 0.9 and sig["p_value"] < 0.01, "signal case failed"
    assert nul["p_value"] > 0.01, "noise case should not be significant"

    test = [synthetic_take(2, 0, rng=rng, label="yes") for _ in range(5)] + \
           [synthetic_take(3, 2, rng=rng, label="no") for _ in range(5)]
    cross = evaluate_cross(real, test, ALL_FEATURES, permutations=200)
    print_eval(cross, "selftest: cross-set")
    assert cross["accuracy"] > 0.9, "cross-set case failed"

    quiet = np.zeros((20, len(COLS)))
    quiet[:, 0] = np.arange(20) / 25
    burst = synthetic_take(2, 0, noise=0).samples.copy()
    burst[:, 0] += 0.8
    tail = quiet.copy()
    tail[:, 0] += 2
    trimmed = trim(np.concatenate([quiet, burst, tail]))
    assert 20 <= len(trimmed) < 40, f"trim kept {len(trimmed)} frames"
    print("\nselftest OK")
    return 0


# ----------------------------------------------------------------------------- main

def parse_features(s: str):
    fs = tuple(x.strip() for x in s.split(",") if x.strip())
    bad = [x for x in fs if x not in ALL_FEATURES]
    if bad:
        raise SystemExit(f"unknown features {bad}; choose from {ALL_FEATURES}")
    return fs or ALL_FEATURES


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--permutations", type=int, default=500)
    common.add_argument("--band", type=float, default=0.25, help="Sakoe-Chiba band fraction")
    common.add_argument("--features", default="accel,rot,att")
    common.add_argument("--no-ablation", action="store_true")
    common.add_argument("--kinds", default="train,segment",
                        help="events.csv kinds to treat as labelled takes")
    common.add_argument("--json", help="write all results to this file")

    p = sub.add_parser("dataset", parents=[common], help="analyse dataset.json from the Train screen")
    p.add_argument("path")
    p = sub.add_parser("session", parents=[common], help="analyse one or more session folders (pooled)")
    p.add_argument("paths", nargs="+")
    p = sub.add_parser("cross", parents=[common], help="train on first, test on second")
    p.add_argument("train")
    p.add_argument("test")
    sub.add_parser("selftest", help="run synthetic checks")
    args = ap.parse_args(argv)

    if args.cmd == "selftest":
        return selftest()

    fs = parse_features(args.features)
    kinds = tuple(k.strip() for k in args.kinds.split(","))
    out: dict = {"command": args.cmd}

    if args.cmd == "cross":
        train = load_any(args.train, kinds)
        test = load_any(args.test, kinds)
        print("train:")
        describe_takes(train)
        print("test:")
        describe_takes(test)
        r = evaluate_cross(train, test, fs, args.permutations, args.band)
        if r is None:
            raise SystemExit("need >= 2 labels in train and overlapping labels in test")
        print_eval(r, "Cross-session")
        if r["labels_only_in_test"]:
            print(f"  ignored test labels not in train: {r['labels_only_in_test']}")
        out["cross"] = r
        if not args.no_ablation:
            rows = [evaluate_cross(train, test, c, args.permutations, args.band) for c in feature_combos()]
            rows = [x for x in rows if x]
            print_ablation(rows)
            out["ablation"] = rows
    else:
        if args.cmd == "dataset":
            takes = load_dataset(args.path)
        else:
            takes = []
            for path in args.paths:
                t, M, meta = load_session(path, kinds)
                describe_motion(M, meta, Path(path).expanduser().name)
                takes += t
        if not takes:
            raise SystemExit("no labelled takes found")
        describe_takes(takes)
        r = evaluate(takes, fs, args.permutations, args.band)
        if r is None:
            raise SystemExit("need >= 2 labels with >= 2 takes each")
        print_eval(r, "Leave-one-out")
        out["loo"] = r
        if not args.no_ablation:
            rows = [evaluate(takes, c, args.permutations, args.band) for c in feature_combos()]
            rows = [x for x in rows if x]
            print_ablation(rows)
            out["ablation"] = rows

    if args.json:
        Path(args.json).expanduser().write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
