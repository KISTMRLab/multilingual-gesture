"""Convert a licensed BVH and word-aligned transcript to multilingual GestureCLR contracts.

Library units come from Algorithm 1 (``multilingual_gesture.pipeline``) over
the continuous take. Paired 2D/3D windows use an explicit orthographic XY
projection of the 3D joints. It is a reproducible substitute for
video-estimated 2D pose, not a claim to reproduce the paper data. Run once per
take; ``multigesture`` commands accept several output files at once.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import numpy as np


JOINTS = ("Hips", "Neck", "Head", "LeftShoulder", "LeftArm", "LeftForeArm",
          "LeftHand", "RightShoulder", "RightArm", "RightForeArm", "RightHand")


def _rotation(axis: str, degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    if axis == "X":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def load_bvh(path: Path, target_fps: int = 15, joints: tuple[str, ...] = JOINTS) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    nodes: list[dict] = []
    stack: list[int | None] = []
    pending: int | None = None
    motion_line = next((i for i, line in enumerate(lines) if line.strip() == "MOTION"), None)
    if motion_line is None:
        raise ValueError("BVH is missing MOTION")
    for line in lines[:motion_line]:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] in ("ROOT", "JOINT"):
            pending = len(nodes)
            nodes.append({"name": parts[1], "parent": stack[-1] if stack else None,
                          "offset": np.zeros(3), "channels": []})
        elif parts[0] == "End":
            pending = None
        elif parts[0] == "{":
            stack.append(pending)
        elif parts[0] == "}":
            stack.pop()
        elif parts[0] == "OFFSET" and stack and stack[-1] is not None:
            nodes[stack[-1]]["offset"] = np.array([float(x) for x in parts[1:4]])
        elif parts[0] == "CHANNELS" and stack and stack[-1] is not None:
            nodes[stack[-1]]["channels"] = parts[2:]
    names = [node["name"] for node in nodes]
    missing = [name for name in joints if name not in names]
    if missing:
        raise ValueError(f"BVH missing required joints: {missing}")
    frame_time = float(lines[motion_line + 2].split(":")[-1])
    if frame_time <= 0:
        raise ValueError("BVH Frame Time must be positive")
    values = np.array([[float(v) for v in line.split()] for line in lines[motion_line + 3:] if line.strip()], dtype=np.float64)
    expected = sum(len(node["channels"]) for node in nodes)
    if values.ndim != 2 or values.shape[1] != expected:
        raise ValueError(f"BVH frames must have {expected} channel values")
    output = np.empty((len(values), len(joints), 3), dtype=np.float32)
    lookup = {name: i for i, name in enumerate(joints)}
    for frame, row in enumerate(values):
        transforms = []
        cursor = 0
        for node in nodes:
            local = np.array(node["offset"], dtype=np.float64)
            rotation = np.eye(3)
            for channel in node["channels"]:
                value = row[cursor]
                cursor += 1
                if channel.endswith("position"):
                    local["XYZ".index(channel[0])] += value
                elif channel.endswith("rotation"):
                    rotation = rotation @ _rotation(channel[0], value)
                else:
                    raise ValueError(f"unsupported BVH channel {channel}")
            parent = node["parent"]
            position = local if parent is None else transforms[parent][0] + transforms[parent][1] @ local
            world_rotation = rotation if parent is None else transforms[parent][1] @ rotation
            transforms.append((position, world_rotation))
            if node["name"] in lookup:
                output[frame, lookup[node["name"]]] = position
    duration = (len(output) - 1) * frame_time
    timestamps = np.arange(0, duration + 1e-8, 1 / target_fps)
    source = np.arange(len(output)) * frame_time
    resampled = np.empty((len(timestamps), len(joints), 3), np.float32)
    for j in range(len(joints)):
        for axis in range(3):
            resampled[:, j, axis] = np.interp(timestamps, source, output[:, j, axis])
    return resampled - resampled[:, 1:2]


def textgrid_words(path: Path, tier: str = "words") -> list[dict]:
    """Read word intervals from a Praat long-format TextGrid (BEAT ships these)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"item \[\d+\]:", text)[1:] or [text]
    chosen = next((b for b in blocks if re.search(rf'name = "{re.escape(tier)}"', b)), blocks[0])
    words = []
    for xmin, xmax, label in re.findall(r'xmin = ([\d.eE+-]+)\s+xmax = ([\d.eE+-]+)\s+text = "((?:[^"]|"")*)"', chosen):
        label = label.replace('""', '"').strip()
        if label:
            words.append({"word": label, "start_seconds": float(xmin), "end_seconds": float(xmax)})
    return words


def timed_words(path: Path, fps: int, frames: int) -> list[dict]:
    if path.suffix.lower() == ".textgrid":
        rows = textgrid_words(path)
    else:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    words = rows[0]["words"] if len(rows) == 1 and "words" in rows[0] else rows
    if path.suffix.lower() == ".textgrid":
        # Clamp the final interval to the resampled motion and merge sub-frame words.
        words = [dict(w, end_seconds=min(w["end_seconds"], frames / fps)) for w in words
                 if round(w["start_seconds"] * fps) < frames]
        fixed, last_end = [], 0
        for w in words:
            s = max(round(w["start_seconds"] * fps), last_end)
            e = max(round(w["end_seconds"] * fps), s + 1)
            if e > frames:
                break
            fixed.append({"word": w["word"], "start_frame": s, "end_frame": e}); last_end = e
        words = fixed
    result = []
    for item in words:
        start = item.get("start_frame", round(float(item["start_seconds"]) * fps) if "start_seconds" in item else None)
        end = item.get("end_frame", round(float(item["end_seconds"]) * fps) if "end_seconds" in item else None)
        if start is None or end is None or not 0 <= int(start) < int(end) <= frames:
            raise ValueError("word timestamps must be ordered and inside the resampled motion")
        result.append({"word": str(item["word"]), "start_frame": int(start), "end_frame": int(end)})
    if not result or any(a["end_frame"] > b["start_frame"] for a, b in zip(result, result[1:])):
        raise ValueError("word timestamps must be non-overlapping and chronological")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bvh", type=Path, required=True)
    p.add_argument("--transcript", type=Path, required=True, help="timed JSONL words or a BEAT TextGrid")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--unit-seconds", type=float, default=3, help="paired-window length for training and wild proxy")
    p.add_argument("--take", help="take name used in IDs (default: BVH file stem)")
    p.add_argument("--speaker", default="", help="speaker label stored with every row")
    p.add_argument("--variance", default="auto", help="Algorithm 1 variance threshold: 'auto' (elbow) or number")
    p.add_argument("--closure", type=float, default=float("inf"), help="optional Algorithm 1 closure cap")
    args = p.parse_args()
    if args.fps <= 0 or args.unit_seconds <= 0:
        raise ValueError("fps and unit-seconds must be positive")
    from multilingual_gesture.pipeline import extract_unit_spans, pad_units
    take = args.take or args.bvh.stem
    motion = load_bvh(args.bvh, args.fps)
    words = timed_words(args.transcript, args.fps, len(motion))
    length = round(args.fps * args.unit_seconds)
    if len(motion) < length:
        raise ValueError("BVH is shorter than one gesture unit")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    flat = motion.reshape(len(motion), -1)
    # Library units: Algorithm 1 over the continuous take (2-3 s, minimal closure, variance elbow).
    variance = args.variance if args.variance == "auto" else float(args.variance)
    spans, info = extract_unit_spans(flat, args.fps, 2.0, 3.0, variance, args.closure)
    if not spans:
        raise ValueError("Algorithm 1 found no units; lower --variance or check the motion")
    unit_clips, unit_lengths = pad_units([flat[s:e] for s, e in spans], round(3.0 * args.fps))
    unit_ids = np.asarray([f"{take}:{s}-{e}" for s, e in spans])
    # Training pairs and the wild proxy: fixed windows with overlapping words (silent windows skipped).
    starts = [s for s in range(0, len(motion) - length + 1, length)
              if any(w["end_frame"] > s and w["start_frame"] < s + length for w in words)]
    if len(starts) < 2:
        raise ValueError("need at least two windows with overlapping transcript words")
    clips = np.stack([flat[s:s + length] for s in starts])
    projected = np.stack([motion[s:s + length, :, :2].reshape(length, -1) for s in starts])
    texts = np.asarray([" ".join(w["word"] for w in words if w["end_frame"] > s and w["start_frame"] < s + length) for s in starts])
    window_ids = np.asarray([f"{take}:{s}-{s + length}" for s in starts])
    speakers = np.asarray([args.speaker] * len(starts))
    np.savez(args.output_dir / "pairs.npz", pose2d=projected, motion3d=clips, ids=window_ids,
             lengths=np.full(len(starts), length), speakers=speakers)
    np.savez(args.output_dir / "units.npz", motion3d=unit_clips, ids=unit_ids, lengths=unit_lengths,
             speakers=np.asarray([args.speaker] * len(spans)),
             starts=np.asarray([s for s, _ in spans]), ends=np.asarray([e for _, e in spans]))
    np.savez(args.output_dir / "wild.npz", pose2d=projected, texts=texts, ids=window_ids,
             lengths=np.full(len(starts), length), speakers=speakers)
    np.save(args.output_dir / "speaker_motion.npy", flat)
    (args.output_dir / "joint_order.json").write_text(json.dumps({"joints": JOINTS, "fps": args.fps, "take": take, "speaker": args.speaker, "projection": "orthographic XY; neck centered", "algorithm1": info}, indent=2), encoding="utf-8")
    print(json.dumps({"frames": len(motion), "units": len(spans), "pairs": len(clips), "words": len(words),
                      "variance_threshold": info["variance_threshold"],
                      "note": "wild.npz is a projected proxy of the same take; use held-out speakers or real video pose for research use"}))


if __name__ == "__main__":
    main()
