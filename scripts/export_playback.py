"""Join retrieval IDs to actual motion clips for the browser viewer."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def trim_frozen_tail(clip: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    """Drop trailing frames identical to the last one (edge padding)."""
    flat = clip.reshape(len(clip), -1)
    end = len(flat)
    while end > 1 and np.abs(flat[end - 2] - flat[end - 1]).max() <= tolerance:
        end -= 1
    return clip[:end]


def _as_frames(clip) -> np.ndarray:
    clip = np.asarray(clip, np.float32)
    if clip.ndim == 2:
        if clip.shape[-1] % 3:
            raise ValueError("flattened motion width must be divisible by 3")
        clip = clip.reshape(len(clip), -1, 3)
    if clip.ndim != 3 or clip.shape[-1] not in (2, 3) or not np.isfinite(clip).all():
        raise ValueError("motion clips must be finite [F,J,2|3] arrays")
    if clip.shape[-1] == 2:
        clip = np.pad(clip, ((0, 0), (0, 0), (0, 1)))
    return clip


def make_playback(sequence: list[dict] | dict, library: dict[str, np.ndarray], fps: int = 15,
                  lengths: dict[str, int] | None = None, idle_pose: np.ndarray | None = None) -> dict:
    """Build viewer slots. Unit ``lengths`` trim padded frames (frozen tails are
    also trimmed when lengths are unknown); each slot carries its
    ``blend_frames`` hint. Idle slots hold a neutral pose for their duration."""
    slots = sequence.get("gestures", []) if isinstance(sequence, dict) else sequence
    if fps <= 0 or not slots:
        raise ValueError("playback requires a positive FPS and non-empty retrieval sequence")
    lengths = lengths or {}
    if idle_pose is None and library:
        firsts = [_as_frames(v)[0] for v in library.values()]
        idle_pose = np.median(np.stack(firsts), axis=0)
    output = []
    for slot in slots:
        gid = str(slot["gesture_id"])
        if gid not in library and slot.get("idle"):
            if idle_pose is None:
                raise KeyError("idle slot needs a library or idle pose")
            count = max(int(round(float(slot.get("duration_seconds", 2.0)) * fps)), 1)
            clip = np.repeat(np.asarray(idle_pose, np.float32)[None], count, axis=0)
        else:
            if gid not in library:
                raise KeyError(f"retrieved gesture {gid!r} is absent from the motion library")
            clip = _as_frames(library[gid])
            clip = clip[:int(lengths[gid])] if gid in lengths else trim_frozen_tail(clip)
        output.append({"gesture_id": gid, "frames": clip.tolist(),
                       "text": slot.get("text", slot.get("english_text", "")),
                       "similarity": slot.get("similarity"),
                       "source": slot.get("source"),
                       "cluster_id": slot.get("cluster_id"),
                       "idle": bool(slot.get("idle", False)),
                       "blend_frames": int(slot.get("blend_frames", 5)),
                       "start_seconds": slot.get("start_seconds"),
                       "duration_seconds": slot.get("duration_seconds", len(clip) / fps)})
    canonical = ["Hips", "Neck", "Head", "LeftShoulder", "LeftArm",
                 "LeftForeArm", "LeftHand", "RightShoulder", "RightArm", "RightForeArm", "RightHand"]
    count = len(output[0]["frames"][0])
    return {"fps": fps, "joint_order": canonical if count == len(canonical) else [f"joint_{i}" for i in range(count)],
            "blend_frames": max(s["blend_frames"] for s in output), "slots": output}


def load_library(path: Path):
    """Return ``(library, lengths)`` from a units/pairs/bank NPZ."""
    data = np.load(path, allow_pickle=False)
    key = "motion3d" if "motion3d" in data else "motion" if "motion" in data else None
    if key is None:
        return {k: data[k] for k in data.files}, {}
    ids = [str(gid) for gid in data["ids"]]
    library = dict(zip(ids, data[key]))
    lengths = dict(zip(ids, (int(n) for n in data["lengths"]))) if "lengths" in data else {}
    return library, lengths


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", type=Path, required=True)
    p.add_argument("--motion", type=Path, required=True, help="NPZ bank, units or training pairs")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fps", type=int, default=15)
    a = p.parse_args()
    library, lengths = load_library(a.motion)
    sequence = json.loads(a.sequence.read_text(encoding="utf-8"))
    playback = make_playback(sequence, library, a.fps, lengths)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(playback), encoding="utf-8")
    print(json.dumps({"slots": len(playback["slots"]), "output": str(a.output)}))


if __name__ == "__main__":
    main()
