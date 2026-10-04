"""Exercise GestureCLR, unit extraction, multilingual handoff, and retrieval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import BoW, Dense, Normalize

from multilingual_gesture.model import GestureCLR, ntxent
from multilingual_gesture.pipeline import bisect, extract_units, require_english, retrieve


def make_text_encoder(path: Path, texts: list[str]) -> None:
    torch.manual_seed(11)
    vocab = sorted({token for text in texts for token in text.lower().split()})
    SentenceTransformer(modules=[BoW(vocab), Dense(len(vocab), 384), Normalize()]).save_pretrained(str(path))


def run_cli(*args: object) -> None:
    subprocess.run([sys.executable, "-m", "multilingual_gesture.cli", *map(str, args)], check=True)


def text_embedding(texts: list[str], width: int = 384) -> np.ndarray:
    rows = []
    for text in texts:
        row = np.zeros(width, np.float32)
        for token in text.lower().split(): row[sum(token.encode("utf-8")) % width] += 1
        row /= max(np.linalg.norm(row), 1e-8); rows.append(row)
    return np.stack(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/smoke"))
    args = parser.parse_args(); out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(11); rng = np.random.default_rng(11)
    pose2d = rng.normal(size=(6, 45, 8)).astype("float32")
    motion3d = np.concatenate((pose2d, pose2d[..., :4] * 0.5), axis=-1)
    np.savez(out / "pairs.npz", pose2d=pose2d, motion3d=motion3d)
    model = GestureCLR(8, 12); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train(); z2, z3 = model(torch.from_numpy(pose2d), torch.from_numpy(motion3d)); loss = ntxent(z2, z3)
    optimizer.zero_grad(); loss.backward(); optimizer.step(); torch.save({"state": model.state_dict(), "d2": 8, "d3": 12}, out / "gestureclr.pt")

    continuous = np.concatenate((motion3d[0], motion3d[1]), axis=0)
    units = extract_units(continuous, fps=15, variance_threshold=0.0, closure_threshold=float("inf"))
    if not units: raise RuntimeError("unit extraction produced no clips")
    checkpoint = torch.load(out / "gestureclr.pt", map_location="cpu", weights_only=True)
    model = GestureCLR(checkpoint["d2"], checkpoint["d3"]); model.load_state_dict(checkpoint["state"]); model.eval()
    with torch.no_grad(): latents = model.motion3d(torch.from_numpy(motion3d)).numpy()
    labels, centers = bisect(latents, 3, seed=11)
    ids = np.asarray([f"unit_{i}" for i in range(len(latents))])
    np.savez(out / "library.clusters.npz", ids=ids, labels=labels, centroids=centers)
    texts = ["welcome to the demonstration", "show the important result", "move to the next topic"]
    rules = [{"text": t, "text_embedding": e.tolist(), "cluster_id": int(labels[i])} for i, (t, e) in enumerate(zip(texts, text_embedding(texts)))]
    (out / "library.rules.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rules), encoding="utf-8")
    translations = {"결과를 보여 주세요": "show the important result"}
    english = require_english("결과를 보여 주세요", "ko", translations)
    clusters = {int(k): [str(x) for x in ids[labels == k]] for k in np.unique(labels)}
    gestures = retrieve(english, rules, text_embedding, clusters, seed=11)
    result = {"source_text": "결과를 보여 주세요", "english_text": english, "gestures": gestures}
    (out / "sequence.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    np.save(out / "speaker_motion.npy", continuous)
    np.savez(out / "wild.npz", pose2d=pose2d[:3], texts=np.asarray(texts))
    (out / "translations.json").write_text(json.dumps(translations, ensure_ascii=False), encoding="utf-8")
    make_text_encoder(out / "tiny-sbert", texts + [english])
    run_cli("extract-units", "--motion", out / "speaker_motion.npy", "--variance", 0.0, "--closure", 1000.0, "--output", out / "cli-units.npz")
    run_cli("train", "--pairs", out / "pairs.npz", "--output", out / "cli-gestureclr.pt", "--epochs", 1, "--batch-size", 6, "--seed", 11)
    run_cli("mine", "--wild", out / "wild.npz", "--units", out / "cli-units.npz", "--checkpoint", out / "cli-gestureclr.pt", "--output-prefix", out / "cli-library", "--clusters", 2, "--sbert", out / "tiny-sbert")
    run_cli("retrieve", "--rules", out / "cli-library.rules.jsonl", "--clusters", out / "cli-library.clusters.npz", "--source-language", "ko", "--translations", out / "translations.json", "--text", "결과를 보여 주세요", "--sbert", out / "tiny-sbert", "--seed", 11, "--output", out / "cli-sequence.json")
    cli_result = json.loads((out / "cli-sequence.json").read_text(encoding="utf-8"))
    if not cli_result["gestures"]: raise RuntimeError("installed CLI produced no gestures")
    print(json.dumps({"loss": float(loss.detach()), "extracted_units": len(units), "gestures": len(gestures), "output": str(out)}))


if __name__ == "__main__": main()
