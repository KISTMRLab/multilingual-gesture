import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_playback import make_playback, trim_frozen_tail
from multilingual_gesture import cli
from multilingual_gesture.model import GestureCLR
from multilingual_gesture.pipeline import (augment_2d, extract_unit_spans, multilingual_retrieve, retrieve,
                                           schedule, sixgram_spans, split_long_text, temporal_shift)
from multilingual_gesture.translate import DictTranslator, HTTPTranslator, make_translator


def gesturing_motion(seconds=40, fps=15, scale=20.0, seed=0):
    """Neck-centred centimetre-scale motion: gesture strokes separated by rests."""
    rng = np.random.default_rng(seed)
    t = np.arange(seconds * fps) / fps
    base = rng.normal(size=(1, 33)) * scale
    amp = (np.sin(2 * np.pi * t / 5.0) > -0.3).astype(float)[:, None]
    stroke = np.sin(2 * np.pi * t[:, None] * rng.uniform(.3, .6, 33) + rng.uniform(0, 6, 33))
    return (base + amp * stroke * 0.3 * scale).astype(np.float32)


def test_algorithm1_yields_units_on_centimetre_data_with_readme_thresholds():
    motion = gesturing_motion()
    for variance, closure in ((0.002, 0.3), ("auto", float("inf"))):
        spans, info = extract_unit_spans(motion, 15, 2, 3, variance, closure)
        assert spans, (variance, info)
        assert all(30 <= e - s <= 45 for s, e in spans)
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))  # disjoint, ordered


def test_algorithm1_is_scale_invariant_and_rejects_static_motion():
    motion = gesturing_motion()
    assert extract_unit_spans(motion, variance_threshold=0.002)[0] == \
        extract_unit_spans(motion / 100, variance_threshold=0.002)[0]
    static = np.repeat(motion[:1], 300, 0) + 1e-6
    spans, info = extract_unit_spans(static, variance_threshold=0.01)
    assert spans == [] and info["discarded_low_variance"] > 0


def test_extract_cli_merges_takes_with_unique_ids(tmp_path):
    a, b = tmp_path / "take_a.npy", tmp_path / "take_b.npy"
    np.save(a, gesturing_motion(seed=1)); np.save(b, gesturing_motion(seed=2))
    manifest = tmp_path / "takes.json"
    manifest.write_text(json.dumps([{"path": "take_a.npy", "speaker": "s1"}, {"path": "take_b.npy", "speaker": "s2"}]))
    cli.main(["extract-units", "--motion", str(manifest), "--output", str(tmp_path / "u.npz")])
    d = np.load(tmp_path / "u.npz")
    assert len(set(d["ids"].tolist())) == len(d["ids"]) and {"s1", "s2"} == set(d["speakers"].tolist())
    assert d["motion3d"].shape[1] == 45 and (d["lengths"] <= 45).all() and (d["lengths"] >= 30).all()


def test_noise_uses_sqrt_variance_and_shift_follows_paper():
    rng = np.random.default_rng(0)
    x = np.zeros((45, 4), np.float32)
    noisy, n, cond = augment_2d(x, rng, conditions=("noise",), variances=(0.01,))
    assert cond == "noise" and abs(float(noisy.std()) - 0.1) < 0.02
    seq = np.arange(45, dtype=np.float32)[:, None] + 1
    sources, offsets = set(), set()
    for _ in range(200):
        out = temporal_shift(seq, rng, fill="zero")
        nz = np.nonzero(out[:, 0])[0]
        assert out.shape == (45, 1) and len(nz) == 30 and np.all(np.diff(out[nz, 0]) == 1)
        offsets.add(int(nz[0])); sources.add(int(out[nz[0], 0]) - 1)
    assert min(offsets) >= 1 and max(offsets) <= 15 and len(sources) > 5
    mean_fill = temporal_shift(seq, rng, fill="mean")
    assert np.isclose(mean_fill[0, 0], seq.mean())
    clean, _, _ = augment_2d(seq, rng, conditions=("clean",))
    assert np.array_equal(clean, seq)
    drawn = {augment_2d(seq, rng)[2] for _ in range(100)}
    assert drawn == {"clean", "noise", "shift_mean", "shift_zero"}


def test_padding_mask_makes_padded_and_trimmed_units_equal():
    torch.manual_seed(0)
    model = GestureCLR(6, 6).eval()
    x = torch.randn(3, 45, 6); lengths = torch.tensor([30, 38, 45])
    padded = x.clone()
    for i, n in enumerate(lengths):
        padded[i, n:] = padded[i, n - 1]
    with torch.no_grad():
        z = model.motion3d(padded, lengths)
        for i, n in enumerate(lengths):
            assert torch.allclose(z[i], model.motion3d(x[i:i + 1, :n])[0], atol=1e-5)


def test_long_text_split_spans_idle_and_schedule():
    text = ("We started the project last spring with a very small team. " * 3).strip()
    assert len(text.split()) > 30
    chunks = split_long_text(text)
    assert len(chunks) == 3 and all(len(c.split()) <= 30 for c in chunks)
    run_on = " ".join(["word"] * 70)
    assert [len(c.split()) for c in split_long_text(run_on)] == [30, 30, 10]
    assert split_long_text("short input here") == ["short input here"]
    spans = sixgram_spans("one two three four five six seven")
    assert spans[1] == {"text": "seven", "word_start": 6, "word_end": 7, "char_start": 28, "char_end": 33}
    rules = [{"text_embedding": [1., 0.], "cluster_id": 0}]
    out = retrieve("a b c d e f g", rules, lambda _: np.array([[0., 1.]]), {0: ["u"]}, min_similarity=.5, idle_id="rest")
    assert all(g["gesture_id"] == "rest" and g["idle"] for g in out)
    timed = schedule([dict(g) for g in out], audio_seconds=4.0)
    assert timed[1]["start_seconds"] == 2.0
    words = [{"start": i * .4, "end": i * .4 + .3} for i in range(7)]
    refined = schedule([dict(g) for g in out], word_times=words)
    assert refined[1]["start_seconds"] == pytest.approx(2.4)


def test_translators_and_multilingual_path(tmp_path):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body))
            if self.path.endswith("/translate"):
                reply = {"translatedText": "show the result"}
            else:
                reply = {"choices": [{"message": {"content": "show the result"}}]}
            data = json.dumps(reply).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/v1"
        assert HTTPTranslator(url, "m", api_key_env=None).translate("결과를 보여 주세요", "ko") == "show the result"
        assert seen[-1][0] == "/v1/chat/completions" and "Korean" in seen[-1][1]["messages"][0]["content"]
        assert HTTPTranslator(url, api="libretranslate").translate("x", "ko") == "show the result"
        assert seen[-1][1]["source"] == "ko"
    finally:
        server.shutdown()
    with pytest.raises(ValueError):
        DictTranslator({}).translate("안녕", "ko")
    table = {"첫 문장입니다.": "this is the first sentence.", "둘째 문장입니다.": "this is the second sentence."}
    translator = make_translator("dict", None); translator.table = table
    rules = [{"text_embedding": [1., 0.], "cluster_id": 0}]
    long_korean = " ".join(["첫 문장입니다.", "둘째 문장입니다."] * 8)
    out = multilingual_retrieve(long_korean, "ko", translator, rules, lambda _: np.array([[1., 0.]]), {0: ["u"]})
    assert len(out["chunks"]) == 16 and out["chunks"][1]["english_text"] == "this is the second sentence."
    assert [g["tts_word_start"] for g in out["gestures"]][:3] == [0, 2, 4]
    with pytest.raises(ValueError):
        multilingual_retrieve(long_korean + " 모르는 말", "ko", translator, rules,
                              lambda _: np.array([[1., 0.]]), {0: ["u"]})
    out = multilingual_retrieve("첫 문장입니다.", "ko", translator, rules, lambda _: np.array([[1., 0.]]), {0: ["u"]})
    assert out["english_text"] == "this is the first sentence." and out["tts_text"] == "첫 문장입니다."
    assert out["gestures"][0]["tts_word_start"] == 0 and out["gestures"][0]["tts_word_end"] == 2


def test_playback_honours_lengths_blend_and_idle():
    clip = np.zeros((45, 11, 3), np.float32); clip[:30, 0, 0] = np.arange(30); clip[30:] = clip[29]
    seq = [{"gesture_id": "u", "blend_frames": 5}, {"gesture_id": "idle", "idle": True, "duration_seconds": 1.0}]
    play = make_playback(seq, {"u": clip}, lengths={"u": 30})
    assert len(play["slots"][0]["frames"]) == 30 and play["slots"][0]["blend_frames"] == 5
    assert play["slots"][1]["idle"] and len(play["slots"][1]["frames"]) == 15 and play["blend_frames"] == 5
    assert len(trim_frozen_tail(clip)) == 30


def test_train_and_mine_cli_with_validation_masks_and_multiple_files(tmp_path):
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import BoW, Dense, Normalize
    rng = np.random.default_rng(0)
    pose = rng.normal(size=(12, 45, 4)).astype("float32") * 20
    motion = np.concatenate([pose, pose[..., :2]], -1)
    lengths = np.asarray([30, 45] * 6)
    np.savez(tmp_path / "p1.npz", pose2d=pose[:6], motion3d=motion[:6], lengths=lengths[:6])
    np.savez(tmp_path / "p2.npz", pose2d=pose[6:], motion3d=motion[6:], lengths=lengths[6:])
    cli.main(["train", "--pairs", str(tmp_path / "p1.npz"), str(tmp_path / "p2.npz"), "--output", str(tmp_path / "g.pt"),
              "--epochs", "3", "--batch-size", "4", "--val-fraction", ".25", "--log-every", "0"])
    ck = torch.load(tmp_path / "g.pt", weights_only=True)
    assert ck["val_pairs"] == 3 and ck["normalize"] and "val_loss" in ck["metrics"] and ck["preset"] == "demo"
    np.savez(tmp_path / "u1.npz", motion3d=motion[:5], ids=np.asarray([f"a:{i}" for i in range(5)]), lengths=lengths[:5])
    np.savez(tmp_path / "u2.npz", motion3d=motion[5:], ids=np.asarray([f"b:{i}" for i in range(7)]), lengths=lengths[5:])
    texts = np.asarray(["open both hands", "point at the chart", "move on now"])
    np.savez(tmp_path / "w.npz", pose2d=pose[:3], texts=texts)
    torch.manual_seed(0)
    vocab = sorted({w for t in texts for w in t.split()})
    SentenceTransformer(modules=[BoW(vocab), Dense(len(vocab), 16), Normalize()]).save_pretrained(str(tmp_path / "sbert"))
    cli.main(["mine", "--wild", str(tmp_path / "w.npz"), "--units", str(tmp_path / "u1.npz"), str(tmp_path / "u2.npz"),
              "--checkpoint", str(tmp_path / "g.pt"), "--output-prefix", str(tmp_path / "lib"), "--clusters", "3",
              "--sbert", str(tmp_path / "sbert")])
    clusters = np.load(tmp_path / "lib.clusters.npz")
    assert len(clusters["ids"]) == 12 and "lengths" in clusters.files
    (tmp_path / "t.json").write_text(json.dumps({"손을 펼쳐 주세요": "open both hands"}), encoding="utf-8")
    cli.main(["retrieve", "--rules", str(tmp_path / "lib.rules.jsonl"), "--clusters", str(tmp_path / "lib.clusters.npz"),
              "--text", "손을 펼쳐 주세요", "--source-language", "ko", "--translations", str(tmp_path / "t.json"),
              "--sbert", str(tmp_path / "sbert"), "--min-similarity", "-1", "--audio-seconds", "2",
              "--output", str(tmp_path / "seq.json")])
    seq = json.loads((tmp_path / "seq.json").read_text(encoding="utf-8"))
    assert seq["english_text"] == "open both hands" and seq["tts_text"] == "손을 펼쳐 주세요"
    assert seq["gestures"][0]["duration_seconds"] == 2.0 and not seq["gestures"][0]["idle"]
