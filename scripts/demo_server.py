"""Local query-driven viewer backed by this repository's retrieval pipeline."""
from __future__ import annotations
import argparse
import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import numpy as np
from export_playback import make_playback
from beat_runtime import serve_beat, library as beat_library, query_application
from speech_backend import SpeechBackend, speech_route
import paper_method_common as pm

MODE = "multilingual"
PACKAGE = "multilingual_gesture"
SRC = Path(__file__).resolve().parents[1] / "src"


def use_repository_package():
    """Prepared mode runs this repository's ``src`` package, not the copy that
    ``beat_runtime`` vendors for the no-data stand-in (``scripts/beat_deps``)."""
    if SRC.is_dir():
        sys.path.insert(0, str(SRC))
        for name in [n for n in sys.modules if n == PACKAGE or n.startswith(PACKAGE + ".")]:
            del sys.modules[name]


def make_server(a):
    """Build the HTTP server; ``serve`` runs it (tests use port 0)."""
    root = Path(__file__).resolve().parents[1] / "static"
    speech = SpeechBackend()
    prepared = None
    if a.prepared:
        # BEAT paper-method artifacts from scripts/prepare_paper_method.py
        from paper_method_demo import PreparedDemo
        prepared = PreparedDemo(a.prepared, a)
        query = prepared.query
    elif a.example:
        from example_demo import query as example_query
        def query(text, params):
            if beat_library(root.parent, MODE)['ready']:
                return query_application(root.parent, MODE, text, params)
            return example_query(MODE, text, params)
    elif MODE == "automatic":
        from automatic_text_to_gesture.core import TOKEN, load_glove, mine_rules, retrieve
        if not a.glove:
            raise ValueError("automatic demo needs --glove")
        video = np.load(a.data_dir / "video.npz")
        bank_npz = np.load(a.data_dir / "bank.npz")
        library = {key: bank_npz[key] for key in bank_npz.files}
        words = json.loads(str(video["words_json"]))
        vectors = {}
        def query(text, params):
            threshold = float(params.get("threshold", ["0.92"])[0])
            rules = mine_rules(video["pose"], words, library, threshold, a.seed)
            if not rules:
                raise ValueError("no rules passed this pose-cosine threshold")
            required = {w for rule in rules for w in TOKEN.findall(rule.phrase.lower())}
            required.update(TOKEN.findall(text.lower()))
            missing = required - vectors.keys()
            if missing:
                vectors.update(load_glove(a.glove, missing))
            sequence = retrieve(text, rules, vectors, audio_seconds=float(params.get("duration", ["3"])[0]))
            result = make_playback(sequence, library)
            result.update({"algorithm": "frame-cosine rule mining + summed GloVe retrieval",
                           "threshold": threshold, "rule_count": len(rules),
                           "trace": sequence, "data_label": a.data_label})
            return result
    elif MODE == "wild":
        from sentence_transformers import SentenceTransformer
        from wild_pose_matching.pipeline import read_jsonl, retrieve
        rules = read_jsonl(a.rules)
        d = np.load(a.clusters)
        library_npz = np.load(a.data_dir / "units.npz")
        library = {str(k): v for k, v in zip(library_npz["ids"], library_npz["motion3d"])}
        groups = {int(k): [str(x) for x in d["ids"][d["labels"] == k]] for k in np.unique(d["labels"])}
        encoder = SentenceTransformer(pm.find_sbert(a.sbert)[0] or "all-MiniLM-L6-v2")
        def query(text, params):
            seed = int(params.get("seed", [str(a.seed)])[0])
            sequence = retrieve(text, rules, lambda x: encoder.encode(x, normalize_embeddings=True), groups, seed)
            result = make_playback(sequence, library)
            result.update({"algorithm": "GestureCLR pose-unit matching + Bisecting K-Means + Sentence-BERT",
                           "trace": sequence, "cluster_count": len(groups), "seed": seed, "data_label": a.data_label})
            return result
    elif MODE == "multilingual":
        from sentence_transformers import SentenceTransformer
        from multilingual_gesture.pipeline import multilingual_retrieve, schedule
        from multilingual_gesture.translate import translator_from_args
        from export_playback import load_library
        rules = [json.loads(x) for x in Path(a.rules).read_text(encoding="utf-8").splitlines() if x]
        d = np.load(a.clusters)
        library, lengths = {}, {}
        for units_path in a.units or [a.data_dir / "units.npz"]:
            part, part_lengths = load_library(Path(units_path))
            library.update(part); lengths.update(part_lengths)
        groups = {int(k): [str(x) for x in d["ids"][d["labels"] == k]] for k in np.unique(d["labels"])}
        encoder = SentenceTransformer(pm.find_sbert(a.sbert)[0] or "all-MiniLM-L6-v2")
        translator = translator_from_args(a)
        def query(text, params):
            language = params.get("language", ["en"])[0]
            seed = int(params.get("seed", [str(a.seed)])[0])
            floor = params.get("min_similarity", [None])[0]
            floor = a.min_similarity if floor in (None, "") else float(floor)
            out = multilingual_retrieve(text, language, translator, rules,
                                        lambda x: encoder.encode(x, normalize_embeddings=True), groups, seed,
                                        floor, a.idle_id, a.max_words, params.get("tts_language", [None])[0])
            sequence = out["gestures"]
            duration = params.get("duration", [None])[0]
            schedule(sequence, float(duration) if duration else None)
            result = make_playback({"gestures": sequence}, library, lengths=lengths)
            result.update({"algorithm": f"{a.translator} translation + GestureCLR clusters + Sentence-BERT",
                           "source_text": text, "english_text": out["english_text"], "source_language": language,
                           "tts_text": out["tts_text"], "tts_language": out["tts_language"],
                           "chunks": [{k: c[k] for k in ("index", "source_text", "english_text", "tts_text")} for c in out["chunks"]],
                           "min_similarity": floor, "idle_count": sum(s["idle"] for s in sequence),
                           "trace": sequence, "cluster_count": len(groups), "seed": seed, "data_label": a.data_label})
            return result
    else:
        import torch
        from sentence_transformers import SentenceTransformer
        from ridge_gesture.model import TextMotionModel
        from ridge_gesture.pipeline import hybrid_retrieve
        rules = [json.loads(x) for x in Path(a.rules).read_text(encoding="utf-8").splitlines() if x]
        ck = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
        model = TextMotionModel(ck["text_dim"], ck["motion_dim"])
        model.load_state_dict(ck["state"]); model.eval()
        encoder = SentenceTransformer(pm.find_sbert(a.sbert)[0] or "all-MiniLM-L6-v2")
        library_npz = np.load(a.data_dir / "train_pairs.npz")
        library = {str(k): v for k, v in zip(library_npz["ids"], library_npz["motion"])}
        latent = ck["motion_latents"].cpu().numpy().astype("float32")
        ids = [str(x) for x in ck["ids"]]
        def fallback(text):
            with torch.no_grad():
                z = model.text(torch.from_numpy(encoder.encode([text], normalize_embeddings=True).astype("float32"))).numpy()[0]
            return z / max(np.linalg.norm(z), 1e-8)
        def query(text, params):
            threshold = float(params.get("threshold", ["0.72"])[0])
            sequence = hybrid_retrieve(text, rules, lambda x: encoder.encode(x, normalize_embeddings=True),
                                       threshold, latent, ids, fallback)
            result = make_playback(sequence, library)
            result.update({"algorithm": "threshold-gated Sentence-BERT rules + contrastive fallback",
                           "trace": sequence, "threshold": threshold,
                           "rule_count": sum(s["source"] == "rule" for s in sequence),
                           "fallback_count": sum(s["source"] == "fallback" for s in sequence),
                           "data_label": a.data_label})
            return result

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)
        def do_GET(self):
            if prepared is not None and pm.handle(self, prepared): return
            if serve_beat(self, root.parent, MODE): return
            parsed = urlparse(self.path)
            if parsed.path == "/api/query":
                params = parse_qs(parsed.query)
                try:
                    text = params.get("text", [""])[0].strip()
                    if not text:
                        raise ValueError("enter query text")
                    body, code = json.dumps(query(text, params), ensure_ascii=False).encode(), 200
                except (ValueError, KeyError, IndexError) as exc:
                    body, code = json.dumps({"error": str(exc)}).encode(), 400
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            return super().do_GET()
        def do_POST(self):
            if prepared is not None and pm.handle(self, prepared): return
            if serve_beat(self, root.parent, MODE): return
            if speech_route(self, speech):
                return
            self.send_error(404)
    return ThreadingHTTPServer((a.host, a.port), Handler)


def serve(a):
    server = make_server(a)
    print(f"Demo: http://{a.host}:{server.server_port}/")
    server.serve_forever()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--example", action="store_true", help="run authored motion with illustrative vectors; no weights")
    p.add_argument("--prepared", type=Path, help="folder written by scripts/prepare_paper_method.py (paper method on public BEAT)")
    p.add_argument("--rules", type=Path)
    p.add_argument("--clusters", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--glove", type=Path)
    p.add_argument("--sbert", help="Sentence-BERT name or directory (default all-MiniLM-L6-v2; prepared mode: the model recorded by prepare_paper_method.py)")
    if "--example" not in sys.argv[1:]:
        use_repository_package()
    try:
        from multilingual_gesture.translate import add_translator_arguments
    except ImportError:  # example mode with an older vendored package: translators unused
        add_translator_arguments = None
    if add_translator_arguments:
        add_translator_arguments(p)
    p.add_argument("--units", nargs="+", type=Path, help="unit NPZ files (default: <data-dir>/units.npz)")
    p.add_argument("--min-similarity", type=float, help="play --idle-id when the best rule is below this")
    p.add_argument("--idle-id", default="idle")
    p.add_argument("--max-words", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-label", default="User-prepared motion")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args()
    if not a.example and not a.prepared and a.data_dir is None:
        p.error("--data-dir is required for prepared-data mode")
    if not a.example and not a.prepared and MODE in ("wild", "multilingual") and (not a.rules or not a.clusters):
        p.error("--rules and --clusters are required")
    if not a.example and not a.prepared and MODE == "ridge" and (not a.rules or not a.checkpoint):
        p.error("--rules and --checkpoint are required")
    serve(a)


if __name__ == "__main__":
    main()
