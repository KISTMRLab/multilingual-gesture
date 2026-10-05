"""Prepared demo mode: serve the multilingual rule map mined from public BEAT to the browser viewer.

``scripts/demo_server.py --prepared outputs/paper-method/<key>`` loads the
artifacts written by ``prepare_paper_method.py`` and answers ``/api/beat-library``,
``/api/beat-query`` and ``/api/query`` with the repository's own runtime:
>30-word sentence split, translation to English, six-word Sentence-BERT lookup
and random choice inside the matched GestureCLR cluster.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import paper_method_common as pm

ALGORITHM = ("Multilingual Gesture: Algorithm 1 units from library speakers, GestureCLR with paper augmentation, "
             "rules mined from held-out projected+corrupted wild speakers, translation to English, "
             "six-word Sentence-BERT retrieval and Bisecting K-Means cluster sampling")
DATA_LABEL = "Public BEAT, disjoint speakers; projected BEAT motion stands in for wild video (local demo-scale fit)"
TRANSLATIONS = pm.ROOT / "examples" / "beat-translations.json"


class PreparedDemo:
    def __init__(self, folder, args=None, encoder=None, translator=None):
        pm.use_repository_package("multilingual_gesture")
        from multilingual_gesture.translate import make_translator, translator_from_args
        self.folder, self.manifest = pm.load_manifest(folder)
        files = self.manifest["files"]
        units = np.load(pm.resolve(files["units"]))
        self.motion = {str(i): m for i, m in zip(units["ids"], units["motion3d"])}
        self.lengths = {str(i): int(n) for i, n in zip(units["ids"], units["lengths"])}
        self.info = {u["id"]: u for u in json.loads((self.folder / files["unit_info"]).read_text(encoding="utf-8"))}
        self.rules = [json.loads(x) for x in (self.folder / files["rules"]).read_text(encoding="utf-8").splitlines() if x.strip()]
        clusters = np.load(self.folder / files["clusters"])
        self.groups = {int(k): [str(x) for x in clusters["ids"][clusters["labels"] == k]] for k in np.unique(clusters["labels"])}
        self.rest = np.median(np.stack([m.reshape(len(m), -1, 3)[0] for m in self.motion.values()]), axis=0)
        if translator is None:
            if args is not None and getattr(args, "translator", "dict") != "dict":
                translator = translator_from_args(args)
            else:
                table = getattr(args, "translations", None) or (TRANSLATIONS if TRANSLATIONS.is_file() else None)
                translator = make_translator("dict", table)
        self.translator = translator
        self.translator_name = getattr(args, "translator", None) or "dict"
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(str(getattr(args, "sbert", None) or self.manifest["sbert"]))
            encoder = lambda texts: model.encode(list(texts), normalize_embeddings=True)
        self.encode = encoder

    def library(self):
        clips = [{"id": gid, "text": row.get("text", ""), "duration": self.lengths[gid] / pm.FPS,
                  "source": {k: row.get(k) for k in ("speaker", "take", "start_frame", "end_frame", "role", "kind",
                                                     "motion_url", "alignment_url")}}
                 for gid, row in self.info.items()]
        korean = list(json.loads(TRANSLATIONS.read_text(encoding="utf-8")))[:3] if TRANSLATIONS.is_file() else []
        suggested = korean + list(self.manifest["suggested_queries"]) + list(self.manifest["heldout_probes"])
        return {"ready": True, "prepared": True, "mode": "multilingual", "clips": clips, "suggested_queries": suggested,
                "heldout_probes": self.manifest["heldout_probes"], "metrics": self.manifest["metrics"],
                "roles": self.manifest["roles"]["roles"], "translator": self.translator_name,
                "algorithm": ALGORITHM, "data_label": DATA_LABEL}

    def query(self, text, params):
        from multilingual_gesture.pipeline import multilingual_retrieve
        language = pm.param(params, "source_language", pm.param(params, "language", "en"))
        seed = pm.param(params, "seed", self.manifest["seed"], int)
        floor = pm.param(params, "min_similarity", self.manifest["min_similarity"], float)
        out = multilingual_retrieve(text, language, self.translator, self.rules, self.encode, self.groups, seed,
                                    floor, "idle", 30, pm.param(params, "tts_language"))
        slots = []
        for g in out["gestures"]:
            if g["idle"]:
                slots.append(pm.idle_slot(g["english_text"], self.rest, "below similarity floor", floor=floor,
                                          similarity=round(g["similarity"], 5), english_text=g["english_text"]))
                continue
            gid = g["gesture_id"]
            row = self.info.get(gid, {})
            slots.append({"gesture_id": gid, "text": g["english_text"], "english_text": g["english_text"],
                          "frames": pm.frames_m(self.motion[gid], self.lengths[gid]),
                          "route": "learned_pose_rule", "confidence": round(g["similarity"], 5),
                          "similarity": round(g["similarity"], 5), "cluster_id": g["cluster_id"],
                          "source": {k: row.get(k) for k in ("speaker", "take", "start_frame", "end_frame", "kind",
                                                             "motion_url", "alignment_url")},
                          "rule_source": {"cluster_id": g["cluster_id"], "chunk_index": g["chunk_index"]},
                          "blend_frames": g["blend_frames"]})
        metrics = {k: self.manifest["metrics"].get(k) for k in ("rules", "library_units", "clusters",
                                                                 "heldout_cross_view_top1", "heldout_chance")}
        metrics.update(rule_count=len(self.rules), min_similarity=floor,
                       learned_rule_usage=sum(s["route"] == "learned_pose_rule" for s in slots))
        translated = out["english_text"] != text
        return pm.query_result(slots, algorithm=ALGORITHM, data_label=DATA_LABEL, metrics=metrics,
                               trace={"input": text, "retrieval_text": out["english_text"], "seed": seed,
                                      "translation": self.translator_name if translated else None,
                                      "chunks": [{k: c[k] for k in ("index", "source_text", "english_text", "tts_text")}
                                                 for c in out["chunks"]]},
                               joints=pm.ingest().UPPER_BODY,
                               extra={"english_text": out["english_text"], "source_language": language,
                                      "tts_text": out["tts_text"], "tts_language": out["tts_language"],
                                      "rule_count": len(self.rules), "cluster_count": len(self.groups), "seed": seed})
