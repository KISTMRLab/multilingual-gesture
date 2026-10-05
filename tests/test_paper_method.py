"""Paper-method preparation hook and prepared demo endpoints on tiny synthetic BEAT fixtures."""
import argparse
import contextlib
import io
import json
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import beat_fixture  # noqa: E402
import paper_method_common as pm  # noqa: E402
import prepare_paper_method as prep  # noqa: E402

QUICK = ["--epochs", "2", "--batch-size", "8", "--max-takes-per-speaker", "1"]


def run(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = prep.main(argv)
    return code, json.loads(out.getvalue().strip().splitlines()[-1])


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    base = tmp_path_factory.mktemp("multilingual")
    source = beat_fixture.make_processed(base / "processed")
    sbert = beat_fixture.make_sbert(base / "sbert")
    argv = ["--processed", str(source), "--sbert", str(sbert), "--output-root", str(base / "out"), *QUICK]
    code, result = run(argv)
    return {"code": code, "result": result, "argv": argv, "sbert": sbert}


def test_processed_route_runs_the_cli_pipeline_with_disjoint_roles(prepared):
    assert prepared["code"] == 0 and prepared["result"]["ready"] is True, prepared["result"]
    args = prepared["result"]["server_args"]
    assert args[:2] == ["scripts/demo_server.py", "--prepared"]
    folder = Path(args[2])
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    roles = manifest["roles"]["roles"]
    assert manifest["roles"]["unit"] == "speaker"
    assert not set(roles["library"]) & set(roles["train"]) and not set(roles["wild"]) & set(roles["library"] + roles["train"])
    metrics = manifest["metrics"]
    assert metrics["rules"] == manifest["export"]["wild_windows"] > 0
    assert 0 <= metrics["heldout_cross_view_top1"] <= 1
    assert metrics["heldout_chance"] == pytest.approx(1 / metrics["heldout_windows"], abs=1e-4)
    units = np.load(pm.resolve(manifest["files"]["units"]))
    assert len(units["ids"]) == metrics["library_units"] and (units["lengths"] >= 30).all()  # Algorithm 1: 2-3 s units
    rules = [json.loads(x) for x in (folder / "mined.rules.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["source_gesture_id"] for r in rules} <= {str(i) for i in units["ids"]}
    assert set(r["wild_speaker"] for r in rules) <= set(roles["wild"]) or all(r["wild_speaker"] == "" for r in rules)


def test_cached_result_is_reused(prepared):
    code, result = run(prepared["argv"])
    assert code == 0 and result["ready"] and result["summary"]["cached"] is True


def test_raw_bvh_textgrid_route(tmp_path, prepared):
    raw = beat_fixture.make_raw(tmp_path / "beat_english_v0.2.1")
    code, result = run(["--beat-root", str(raw), "--sbert", str(prepared["sbert"]), "--output-root", str(tmp_path / "out"), *QUICK])
    assert code == 0 and result["ready"], result
    manifest = json.loads((Path(result["server_args"][2]) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"]["kind"] == "raw" and manifest["metrics"]["rules"] > 0


def test_missing_source_or_sbert_is_not_ready(tmp_path, monkeypatch):
    for name in (pm.ENV_PROCESSED, pm.ENV_RAW, pm.ENV_SBERT):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(pm, "ROOT", tmp_path)
    code, result = run(["--output-root", str(tmp_path / "out")])
    assert code == 0 and result["ready"] is False and "BEAT" in result["reason"]
    monkeypatch.setattr(pm, "DEFAULT_SBERT_DIR", tmp_path / "missing")
    source = beat_fixture.make_processed(tmp_path / "p", speakers=("1", "2", "3"), seconds=6)
    code, result = run(["--processed", str(source), "--output-root", str(tmp_path / "out")])
    assert result["ready"] is False and "save" in result["next_steps"][0].lower()


@pytest.fixture(scope="module")
def server(prepared):
    import demo_server
    args = argparse.Namespace(prepared=Path(prepared["result"]["server_args"][2]), example=False, sbert=None,
                              host="127.0.0.1", port=0)
    httpd = demo_server.make_server(args)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def get(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read())


def test_prepared_server_library_and_queries(server):
    library = get(server + "/api/beat-library")
    assert library["ready"] and library["prepared"] and library["clips"]
    korean = library["suggested_queries"][0]
    english = [q for q in library["suggested_queries"] if q.isascii()][0]
    for path in ("/api/beat-query", "/api/query"):
        result = get(server + path + "?" + urllib.parse.urlencode({"text": english, "language": "en"}))
        slot = result["slots"][0]
        assert slot["route"] == "learned_pose_rule" and len(slot["frames"][0]) == 11
        assert slot["gesture_id"] in {c["id"] for c in library["clips"]}
    translated = get(server + "/api/beat-query?" + urllib.parse.urlencode({"text": korean, "language": "ko"}))
    table = json.loads((ROOT / "examples" / "beat-translations.json").read_text(encoding="utf-8"))
    assert translated["english_text"] == table[korean] and translated["trace"]["translation"] == "dict"
    assert translated["tts_text"] == korean
    request = urllib.request.Request(server + "/api/beat-query",
                                     json.dumps({"text": "zzzz qqqq", "min_similarity": 0.9999}).encode(),
                                     {"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        idle = json.loads(response.read())
    assert idle["no_match"] is True and idle["slots"][0]["route"] == "idle_no_match"
    with pytest.raises(urllib.error.HTTPError):  # untranslated non-English text is refused, as in the paper runtime
        get(server + "/api/beat-query?" + urllib.parse.urlencode({"text": "번역 없는 문장", "language": "ko"}))
