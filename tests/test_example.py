import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from example_demo import query


def test_authored_example_translates_then_retrieves():
    result=query("multilingual","결과를 보여 주세요",{"language":["ko"]})
    assert result["english_text"]=="point to the result"
    assert result["slots"][0]["gesture_id"]=="point_right"
    assert "no trained weights" in result["data_label"]
