import numpy as np, pytest
from multilingual_gesture.pipeline import extract_units,require_english,sixgrams
from multilingual_gesture.pipeline import retrieve
def test_contracts():
    motion=np.sin(np.linspace(0,6,90))[:,None].astype(np.float32); units=extract_units(motion,15,2,3,0.01,2.0); assert units and 30<=len(units[0])<=45
    assert require_english("hello","en",{})=="hello"
    with pytest.raises(ValueError): require_english("안녕","ko",{})
    assert len(sixgrams("a b c d e f g"))==2

def test_translation_and_cluster_route():
    english=require_english("안녕하세요","ko",{"안녕하세요":"hello everyone"})
    rules=[{"text_embedding":[1.,0.],"cluster_id":2}]
    out=retrieve(english,rules,lambda _:np.array([[1.,0.]],np.float32),{2:["wave"]},seed=1)
    assert out[0]["gesture_id"]=="wave" and out[0]["blend_frames"]==5
