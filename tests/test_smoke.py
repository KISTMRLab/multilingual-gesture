import numpy as np, pytest
from multilingual_gesture.pipeline import extract_units,require_english,sixgrams
def test_contracts():
    motion=np.sin(np.linspace(0,6,90))[:,None].astype(np.float32); units=extract_units(motion,15,2,3,0.01,2.0); assert units and 30<=len(units[0])<=45
    assert require_english("hello","en",{})=="hello"
    with pytest.raises(ValueError): require_english("안녕","ko",{})
    assert len(sixgrams("a b c d e f g"))==2

