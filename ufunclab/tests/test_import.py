import pytest
import ufunclab


def test_bad_import():
    with pytest.raises(AttributeError,
                       match=f"module 'ufunclab' has no attribute 'aabbcc123'"):
        ufunclab.aabbcc123
