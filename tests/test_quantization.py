import pytest

from quantization.quantize import verify_artifact


@pytest.mark.skip(reason="Task 8: not implemented")
def test_verify_artifact_rejects_missing(tmp_path):
    assert not verify_artifact(tmp_path)
