import pytest

from training.prepare_data import build_examples


@pytest.mark.skip(reason="Task 7: not implemented")
def test_build_examples_excludes_heldout():
    build_examples([])
