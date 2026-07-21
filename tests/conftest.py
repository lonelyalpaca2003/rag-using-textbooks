from unittest.mock import MagicMock, patch
_mock_engine = MagicMock()
_mock_index = MagicMock()
_engine_patch = patch(
    "src.indexing.create_query_engine",
    return_value=(_mock_engine, _mock_index),
)
_engine_patch.start()

import pytest


@pytest.fixture
def mock_engine():
    _mock_engine.reset_mock()
    return _mock_engine


@pytest.fixture
def mock_index():
    _mock_index.reset_mock()
    return _mock_index
