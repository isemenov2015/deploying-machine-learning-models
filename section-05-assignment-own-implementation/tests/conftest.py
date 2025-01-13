import pytest

from regression_model.processing.data_manager import load_dataset


@pytest.fixture()
def sample_input_data():
    return load_dataset()
