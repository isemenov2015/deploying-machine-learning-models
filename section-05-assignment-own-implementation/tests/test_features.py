
def test_dataset_validation(sample_input_data):
    # Given
    assert len(sample_input_data) == 1309

    # When
    subject = sample_input_data.iloc[0]

    # Then
    assert subject["age"] == 29.0
