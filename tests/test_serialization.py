from prodml.export import export_and_verify


def test_onnx_parity_within_tolerance(predictor, train_frame):
    """Pickle and ONNX predictions must agree within the project's 1e-4 tolerance."""
    max_difference = export_and_verify(predictor, train_frame, n_rows=len(train_frame))
    assert max_difference < 1e-4
