from otter.test_files import test_case

OK_FORMAT = False

name = "q1.1"
points = 10


@test_case(points=1.5, hidden=False)
def test_q1_public(np, Dense):
    _W = np.array(
        [[4., 0.5, 0.2, 0.2],
         [1,  1.3,   7, 0.9],
         [2,  2.7,  4.2, 0.5]]
    )
    _input_size = 3
    _output_size = 4
    _bias = np.ones((1, _output_size))
    _X = np.array(
        [[1, 2, 3],
         [4, 5, 6]]
    )
    _Y_np = np.array(
        [[13. , 12.2, 27.8,  4.5],
         [34. , 25.7, 62. ,  9.3]]
    )
    _dense_layer = Dense(
        _input_size,
        _output_size,
        weights=_W,
        bias=_bias
    )
    _y_dense_layer = _dense_layer.forward(_X)
    np.testing.assert_allclose(_y_dense_layer, _Y_np, atol=1e-6)
    assert _dense_layer.weights.shape == (_input_size, _output_size)
    assert _dense_layer.bias.shape == (1, _output_size)
    assert _y_dense_layer.shape == _Y_np.shape