from otter.test_files import test_case

OK_FORMAT = False

name = "q4.1"
points = 10


def test_q4a_public(
        np,
        Dense,
        MLP,
        TanhLayer,
        SoftmaxLayer,
    ):
    # first check that the dense layer's gradients work
    # with customized weight and bias
    _W1 = np.array([[4., 0.5, 0.2, 0.2, 0.8],
                    [1,  1.3,   7, 0.9, 0.7],
                    [2,  2.7,  4.2, 0.5, 0.6]])
    _W2 = np.array([[1, 2, 3, 8],
                    [4, 5, 6, 1],
                    [7, 8, 9, 5],
                    [10, 11, 12, 3],
                    [13, 14, 15, 6]]) / 100
    _input_size = 3
    _output_size = 4
    _hidden_size = _W1.shape[1]
    _bias1 = np.ones((1, _hidden_size))
    _bias2 = np.ones((1, _output_size))
    assert _W1.shape == (_input_size, _hidden_size)
    assert _W2.shape == (_hidden_size, _output_size)

    # now check that the MLP's gradients work
    _layers = [
        Dense(
            _input_size,
            _hidden_size,
            weights=_W1,
            bias=_bias1
        ),
        TanhLayer(),
        Dense(
            _hidden_size,
            _output_size,
            weights=_W2,
            bias=_bias2
        ),
    ]
    _mlp = MLP(_layers)
    _x = np.random.randn(10, _input_size)
    _y = _mlp.forward(_x)
    #assert _y.shape == (10, _hidden_size_2)
    _output_grad = np.random.randn(10, _output_size)
    _mlp.backward(_output_grad)
    _mlp.update(0.01)