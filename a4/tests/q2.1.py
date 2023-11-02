from otter.test_files import test_case

OK_FORMAT = False

name = "q2.1"
points = 10


@test_case(points=1.5, hidden=False)
def test_q2a_public(
    np,
    ReLULayer,
    TanhLayer,
    SoftmaxLayer,
):
    _relu_layer = ReLULayer()
    _tanh_layer = TanhLayer()
    _softmax_layer = SoftmaxLayer()
    _x = np.linspace(-1, 1, 10).reshape(10, 1)
    _y_relu_layer = _relu_layer.forward(_x)
    _y_tanh_layer = _tanh_layer.forward(_x)
    _y_softmax_layer = _softmax_layer.forward(_x)
    _y_relu_np = np.array(
        [[0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.11111111111111116],
        [0.33333333333333326],
        [0.5555555555555554],
        [0.7777777777777777],
        [1.0]]
    )
    _y_tanh_np = np.array(
       [[-0.76159416],
       [-0.65142936],
       [-0.5046724 ],
       [-0.32151274],
       [-0.11065611],
       [ 0.11065611],
       [ 0.32151274],
       [ 0.5046724 ],
       [ 0.65142936],
       [ 0.76159416]]
    )
    np.testing.assert_allclose(_y_relu_layer, _y_relu_np)
    np.testing.assert_allclose(_y_tanh_layer, _y_tanh_np)
    np.testing.assert_allclose(_y_softmax_layer.sum(axis=1), np.ones(10))