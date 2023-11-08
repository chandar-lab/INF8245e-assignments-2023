from otter.test_files import test_case

OK_FORMAT = False

name = "q2.1"
points = 15

@test_case(points=0.5, hidden=False)
def test_q2a_public(
    np,
    ReLULayer,
    TanhLayer,
    SoftmaxLayer,
):
    # test forward
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
    max_values = np.max(_x, axis=1, keepdims=True)
    exp_values = np.exp(_x - max_values)  # Subtract the maximum for numerical stability
    _y_softmax_layer = np.ones(10).reshape((10, 1))
    np.testing.assert_allclose(_y_relu_layer, _y_relu_np)
    np.testing.assert_allclose(_y_tanh_layer, _y_tanh_np)
    np.testing.assert_allclose(_y_softmax_layer.sum(axis=1), np.ones((10,)))
    np.testing.assert_allclose(_y_softmax_layer, _y_softmax_layer)

    # test backward
    _output_grad = np.linspace(-1, 1, 10).reshape(10, 1)
    _relu_layer.backward(_output_grad)
    _tanh_layer.backward(_output_grad)
    _softmax_layer.backward(_output_grad)
    _np_y_relu_grad = np.array(
        [[-0.0], [-0.0], [-0.0], [-0.0], [-0.0],
         [0.11111111111111116], [0.33333333333333326],
         [0.5555555555555554], [0.7777777777777777], [1.0]]
    )
    _np_y_tanh_grad = np.array(
        [[-0.41997434161402614], [-0.44771983822303985], [-0.41405876165426225],
         [-0.2988765198683048], [-0.10975058057817083],
         [0.10975058057817083], [0.2988765198683047], [0.4140587616542622],
         [0.4477198382230398], [0.41997434161402614]]
    )
    _np_y_softmax_grad = [[-0.0], [-0.0], [-0.0], [-0.0],
                          [-0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
    np.testing.assert_allclose(_relu_layer.input_grad, _np_y_relu_grad)
    np.testing.assert_allclose(_tanh_layer.input_grad, _np_y_tanh_grad)
    np.testing.assert_allclose(_softmax_layer.input_grad, _np_y_softmax_grad)

