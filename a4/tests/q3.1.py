from otter.test_files import test_case

OK_FORMAT = False

name = "q3.1"
points = 5

@test_case(points=0.5, hidden=False)
def test_q3a_public(np, MLP, Dense, ReLULayer, SoftmaxLayer):
    _layers = [
        Dense(2, 4, weights=np.array([[1, 2, 3, 4], [5, 6, 7, 8]])),
        ReLULayer(),
        Dense(4, 3, weights=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])),
        SoftmaxLayer()
    ]
    _mlp = MLP(_layers)
    _x = np.array(
        [[0.6170757643703014, -0.2206256722803427],
         [0.31815323352996216, 1.14863945157472],
         [0.6591910731621375, 0.34610828317306847],
         [0.8809841030600281, 0.47859171012276397],
         [0.25945569725415996, -0.988257672225727],
         [-1.2277661659039383, 1.1883942574909925],
         [-0.8626005667684048, 1.1732528515599918],
         [-0.18345427326864433, -0.49032797762019953],
         [1.1708488595804007, -0.040961956226513566],
         [0.3772712890700809, 0.3697891293357918]]
    )
    _y_np = np.array(
        [[0.08860130661174538, 0.2432991779906077, 0.6680995153976469],
         [1.979118501560042e-29, 4.448728471777114e-15, 0.9999999999999956],
         [2.871225506062856e-14, 1.6944689491231543e-07, 0.9999998305530765],
         [3.464837654943057e-19, 5.886287159919702e-10, 0.9999999994113713],
         [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
         [6.704429995464882e-17, 8.18805834161459e-09, 0.9999999918119417],
         [9.920118966488803e-20, 3.149622035001079e-10, 0.9999999996850377],
         [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
         [5.69091819972243e-10, 2.3855360901852346e-05, 0.9999761440700063],
         [2.355043207534296e-12, 1.5346138255101459e-06, 0.9999984653838194]]
    )
    _y_mlp = _mlp.forward(_x)
    assert _y_mlp.shape == _y_np.shape
    np.testing.assert_allclose(_y_mlp, _y_np)

