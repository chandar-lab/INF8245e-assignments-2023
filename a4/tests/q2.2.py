from otter.test_files import test_case

OK_FORMAT = False

name = "q2.2"
points = 10


def test_q2b_public(
    np,
    CrossEntropyLossLayer,
):
    _cross_entropy_loss_layer = CrossEntropyLossLayer()
    _x = np.array([[0.5, 0.5]])
    _y = np.array([[1, 0]])
    _loss = _cross_entropy_loss_layer.forward(_x, _y)
    _loss_np = 1.386294
    np.testing.assert_allclose(_loss, _loss_np, rtol=1e-3)