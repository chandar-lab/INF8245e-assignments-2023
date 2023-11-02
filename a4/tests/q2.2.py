from otter.test_files import test_case

OK_FORMAT = False

name = "q2.2"
points = 10


@test_case(points=1.5, hidden=False)
def test_q2b_public(
    np,
    CrossEntropyLossLayer,
):
    _cross_entropy_loss_layer = CrossEntropyLossLayer()
    _x = np.array([[0.5, 0.5]])
    _y = np.array([[1.0, 0.0]])
    _loss = _cross_entropy_loss_layer.forward(_x, _y)
    _loss_np = 0.6931471805599433
    np.testing.assert_allclose(_loss, _loss_np)