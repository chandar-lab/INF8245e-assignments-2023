from otter.test_files import test_case

OK_FORMAT = False

name = "q3.4"
points = 3

@test_case(points=1.0, hidden=False)
def test_q34_public(np, nll_loss):
    prediction = np.array([[0.3,0.7], [0.4,0.6]])
    target = np.array([1,0])
    np.testing.assert_allclose(nll_loss(prediction,target), 0.6364828379064438)

