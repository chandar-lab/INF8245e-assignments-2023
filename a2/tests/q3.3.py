from otter.test_files import test_case

OK_FORMAT = False

name = "q3.3"
points = 3

@test_case(points=0.5, hidden=False)
def test_q33_public(np, LogisticRegressionModel):
    _X = np.array([[1, 2],[1, 1], [2, 1], [2, 2]])
    _W = np.array([[1, 2, 3], [1, 2, 3]])
    _b = np.array([1,1,1])
    LR_model = LogisticRegressionModel(_W)
    LR_model.b = _b
    np.testing.assert_allclose(LR_model(_X).sum(axis=1), [1,1,1,1])

