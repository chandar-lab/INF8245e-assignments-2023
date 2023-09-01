from otter.test_files import test_case

OK_FORMAT = False

name = "q2.1"
points = 6

@test_case(points=0.5, hidden=False)
def test_3a1_public(np, linear_regression_gradient):

    np.testing.assert_allclose(
        linear_regression_gradient(
            X=np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
            y=np.asarray([0, 0, 0], dtype=np.float64), w=np.asarray([0, 0, 0], dtype=np.float64)
        ),
        np.asarray([0, 0, 0], dtype=np.float64),
    )

@test_case(points=0.5, hidden=False)
def test_3a2_public(np, ridge_regression_gradient):

    np.testing.assert_allclose(
        ridge_regression_gradient(
            X=np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
            y=np.asarray([0, 0, 0], dtype=np.float64), w=np.asarray([0, 0, 0], dtype=np.float64),
            hyperparameter=0.0
        ),
        np.asarray([0, 0, 0], dtype=np.float64),
    )

@test_case(points=0.5, hidden=False)
def test_3a3_public(np, lasso_regression_gradient):

    np.testing.assert_allclose(
        lasso_regression_gradient(
            X=np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
            y=np.asarray([0, 0, 0], dtype=np.float64), w=np.asarray([0, 0, 0], dtype=np.float64),
            hyperparameter=0.0
        ),
        np.asarray([0, 0, 0], dtype=np.float64),
    )

