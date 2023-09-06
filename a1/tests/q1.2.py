from otter.test_files import test_case

OK_FORMAT = False

name = "q1.2"
points = 10

@test_case(points=1, hidden=False)
def test_2b_public(np, cross_validation_linear_regression):
    outputs = cross_validation_linear_regression(
        k_folds=2, hyperparameters=[0],
        X=np.asarray([[1, 0], [0, 1], [1, 1], [2, 4]], dtype=np.float64),
        y=np.asarray([0, 0, 0, 0], dtype=np.float64),
    )

    np.testing.assert_allclose(
        outputs[0], 0.0, atol=1e-6
    )

    np.testing.assert_allclose(
        outputs[1], 0.0, atol=1e-6
    )

    np.testing.assert_allclose(
        outputs[2], np.asarray([0.0], dtype=np.float64), atol=1e-6
    )

