from otter.test_files import test_case

OK_FORMAT = False

name = "q1.1"
points = 10

@test_case(points=0.5, hidden=False)
def test_2a1_public(np, create_inputs_with_bias):
    np.testing.assert_allclose(
        create_inputs_with_bias(
            np.asarray([[22], [23], [35]], dtype=np.float64)
        ),
        np.asarray([[22, 1], [23, 1], [35, 1]]), atol=1e-6
    )

@test_case(points=0.5, hidden=False)
def test_2a2_public(np, rmse):
    np.testing.assert_allclose(
        rmse(
            np.asarray([0, 0, 0], dtype=np.float64),
            np.asarray([1, 1, 1], dtype=np.float64),
        ),
        1, atol=1e-6
    )

@test_case(points=0.5, hidden=False)
def test_2a3_public(np, predict_linear_regression):
    np.testing.assert_allclose(
        predict_linear_regression(
            np.asarray([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float64),
            np.asarray([0, 0, 0], dtype=np.float64),
        ),
        np.asarray([0, 0, 0], dtype=np.float64), atol=1e-6
    )

@test_case(points=0.5, hidden=False)
def test_2a4_public(np, solve_linear_regression):
    np.testing.assert_allclose(
        solve_linear_regression(
            np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
            np.asarray([0, 0, 0], dtype=np.float64),
        ),
        np.asarray([0, 0, 0], dtype=np.float64), atol=1e-6
    )
        
@test_case(points=0.5, hidden=False)
def test_2a5_public(np, solve_ridge_regression):
    np.testing.assert_allclose(
        solve_ridge_regression(
            np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
            np.asarray([0, 0, 0], dtype=np.float64),
            0.0
        ),
        np.asarray([0, 0, 0], dtype=np.float64), atol=1e-6
    )

