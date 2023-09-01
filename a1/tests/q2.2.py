from otter.test_files import test_case

OK_FORMAT = False

name = "q2.2"
points = 6

@test_case(points=0.25, hidden=False)
def test_3b1_public(np, regression_gradient_descent):

    out = regression_gradient_descent(
            X_train=np.asarray([[1, 0], [0, 1]], dtype=np.float64),
            y_train=np.asarray([[1], [1]], dtype=np.float64),
            X_test=np.asarray([[1, 0], [0, 1]], dtype=np.float64),
            y_test=np.asarray([[1], [1]], dtype=np.float64),
            w_0=np.asarray([[1], [1]], dtype=np.float64),
            hyperparameter=0.0,
            learning_rate=0.1, num_epochs=1, reg_type='linear'
    )

    np.testing.assert_allclose(
        out[0],
        np.asarray([0], dtype=np.float32)
    )

    np.testing.assert_allclose(
        out[1],
        np.asarray([0], dtype=np.float32)
    )

    np.testing.assert_allclose(
        out[2],
        np.asarray([[1.], [1.]], dtype=np.float32)
    )

@test_case(points=0.25, hidden=False)

def test_3b2_public(np, regression_gradient_descent):

    out = regression_gradient_descent(
            X_train=np.asarray([[1, 0], [0, 1]], dtype=np.float64),
            y_train=np.asarray([[1], [1]], dtype=np.float64),
            X_test=np.asarray([[1, 0], [0, 1]], dtype=np.float64),
            y_test=np.asarray([[1], [1]], dtype=np.float64),
            w_0=np.asarray([[1], [1]], dtype=np.float64),
            hyperparameter=0.5,
            learning_rate=0.1, num_epochs=1, reg_type='ridge'
    )

    np.testing.assert_allclose(
        out[0],
        np.asarray([0], dtype=np.float32)
    )

    np.testing.assert_allclose(
        out[1],
        np.asarray([0], dtype=np.float32)
    )

    np.testing.assert_allclose(
        out[2],
        np.asarray([[0.9], [0.9]], dtype=np.float32)
    )

@test_case(points=0.25, hidden=False)

def test_3b3_public(np, regression_gradient_descent):

    out = regression_gradient_descent(
            X_train=np.asarray([[1, 0], [0, 1]], dtype=np.float64),
            y_train=np.asarray([[1], [1]], dtype=np.float64),
            X_test=np.asarray([[1, 0], [0, 1]], dtype=np.float64),
            y_test=np.asarray([[1], [1]], dtype=np.float64),
            w_0=np.asarray([[1], [1]], dtype=np.float64),
            hyperparameter=0.5,
            learning_rate=0.1, num_epochs=1, reg_type='lasso'
    )

    np.testing.assert_allclose(
        out[0],
        np.asarray([0], dtype=np.float32)
    )

    np.testing.assert_allclose(
        out[1],
        np.asarray([0], dtype=np.float32)
    )

    np.testing.assert_allclose(
        out[2],
        np.asarray([[0.9], [0.9]], dtype=np.float32)
    )

