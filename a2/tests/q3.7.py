from otter.test_files import test_case

OK_FORMAT = False

name = "q3.7"
points = 10

@test_case(points=4.0, hidden=False)
def test_q37_public(np, LogisticRegressionModel, train_one_epoch):
    X = np.array([[1, 2],[1, 1]])
    W = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    b = np.array([1.0,1.0,1.0])
    target = np.array([2,0])

    batch_size = 1
    model = LogisticRegressionModel(W)
    model.b = b.copy()
    learning_rate = 0.5
    validation_every_x_step = 1

    train_losses, train_accuracies, train_steps, \
        val_losses, val_accuracies, val_steps = \
            train_one_epoch(model, X, target, X, target, batch_size, learning_rate,
                            validation_every_x_step)

    np.testing.assert_allclose(len(train_losses), 2)
    np.testing.assert_allclose(len(train_accuracies), 2)
    np.testing.assert_allclose(train_steps, [1,2])

