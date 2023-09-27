from otter.test_files import test_case

OK_FORMAT = False

name = "q3.6"
points = 6

@test_case(points=2.0, hidden=False)
def test_q36_pubic(np, LogisticRegressionModel, validation):
    X = np.array([[1, 2],[1, 1]])
    W = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    b = np.array([1.0,1.0,1.0])
    target = np.array([2,0])

    batch_size = 1
    model = LogisticRegressionModel(W)
    model.b = b.copy()

    validation_loss, validation_accuracy = validation(model,X,target,batch_size)

    np.testing.assert_array_less(0,validation_loss)
    np.testing.assert_array_less(validation_accuracy,1.0)

