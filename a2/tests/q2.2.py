from otter.test_files import test_case

OK_FORMAT = False

name = "q2.2"
points = 10

@test_case(points=3.0, hidden=False)
def test_q22_public(np, gnb_predict):
    X = np.array([[1,2,3,4],
                [1,2,1,4],
                [2,3,3,3],
                [2,3,1,3]])
    y = np.array([1,1,0,0])

    prior_probs = [0.5,0.5]
    means = [np.array([2,3,2,3]), np.array([1,2,2,4])]
    vars = [np.array([0.001, 0.001, 1.001, 0.001]), np.array([0.001, 0.001, 1.001, 0.001])]
    num_classes = 2
    preds = gnb_predict(X,prior_probs,means,vars,num_classes)
    np.testing.assert_allclose(preds, y)

