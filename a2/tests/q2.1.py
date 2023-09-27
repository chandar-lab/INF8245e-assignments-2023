from otter.test_files import test_case

OK_FORMAT = False

name = "q2.1"
points = 10

@test_case(points=3.0, hidden=False)
def test_q21_public(np, gnb_fit_classifier):
    X = np.array([[1,2,3,4],
                [3,2,1,4],
                [2,3,1,3],
                [4,3,1,3]])
    y = np.array([1,1,0,0])

    prior_probs, means, vars = gnb_fit_classifier(X,y)
    
    np.testing.assert_allclose(
        prior_probs,
        [0.5,0.5])

    np.testing.assert_allclose(
        means,
        [np.array([3., 3., 1., 3.]), np.array([2., 2., 2., 4.])])
    np.testing.assert_allclose(
        vars,
        [np.array([1.001, 0.001, 0.001, 0.001]), np.array([1.001, 0.001, 1.001, 0.001])])

