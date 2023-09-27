from otter.test_files import test_case

OK_FORMAT = False

name = "q3.5"
points = 6

@test_case(points=2.0, hidden=False)
def test_q35_public(np, compute_gradients):
    X = np.array([[1, 2],[1, 1]])
    prediction = np.array([[0.3,0.2,0.5], [0.5,0.1,0.4]])
    target = np.array([2,0])
    received_grad_w, received_grad_b = compute_gradients(X, prediction,target)
    np.testing.assert_allclose(received_grad_w.shape, [2,3])
    np.testing.assert_allclose(received_grad_b.shape, [3])

