from otter.test_files import test_case

OK_FORMAT = False

name = "q1.2"
points = 16

@test_case(points=4.0, hidden=False)
def test_q12_public(np, get_k_neighbors):
    _x_labels = np.array([1, 2, 1, 3, 3])
    _euclid_dists = np.array([[1, 2], [3, 1], [2, 0] , [4,5], [1,0]])
    _nearest_labels = get_k_neighbors(_euclid_dists, _x_labels, 3)
    np.testing.assert_allclose(_nearest_labels.shape, [3,2])
    nl0 = _nearest_labels[:,0]
    nl1 = _nearest_labels[:,1]
    nl0.sort()
    nl1.sort()
    np.testing.assert_allclose(nl0, np.array([1,1,3]))
    np.testing.assert_allclose(nl1, np.array([1,2,3]))

