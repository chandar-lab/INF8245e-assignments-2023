from otter.test_files import test_case

OK_FORMAT = False

name = "q1.1"
points = 6

@test_case(points=1.0, hidden=False)
def test_q11_public_a(np, get_distance):
    np.testing.assert_allclose(
        get_distance(np.ones((10, 4)), np.ones((5, 4))),
        np.zeros((10, 5)))

@test_case(points=1.0, hidden=False)
def test_q11_public_b(np, get_distance):
    _Xa = np.ones((5,3))
    np.testing.assert_allclose(
        get_distance(_Xa, _Xa), np.zeros((5, 5)))

