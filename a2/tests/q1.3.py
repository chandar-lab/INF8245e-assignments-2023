from otter.test_files import test_case

OK_FORMAT = False

name = "q1.3"
points = 6

@test_case(points=1, hidden=False)
def test_q13_public(np, get_prediction):
    _temp_labels = np.asarray([ [4, 1, 4, 6, 4], [7, 2, 1, 5, 4], [2, 2, 5, 6, 1] ], dtype=np.uint8)
    np.testing.assert_allclose(
        get_prediction(_temp_labels),
        np.asarray([2, 2, 1, 6, 4], dtype=np.uint8))

