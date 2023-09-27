from otter.test_files import test_case

OK_FORMAT = False

name = "q3.2"
points = 2

@test_case(points=0.5, hidden=False)
def test_q32_public(np, softmax):
    _X = np.array([[1, 0.5, 0.2, 3],
                [1,  -1,   7, 3],
                [2,  12,  13, 3]])
    _smx = softmax(_X)
    np.testing.assert_allclose(_smx.sum(axis=1), [1,1,1])

