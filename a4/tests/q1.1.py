from otter.test_files import test_case

OK_FORMAT = False

name = "q1.1"
points = 10


@test_case(points=1.5, hidden=False)
def test_q1_1_public_1(np, pd, output_data_shape):
    assert output_data_shape == (1,2), "output data shape do not match the expected value"