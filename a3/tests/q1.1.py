from otter.test_files import test_case

OK_FORMAT = False

name = "q1.1"
points = 4

@test_case(points=1.5, hidden=False)
def test_q1_1_public_1(np, pd, data_exploration):
    data = pd.read_csv("./data/census.csv")
    
    n_records, n_greater_50k, n_at_most_50k, greater_percent = data_exploration(data)

    assert n_records == 45222, "`n_records` do not match the expected value"
    assert n_greater_50k == 11208, "`n_greater_50k` do not match the expected value"
    assert n_at_most_50k == 34014, "`n_at_most_50k` do not match the expected value"
    assert np.isclose([greater_percent], [24.78439697492371]), "`greater_percent` do not match the expected value"

