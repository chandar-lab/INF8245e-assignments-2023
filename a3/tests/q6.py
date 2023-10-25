from otter.test_files import test_case

OK_FORMAT = False

name = "q6"
points = 20

@test_case(points=7.0, hidden=False)
def test_q6_1_public_1(np, pd, naive_predictor):
    data = pd.read_csv("./data/census.csv")
    income_raw = data["income"]
    income = income_raw.apply(lambda x: 1 if x == ">50K" else 0)

    fscore = naive_predictor(income)
    
    np.testing.assert_allclose([fscore], [0.29172913543228385])

