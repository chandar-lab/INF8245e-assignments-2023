from otter.test_files import test_case

OK_FORMAT = False

name = "q2.1"
points = 5

@test_case(points=1.0, hidden=False)
def test_q2_1_public_1(np, pd, log_transform):
    data = pd.read_csv("./data/census.csv")
    features_raw = data.drop('income', axis = 1)

    features_log_transformed = log_transform(features_raw)

    expected_features_log_transformed = pd.read_csv("./data/q2_1_public_1.csv")

    skewed = ['capital-gain', 'capital-loss']
    
    np.testing.assert_allclose(features_log_transformed[skewed], expected_features_log_transformed[skewed])

